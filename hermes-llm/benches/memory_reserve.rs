//! Memory-reserve routing benchmark across dormant, one-active and two-active
//! states.
//!
//! The compact default model runs on every backend. To measure the production
//! sleep model on CUDA:
//!
//! ```text
//! cargo bench -p hermes-llm --bench memory_reserve --features training-fusion -- \
//!   --model hermes-mal/well-known/retriever_300m_moe_sleep.mal --tokens 8192
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use hermes_llm::{Transformer, default_device, parse_mal, parse_mal_file};
use serde::Serialize;

const COMPACT_MEMORY_MODEL: &str = r#"
ffn base { hidden_dim: 16 activation: swiglu bias: false }
memory reserve {
    tier fast {
        ffn: base
        reserve_experts { capacity: 4 rank: 32 top_k: 1 }
    }
}
model memory-reserve-bench {
    vocab_size: 128 max_seq_len: 1 hidden_size: 128 num_layers: 1
    block: { attention: { num_heads: 4 } memory: reserve }
    embeddings { tie_weights: true }
}
"#;

#[derive(Debug)]
struct Args {
    model: Option<PathBuf>,
    tokens: usize,
    warmup: usize,
    iterations: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: None,
            tokens: 4_096,
            warmup: 5,
            iterations: 20,
        }
    }
}

#[derive(Serialize)]
struct Measurement {
    active_slots_per_layer: usize,
    median_ms: f64,
    min_ms: f64,
    tokens_per_second: f64,
}

#[derive(Serialize)]
struct Report {
    implementation: &'static str,
    model: String,
    device: String,
    tokens: usize,
    warmup: usize,
    iterations: usize,
    measurements: Vec<Measurement>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let config = match &args.model {
        Some(path) => {
            parse_mal_file(path).with_context(|| format!("failed to parse {}", path.display()))?
        }
        None => parse_mal(COMPACT_MEMORY_MODEL).context("compact benchmark MAL is invalid")?,
    };
    let device = default_device();
    device.seed(17);
    let mut model = Transformer::new(&config, &device)?;
    let input = Tensor::<2, Int>::zeros([args.tokens, 1], &device);
    // Inputs are independent one-token rows; row-major end positions are
    // therefore 0..batch, which exercises the router with `tokens` routes
    // without adding a quadratic attention sequence to this microbenchmark.
    let end_positions = Tensor::<1, Int>::arange(0..args.tokens as i64, &device);

    let mut measurements = Vec::with_capacity(3);
    measurements.push(measure(
        &model,
        input.clone(),
        end_positions.clone(),
        &device,
        &args,
        0,
    )?);
    model.activate_memory_slot_all_layers(0, 0)?;
    measurements.push(measure(
        &model,
        input.clone(),
        end_positions.clone(),
        &device,
        &args,
        1,
    )?);
    model.activate_memory_slot_all_layers(0, 1)?;
    measurements.push(measure(&model, input, end_positions, &device, &args, 2)?);

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            implementation: "fixed_width_cached_reserve_router",
            model: config.name,
            device: format!("{device:?}"),
            tokens: args.tokens,
            warmup: args.warmup,
            iterations: args.iterations,
            measurements,
        })?
    );
    Ok(())
}

fn measure(
    model: &Transformer,
    input: Tensor<2, Int>,
    end_positions: Tensor<1, Int>,
    device: &Device,
    args: &Args,
    active_slots_per_layer: usize,
) -> Result<Measurement> {
    let operation = || {
        let output = model.forward_embeddings(input.clone(), end_positions.clone(), None);
        std::hint::black_box(output);
    };
    for _ in 0..args.warmup {
        operation();
        device.sync()?;
    }
    let mut elapsed = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        device.sync()?;
        let started = Instant::now();
        operation();
        device.sync()?;
        elapsed.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    elapsed.sort_by(f64::total_cmp);
    let median_ms = if args.iterations.is_multiple_of(2) {
        (elapsed[args.iterations / 2 - 1] + elapsed[args.iterations / 2]) / 2.0
    } else {
        elapsed[args.iterations / 2]
    };
    Ok(Measurement {
        active_slots_per_layer,
        median_ms,
        min_ms: elapsed[0],
        tokens_per_second: args.tokens as f64 / (median_ms / 1_000.0),
    })
}

fn parse_args() -> Result<Args> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => parsed.model = Some(PathBuf::from(value()?)),
            "--tokens" => parsed.tokens = value()?.parse().context("invalid --tokens")?,
            "--warmup" => parsed.warmup = value()?.parse().context("invalid --warmup")?,
            "--iterations" => {
                parsed.iterations = value()?.parse().context("invalid --iterations")?
            }
            "--bench" => {}
            "--help" | "-h" => {
                println!(
                    "Usage: memory_reserve [--model PATH] [--tokens N] [--warmup N] [--iterations N]"
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument {flag:?}"),
        }
    }
    if parsed.tokens == 0 || parsed.iterations == 0 {
        bail!("--tokens and --iterations must be positive");
    }
    Ok(parsed)
}
