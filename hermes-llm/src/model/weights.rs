//! Safetensors checkpoints shared by training and inference.

use std::path::Path;

use anyhow::{Context, Result};
use burn_store::{ApplyResult, ModuleSnapshot, SafetensorsStore};

use crate::mal::{MemoryTierInit, ModelDef};

use super::Transformer;

/// Load strictly: missing, unexpected, and shape-mismatched tensors are errors.
pub fn load_safetensors(model: &mut Transformer, path: impl AsRef<Path>) -> Result<ApplyResult> {
    let path = path.as_ref();
    let mut store = SafetensorsStore::from_file(path).skip_enum_variants(true);
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("failed to load weights from {}", path.display()))?;
    model.sync_memory_state();
    Ok(result)
}

pub fn save_safetensors(model: &Transformer, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut store = SafetensorsStore::from_file(path).skip_enum_variants(true);
    model
        .save_into(&mut store)
        .with_context(|| format!("failed to save weights to {}", path.display()))
}

/// Upgrade a conventional checkpoint into an explicitly memory-enabled model.
///
/// Ordinary FFN tensors are remapped into tier zero. All later tiers must use
/// `residual_init: zero`, and every preallocated reserve mask must still be
/// dormant, which makes the upgraded model logit-equivalent to the source.
/// The target topology is never inferred implicitly from checkpoint keys.
pub fn upgrade_safetensors_to_memory(
    target: &mut Transformer,
    source_config: &ModelDef,
    source_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<ApplyResult> {
    validate_upgrade_topology(source_config, target.config())?;
    target.prepare_memory_upgrade_state();
    let source_path = source_path.as_ref();
    let output_path = output_path.as_ref();
    let mut store = SafetensorsStore::from_file(source_path)
        .skip_enum_variants(true)
        .allow_partial(true);
    for layer in 0..source_config.num_layers {
        if target.config().block_for_layer(layer).memory.is_some() {
            store = store.with_key_remapping(
                format!(r"^layers\.{layer}\.feed_forward\."),
                format!("layers.{layer}.memory.tiers.0.feed_forward."),
            );
        }
    }
    let result = target.load_from(&mut store).with_context(|| {
        format!(
            "failed to map conventional checkpoint {} into memory topology",
            source_path.display()
        )
    })?;
    if !result.errors.is_empty() || !result.unused.is_empty() {
        anyhow::bail!("memory checkpoint upgrade did not map cleanly:\n{result}");
    }
    let unexpected_missing = result
        .missing
        .iter()
        .filter(|(path, _)| !is_expected_upgrade_tensor(path))
        .map(|(path, _)| path.as_str())
        .collect::<Vec<_>>();
    if !unexpected_missing.is_empty() {
        anyhow::bail!(
            "memory checkpoint upgrade is missing source tensors outside new memory state: {}",
            unexpected_missing.join(", ")
        );
    }
    target.sync_memory_state();
    if target
        .memory_slot_statuses()
        .iter()
        .any(|status| status.active)
    {
        anyhow::bail!("new memory reserve slots must be dormant after upgrade");
    }
    save_safetensors(target, output_path)?;
    Ok(result)
}

/// An ordinary checkpoint can omit only tensors introduced by the target
/// memory topology. Tier zero's base FFN is *not* new: every one of those
/// tensors must have been remapped from the source checkpoint. Keeping this
/// predicate structural prevents a damaged source checkpoint from being
/// accepted merely because the remapped key happens to contain `.memory.`.
fn is_expected_upgrade_tensor(path: &str) -> bool {
    let mut segments = path.split('.');
    if segments.next() != Some("layers")
        || segments
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .is_none()
        || segments.next() != Some("memory")
        || segments.next() != Some("tiers")
    {
        return false;
    }
    let Some(tier) = segments
        .next()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return false;
    };
    if tier > 0 {
        return segments.next().is_some();
    }
    segments.next() == Some("reserve") && segments.next().is_some()
}

fn validate_upgrade_topology(source: &ModelDef, target: &ModelDef) -> Result<()> {
    for (name, matches) in [
        ("vocab_size", source.vocab_size == target.vocab_size),
        ("max_seq_len", source.max_seq_len == target.max_seq_len),
        ("hidden_size", source.hidden_size == target.hidden_size),
        ("num_layers", source.num_layers == target.num_layers),
        (
            "embeddings",
            serde_json::to_value(&source.embeddings)? == serde_json::to_value(&target.embeddings)?,
        ),
        (
            "output",
            serde_json::to_value(&source.output)? == serde_json::to_value(&target.output)?,
        ),
    ] {
        if !matches {
            anyhow::bail!("memory upgrade requires identical source/target {name}");
        }
    }

    let mut memory_layers = 0;
    for layer in 0..source.num_layers {
        let source_block = source.block_for_layer(layer);
        let target_block = target.block_for_layer(layer);
        if source_block.memory.is_some() {
            anyhow::bail!("source layer {layer} is already memory-enabled");
        }
        for (name, matches) in [
            (
                "attention",
                serde_json::to_value(&source_block.attention)?
                    == serde_json::to_value(&target_block.attention)?,
            ),
            (
                "ssm",
                serde_json::to_value(&source_block.ssm)?
                    == serde_json::to_value(&target_block.ssm)?,
            ),
            (
                "norm",
                serde_json::to_value(&source_block.norm)?
                    == serde_json::to_value(&target_block.norm)?,
            ),
            (
                "norm_position",
                source_block.norm_position == target_block.norm_position,
            ),
            ("residual", source_block.residual == target_block.residual),
            ("dropout", source_block.dropout == target_block.dropout),
        ] {
            if !matches {
                anyhow::bail!("memory upgrade layer {layer} changes {name}");
            }
        }
        match &target_block.memory {
            Some(memory) => {
                memory_layers += 1;
                let fast = memory.tiers.first().ok_or_else(|| {
                    anyhow::anyhow!("target layer {layer} memory has no fast tier")
                })?;
                if serde_json::to_value(&fast.ffn)? != serde_json::to_value(&source_block.ffn)? {
                    anyhow::bail!("target layer {layer} fast tier does not match source FFN");
                }
                if memory
                    .tiers
                    .iter()
                    .skip(1)
                    .any(|tier| tier.residual_init != MemoryTierInit::ResidualZero)
                {
                    anyhow::bail!("target layer {layer} slower tiers must use residual_init: zero");
                }
            }
            None => {
                if serde_json::to_value(&source_block.ffn)?
                    != serde_json::to_value(&target_block.ffn)?
                {
                    anyhow::bail!("target layer {layer} changes its ordinary FFN");
                }
            }
        }
    }
    if memory_layers == 0 {
        anyhow::bail!("target model does not contain a memory chain");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use tempfile::tempdir;

    use super::*;
    use crate::model::Device;

    #[test]
    fn conventional_checkpoint_upgrade_preserves_logits() {
        let source_config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: base
                norm: rmsnorm { eps: 1e-5 }
            }
            model source {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let target_config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            ffn slow { hidden_dim: 8 activation: swiglu bias: false }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                }
                tier slow {
                    ffn: slow
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                    residual_init: zero
                }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                memory: cms
                norm: rmsnorm { eps: 1e-5 }
            }
            model target {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(71);
        let source = Transformer::new(&source_config, &device).unwrap();
        device.seed(99);
        let mut target = Transformer::new(&target_config, &device).unwrap();
        let directory = tempdir().unwrap();
        let source_path = directory.path().join("source.safetensors");
        let target_path = directory.path().join("target.safetensors");
        let active_target_path = directory.path().join("target-active.safetensors");
        save_safetensors(&source, &source_path).unwrap();
        let upgrade =
            upgrade_safetensors_to_memory(&mut target, &source_config, &source_path, &target_path)
                .unwrap();
        assert!(
            upgrade
                .missing
                .iter()
                .all(|(path, _)| is_expected_upgrade_tensor(path)),
            "upgrade accepted an unexpected missing tensor: {upgrade}"
        );

        let input = Tensor::<2, Int>::from_data([[1, 3, 5, 7]], &device);
        let source_logits = source.forward(input.clone(), 0).into_data();
        let target_logits = target.forward(input, 0).into_data();
        let source_values = source_logits.convert::<f32>().to_vec::<f32>().unwrap();
        let target_values = target_logits.convert::<f32>().to_vec::<f32>().unwrap();
        let maximum = source_values
            .iter()
            .zip(target_values)
            .map(|(source, target)| (source - target).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum < 1e-6, "upgraded logits differ by {maximum}");
        assert!(
            target
                .memory_slot_statuses()
                .iter()
                .all(|slot| !slot.active)
        );

        target.activate_memory_slot(0, 1, 0).unwrap();
        save_safetensors(&target, &active_target_path).unwrap();
        let mut reloaded = Transformer::new(&target_config, &device).unwrap();
        load_safetensors(&mut reloaded, &active_target_path).unwrap();
        let statuses = reloaded.memory_slot_statuses();
        assert!(statuses.iter().any(|slot| slot.tier == 1 && slot.active));
        assert!(statuses.iter().all(|slot| slot.tier == 1 || !slot.active));
    }

    #[test]
    fn upgrade_missing_tensor_allowlist_is_structural() {
        assert!(is_expected_upgrade_tensor(
            "layers.0.memory.tiers.0.reserve.slots.1.a"
        ));
        assert!(is_expected_upgrade_tensor(
            "layers.17.memory.tiers.2.feed_forward.down_proj.weight"
        ));
        assert!(!is_expected_upgrade_tensor(
            "layers.0.memory.tiers.0.feed_forward.down_proj.weight"
        ));
        assert!(!is_expected_upgrade_tensor(
            "layers.0.memoryish.tiers.1.feed_forward.down_proj.weight"
        ));
        assert!(!is_expected_upgrade_tensor("embedding.weight"));
    }

    #[test]
    fn moe_checkpoint_upgrade_is_reloadable_and_logit_exact() {
        let source_config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12 activation: swiglu bias: false
                moe { experts: 3 top_k: 2 }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: routed
                norm: rmsnorm { eps: 1e-5 }
            }
            model source {
                vocab_size: 29 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let target_config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12 activation: swiglu bias: false
                moe { experts: 3 top_k: 2 }
            }
            ffn slow { hidden_dim: 4 activation: swiglu bias: false }
            memory cms {
                tier fast {
                    ffn: routed
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                }
                tier slow {
                    ffn: slow
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                    residual_init: zero
                }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                memory: cms
                norm: rmsnorm { eps: 1e-5 }
            }
            model target {
                vocab_size: 29 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(113);
        let source = Transformer::new(&source_config, &device).unwrap();
        device.seed(127);
        let mut upgraded = Transformer::new(&target_config, &device).unwrap();
        let directory = tempdir().unwrap();
        let source_path = directory.path().join("source-moe.safetensors");
        let target_path = directory.path().join("target-moe.safetensors");
        save_safetensors(&source, &source_path).unwrap();
        upgrade_safetensors_to_memory(&mut upgraded, &source_config, &source_path, &target_path)
            .unwrap();

        let mut reloaded = Transformer::new(&target_config, &device).unwrap();
        load_safetensors(&mut reloaded, &target_path).unwrap();
        let input = Tensor::<2, Int>::from_data([[2, 4, 6, 8]], &device);
        let source_values = source
            .forward(input.clone(), 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let target_values = reloaded
            .forward(input, 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let maximum = source_values
            .iter()
            .zip(target_values)
            .map(|(source, target)| (source - target).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum < 1e-6, "reloaded MoE logits differ by {maximum}");
    }
}
