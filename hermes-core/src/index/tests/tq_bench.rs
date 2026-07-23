//! TQ vs IVF-PQ vs flat brute-force benchmark (docs/turboquant-quantization.md).
//!
//! Run in release mode; results are meaningless in debug builds:
//! ```bash
//! cargo test --release -p hermes-core --features native \
//!     tq_ivf_pq_flat_benchmark -- --ignored --nocapture
//! ```
//!
//! Clustered unit-norm corpus, queries perturbed from corpus points. Ground
//! truth is the flat index's own exact cosine top-k. Reports recall@k after
//! re-rank, p50/p95 end-to-end query latency, .vectors bytes, and build/train
//! wall time per method.

use crate::directories::MmapDirectory;
use crate::dsl::{DenseVectorConfig, Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::DenseVectorQuery;

const DIM: usize = 768; // pads to 1024 for TQ: the unfavorable case
const DOCS: usize = 100_000;
const CLUSTERS: usize = 256;
const QUERIES: usize = 100;
const K: usize = 10;
const IVF_NUM_CLUSTERS: usize = 1_024;
const NPROBE: usize = 64;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn gaussian_unit(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut values: Vec<f32> = (0..dim)
        .map(|_| {
            let a = (splitmix(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
            let b = (splitmix(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * (1.0 - a).max(f64::MIN_POSITIVE).ln()).sqrt()
                * (2.0 * std::f64::consts::PI * b).cos()) as f32
        })
        .collect();
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    values.iter_mut().for_each(|v| *v /= norm);
    values
}

fn normalize(mut values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    values.iter_mut().for_each(|v| *v /= norm);
    values
}

fn build_corpus() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let centers: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|c| gaussian_unit(DIM, 1_000 + c as u64))
        .collect();
    let corpus: Vec<Vec<f32>> = (0..DOCS)
        .map(|i| {
            let center = &centers[i % CLUSTERS];
            let noise = gaussian_unit(DIM, 50_000 + i as u64);
            normalize(
                center
                    .iter()
                    .zip(&noise)
                    .map(|(c, n)| c + 0.6 * n)
                    .collect(),
            )
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| {
            let base = &corpus[(q * 977) % DOCS];
            let noise = gaussian_unit(DIM, 900_000 + q as u64);
            normalize(base.iter().zip(&noise).map(|(v, n)| v + 0.3 * n).collect())
        })
        .collect();
    (corpus, queries)
}

struct MethodReport {
    label: &'static str,
    build_secs: f64,
    train_secs: f64,
    vectors_bytes: u64,
    ann_kind: String,
    p50_ms: f64,
    p95_ms: f64,
    results: Vec<Vec<u32>>,
}

async fn run_method(
    label: &'static str,
    config: DenseVectorConfig,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> MethodReport {
    let temp = tempfile::tempdir().expect("bench tempdir");
    let dir = MmapDirectory::new(temp.path());
    let mut sb = SchemaBuilder::default();
    let embedding = sb.add_dense_vector_field_with_config("embedding", true, false, config);
    let schema = sb.build();

    let build_start = std::time::Instant::now();
    let index_config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, index_config.clone())
        .await
        .expect("create writer");
    for vector in corpus {
        // add_document is non-blocking; QueueFull is explicit backpressure.
        loop {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vector.clone());
            match writer.add_document(doc) {
                Ok(()) => break,
                Err(crate::Error::QueueFull) => {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Err(error) => panic!("add document: {error:?}"),
            }
        }
    }
    writer.commit().await.expect("commit");
    drop(writer);
    let build_secs = build_start.elapsed().as_secs_f64();

    let train_start = std::time::Instant::now();
    let train_secs = if label == "ivf_pq" {
        let writer = IndexWriter::open(dir.clone(), index_config.clone())
            .await
            .expect("reopen writer for training");
        writer.build_vector_index().await.expect("train IVF-PQ");
        drop(writer);
        train_start.elapsed().as_secs_f64()
    } else {
        0.0
    };

    // Reopen so trained generations and rewritten segments are what serve.
    let index = Index::open(dir, index_config).await.expect("reopen index");
    let segments = index.segment_readers().await.expect("segments");
    let ann_kind = segments
        .iter()
        .filter_map(|segment| segment.vector_indexes().get(&embedding.0))
        .map(|ann| match ann {
            crate::segment::VectorIndex::IvfPq(_) => "ivf_pq",
            crate::segment::VectorIndex::BinaryIvf(_) => "binary_ivf",
            crate::segment::VectorIndex::Tq { .. } => "tq_flat",
        })
        .next()
        .unwrap_or("none")
        .to_string();
    let vectors_bytes: u64 = {
        let mut total = 0u64;
        for entry in std::fs::read_dir(temp.path()).expect("read bench dir") {
            let entry = entry.expect("dir entry");
            if entry.path().extension().is_some_and(|ext| ext == "vectors") {
                total += entry.metadata().expect("metadata").len();
            }
        }
        total
    };

    let reader = index.reader().await.expect("reader");
    let searcher = reader.searcher().await.expect("searcher");
    // Warm the page cache so latency measures compute, not first-touch I/O.
    for query in queries.iter().take(10) {
        let warm = DenseVectorQuery::new(embedding, query.clone()).with_nprobe(NPROBE);
        searcher.search(&warm, K).await.expect("warm query");
    }

    let mut latencies = Vec::with_capacity(queries.len());
    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        let dense = DenseVectorQuery::new(embedding, query.clone()).with_nprobe(NPROBE);
        let started = std::time::Instant::now();
        let hits = searcher.search(&dense, K).await.expect("query");
        latencies.push(started.elapsed().as_secs_f64() * 1_000.0);
        results.push(hits.iter().map(|hit| hit.doc_id).collect::<Vec<u32>>());
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let p50_ms = latencies[latencies.len() / 2];
    let p95_ms = latencies[(latencies.len() * 95) / 100];

    MethodReport {
        label,
        build_secs,
        train_secs,
        vectors_bytes,
        ann_kind,
        p50_ms,
        p95_ms,
        results,
    }
}

fn recall(reference: &[Vec<u32>], candidate: &[Vec<u32>]) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (truth, got) in reference.iter().zip(candidate) {
        let truth: std::collections::HashSet<u32> = truth.iter().copied().collect();
        hits += got.iter().filter(|doc| truth.contains(doc)).count();
        total += truth.len();
    }
    hits as f64 / total as f64
}

/// Ignored benchmark; see the module docs for the release-mode invocation.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn tq_ivf_pq_flat_benchmark() {
    let arch = std::env::consts::ARCH;
    println!(
        "\n=== TQ benchmark: {DOCS} docs, dim {DIM}, {CLUSTERS} clusters, \
         {QUERIES} queries, k={K}, nprobe={NPROBE}, arch={arch} ===",
    );
    let corpus_start = std::time::Instant::now();
    let (corpus, queries) = build_corpus();
    println!(
        "corpus generated in {:.1}s",
        corpus_start.elapsed().as_secs_f64()
    );

    let flat = run_method("flat", DenseVectorConfig::flat(DIM), &corpus, &queries).await;
    let tq = run_method("tq", DenseVectorConfig::tq(DIM), &corpus, &queries).await;
    let ivf_pq = run_method(
        "ivf_pq",
        DenseVectorConfig {
            num_clusters: Some(IVF_NUM_CLUSTERS),
            ..DenseVectorConfig::with_ivf_pq(DIM, Some(IVF_NUM_CLUSTERS), NPROBE)
        },
        &corpus,
        &queries,
    )
    .await;

    assert_eq!(flat.ann_kind, "none", "flat must not build an ANN payload");
    assert_eq!(
        tq.ann_kind, "tq_flat",
        "tq must build its payload at commit"
    );
    assert_eq!(
        ivf_pq.ann_kind, "ivf_pq",
        "ivf_pq must be trained and built"
    );

    println!(
        "\n{:<8} {:>10} {:>10} {:>12} {:>9} {:>9} {:>9}",
        "method", "build(s)", "train(s)", "vectors(MB)", "p50(ms)", "p95(ms)", "recall@10"
    );
    for report in [&flat, &tq, &ivf_pq] {
        println!(
            "{:<8} {:>10.1} {:>10.1} {:>12.1} {:>9.2} {:>9.2} {:>9.3}",
            report.label,
            report.build_secs,
            report.train_secs,
            report.vectors_bytes as f64 / (1024.0 * 1024.0),
            report.p50_ms,
            report.p95_ms,
            recall(&flat.results, &report.results),
        );
    }
    let flat_bytes = flat.vectors_bytes as f64;
    println!(
        "\nANN payload overhead vs flat storage: tq +{:.1} MB, ivf_pq +{:.1} MB",
        (tq.vectors_bytes as f64 - flat_bytes) / (1024.0 * 1024.0),
        (ivf_pq.vectors_bytes as f64 - flat_bytes) / (1024.0 * 1024.0),
    );
}
