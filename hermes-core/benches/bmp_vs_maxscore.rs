//! Benchmark: BMP vs MaxScore for sparse vector retrieval
//!
//! Compares the two sparse vector index formats on synthetic SPLADE-like data:
//! - Index build time
//! - Query latency (top-10, top-100)
//!
//! Run: cargo bench --bench bmp_vs_maxscore
//!
//! With more docs: BMP_BENCH_DOCS=50000 cargo bench --bench bmp_vs_maxscore

use std::sync::Arc;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use hermes_core::directories::RamDirectory;
use hermes_core::dsl::{Document, SchemaBuilder};
use hermes_core::index::{IndexConfig, IndexWriter};
use hermes_core::query::SparseVectorQuery;
use hermes_core::structures::SparseVectorConfig;

// ============================================================================
// Data generation
// ============================================================================

/// Simple LCG pseudo-random number generator (deterministic, no deps).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() % 10000) as f32 / 10000.0
    }
}

/// Generate SPLADE-like sparse vectors: power-law dimension distribution,
/// ~100-200 non-zero dims per doc, weights concentrated at low values.
fn generate_sparse_docs(num_docs: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;

    (0..num_docs)
        .map(|_| {
            let num_dims = 80 + (rng.next_u32() % 120) as usize;
            let mut entries = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() * rng.next_f32() * 2.0 + 0.01;
                entries.push((dim.min(vocab_size - 1), weight));
            }
            entries.sort_by_key(|&(d, _)| d);
            entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1.max(a.1);
                    true
                } else {
                    false
                }
            });
            entries
        })
        .collect()
}

/// Generate clustered sparse docs that simulate topic locality.
///
/// Documents are grouped by "topic" (100 topics). Each topic has a core set of
/// ~500 dimensions. Documents from the same topic share many dimensions.
/// Documents are ordered by topic, so consecutive doc_ids (which form BMP blocks)
/// have similar dimension distributions — enabling effective block pruning.
///
/// This simulates BP (Bipartite Partitioning) document ordering used in the
/// SIGIR 2024 BMP paper, where similar documents are placed adjacent.
fn generate_clustered_sparse_docs(num_docs: usize, seed: u64) -> ClusteredData {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;
    let num_topics = 100;
    let dims_per_topic = 500;

    // Generate topic-dimension assignments
    let mut topic_dims: Vec<Vec<u32>> = Vec::with_capacity(num_topics);
    for t in 0..num_topics {
        let base = (t as u32 * (vocab_size / num_topics as u32)) % vocab_size;
        let mut dims: Vec<u32> = (0..dims_per_topic)
            .map(|_| (base + rng.next_u32() % (vocab_size / 5)) % vocab_size)
            .collect();
        dims.sort_unstable();
        dims.dedup();
        topic_dims.push(dims);
    }

    let docs_per_topic = num_docs / num_topics;
    let mut all_docs = Vec::with_capacity(num_docs);

    for topic in topic_dims.iter().take(num_topics) {
        for _ in 0..docs_per_topic {
            let num_dims = 80 + (rng.next_u32() % 120) as usize;
            let mut entries = Vec::with_capacity(num_dims);

            // 70% of dims from topic's core set (higher weights)
            let topic_count = (num_dims as f32 * 0.7) as usize;
            for _ in 0..topic_count {
                let dim = topic[rng.next_u32() as usize % topic.len()];
                let weight = rng.next_f32() * 1.5 + 0.3;
                entries.push((dim, weight));
            }

            // 30% random dims (noise / cross-topic terms)
            for _ in topic_count..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() * rng.next_f32() * 0.5 + 0.01;
                entries.push((dim.min(vocab_size - 1), weight));
            }

            entries.sort_by_key(|&(d, _)| d);
            entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1.max(a.1);
                    true
                } else {
                    false
                }
            });
            all_docs.push(entries);
        }
    }

    while all_docs.len() < num_docs {
        let t = rng.next_u32() as usize % num_topics;
        let topic = &topic_dims[t];
        let num_dims = 80 + (rng.next_u32() % 120) as usize;
        let mut entries = Vec::with_capacity(num_dims);
        let topic_count = (num_dims as f32 * 0.7) as usize;
        for _ in 0..topic_count {
            let dim = topic[rng.next_u32() as usize % topic.len()];
            let weight = rng.next_f32() * 1.5 + 0.3;
            entries.push((dim, weight));
        }
        for _ in topic_count..num_dims {
            let raw = rng.next_f32();
            let dim = ((raw * raw) * vocab_size as f32) as u32;
            let weight = rng.next_f32() * rng.next_f32() * 0.5 + 0.01;
            entries.push((dim.min(vocab_size - 1), weight));
        }
        entries.sort_by_key(|&(d, _)| d);
        entries.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = b.1.max(a.1);
                true
            } else {
                false
            }
        });
        all_docs.push(entries);
    }

    ClusteredData {
        docs: all_docs,
        topic_dims,
    }
}

/// Generate topical queries matching the clustered data.
fn generate_clustered_queries(
    num_queries: usize,
    topic_dims: &[Vec<u32>],
    seed: u64,
) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let num_topics = topic_dims.len();
    let vocab_size = 30000u32;

    (0..num_queries)
        .map(|_| {
            let t = rng.next_u32() as usize % num_topics;
            let topic = &topic_dims[t];
            let num_dims = 15 + (rng.next_u32() as usize % 25);
            let mut entries = Vec::with_capacity(num_dims);

            let topic_count = (num_dims as f32 * 0.8) as usize;
            for _ in 0..topic_count {
                let dim = topic[rng.next_u32() as usize % topic.len()];
                let weight = rng.next_f32() + 0.2;
                entries.push((dim, weight));
            }
            for _ in topic_count..num_dims {
                let dim = rng.next_u32() % vocab_size;
                let weight = rng.next_f32() * 0.3 + 0.05;
                entries.push((dim, weight));
            }

            entries.sort_by_key(|&(d, _)| d);
            entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1.max(a.1);
                    true
                } else {
                    false
                }
            });
            entries
        })
        .collect()
}

struct ClusteredData {
    docs: Vec<Vec<(u32, f32)>>,
    topic_dims: Vec<Vec<u32>>,
}

/// Generate random queries with 10-50 non-zero dimensions.
fn generate_queries(num_queries: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    generate_queries_with_dims(num_queries, seed, 10, 40)
}

/// Generate random queries with a specified dimension range.
///
/// Each query has `min_dims + random(0..range)` non-zero dimensions.
fn generate_queries_with_dims(
    num_queries: usize,
    seed: u64,
    min_dims: usize,
    range: usize,
) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;

    (0..num_queries)
        .map(|_| {
            let num_dims = min_dims + (rng.next_u32() as usize % range.max(1));
            let mut entries = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() + 0.1;
                entries.push((dim.min(vocab_size - 1), weight));
            }
            entries.sort_by_key(|&(d, _)| d);
            entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1.max(a.1);
                    true
                } else {
                    false
                }
            });
            entries
        })
        .collect()
}

// ============================================================================
// Index building helpers
// ============================================================================

struct BuiltIndex {
    index: hermes_core::index::Index<RamDirectory>,
    sparse_field: hermes_core::dsl::Field,
    build_time_ms: f64,
}

/// Build index with one sparse vector per document.
fn build_index(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    sparse_config: SparseVectorConfig,
    label: &str,
) -> BuiltIndex {
    build_index_inner(rt, docs, sparse_config, label, 1)
}

/// Build index with multiple sparse vectors per document (multi-ordinal).
fn build_index_multi_ordinal(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    sparse_config: SparseVectorConfig,
    label: &str,
    vectors_per_doc: usize,
    seed: u64,
) -> BuiltIndex {
    // For multi-ordinal, we generate extra vectors as slight variations of the original
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;
    let mut multi_docs: Vec<Vec<Vec<(u32, f32)>>> = Vec::with_capacity(docs.len());

    for doc in docs {
        let mut vectors = Vec::with_capacity(vectors_per_doc);
        vectors.push(doc.clone());
        for _ in 1..vectors_per_doc {
            // Generate a variation: shift some dimensions, add noise to weights
            let num_dims = 40 + (rng.next_u32() % 80) as usize;
            let mut entries = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() * rng.next_f32() * 1.5 + 0.01;
                entries.push((dim.min(vocab_size - 1), weight));
            }
            entries.sort_by_key(|&(d, _)| d);
            entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1.max(a.1);
                    true
                } else {
                    false
                }
            });
            vectors.push(entries);
        }
        multi_docs.push(vectors);
    }

    let mut sb = SchemaBuilder::default();
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, false, sparse_config);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Commit interval scales with doc count: at most ~5 segments
    let commit_interval = (docs.len() / 5).max(20_000);

    let start = Instant::now();
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for (i, vectors) in multi_docs.iter().enumerate() {
            let mut doc = Document::new();
            for v in vectors {
                doc.add_sparse_vector(sparse, v.clone());
            }
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(hermes_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        for v in vectors {
                            doc.add_sparse_vector(sparse, v.clone());
                        }
                    }
                    Err(e) => panic!("add_document failed: {}", e),
                }
            }
            if (i + 1) % commit_interval == 0 {
                writer.commit().await.unwrap();
            }
        }
        writer.force_merge().await.unwrap();
    });
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  [{}] built: {:.1}ms ({} vectors/doc)",
        label, build_time_ms, vectors_per_doc
    );

    let index = rt
        .block_on(hermes_core::index::Index::open(dir, config))
        .unwrap();

    BuiltIndex {
        index,
        sparse_field: sparse,
        build_time_ms,
    }
}

fn build_index_inner(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    sparse_config: SparseVectorConfig,
    label: &str,
    _vectors_per_doc: usize,
) -> BuiltIndex {
    let mut sb = SchemaBuilder::default();
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, false, sparse_config);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Commit interval scales with doc count: at most ~5 segments
    let commit_interval = (docs.len() / 5).max(20_000);

    let start = Instant::now();
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for (i, entries) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_sparse_vector(sparse, entries.clone());
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(hermes_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        doc.add_sparse_vector(sparse, entries.clone());
                    }
                    Err(e) => panic!("add_document failed: {}", e),
                }
            }
            if (i + 1) % commit_interval == 0 {
                writer.commit().await.unwrap();
            }
        }
        writer.force_merge().await.unwrap();
    });
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  [{}] built: {:.1}ms", label, build_time_ms);

    let index = rt
        .block_on(hermes_core::index::Index::open(dir, config))
        .unwrap();

    BuiltIndex {
        index,
        sparse_field: sparse,
        build_time_ms,
    }
}

/// Build a BMP index split into `num_segments` segments, then force-merge.
///
/// When `with_simhash` is true, sets the `simhash` attribute on the sparse
/// vector field so the builder auto-computes SimHash from the vector data,
/// and the merger reorders blocks by SimHash similarity during force_merge.
fn build_bmp_segmented(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    with_simhash: bool,
    sparse_config: SparseVectorConfig,
    num_segments: usize,
    label: &str,
) -> BuiltIndex {
    let mut sb = SchemaBuilder::default();
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, false, sparse_config);

    if with_simhash {
        sb.set_simhash(sparse, true);
    }

    let schema = sb.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let commit_interval = (docs.len() / num_segments).max(1);

    let start = Instant::now();
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for (i, entries) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_sparse_vector(sparse, entries.clone());
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(hermes_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        doc.add_sparse_vector(sparse, entries.clone());
                    }
                    Err(e) => panic!("add_document failed: {}", e),
                }
            }
            if (i + 1) % commit_interval == 0 {
                writer.commit().await.unwrap();
            }
        }
        writer.force_merge().await.unwrap();
    });
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  [{}] built: {:.1}ms ({} segments merged)",
        label, build_time_ms, num_segments
    );

    let index = rt
        .block_on(hermes_core::index::Index::open(dir, config))
        .unwrap();

    BuiltIndex {
        index,
        sparse_field: sparse,
        build_time_ms,
    }
}

// ============================================================================
// Diagnostics: measure pruning effectiveness
// ============================================================================

/// Run a few queries and print pruning statistics.
fn print_pruning_diagnostics(
    rt: &tokio::runtime::Runtime,
    bmp: &BuiltIndex,
    maxscore: &BuiltIndex,
    queries: &[Vec<(u32, f32)>],
    label: &str,
) {
    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    let sample = &queries[..queries.len().min(20)];

    // BMP: measure latency
    let bmp_start = Instant::now();
    for q in sample {
        let query = SparseVectorQuery::new(bmp.sparse_field, q.clone());
        let _ = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
    }
    let bmp_avg_us = bmp_start.elapsed().as_micros() as f64 / sample.len() as f64;

    // MaxScore: measure latency
    let ms_start = Instant::now();
    for q in sample {
        let query = SparseVectorQuery::new(maxscore.sparse_field, q.clone());
        let _ = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
    }
    let ms_avg_us = ms_start.elapsed().as_micros() as f64 / sample.len() as f64;

    let (winner, speedup) = if bmp_avg_us < ms_avg_us {
        ("BMP", ms_avg_us / bmp_avg_us)
    } else {
        ("MaxScore", bmp_avg_us / ms_avg_us)
    };

    eprintln!(
        "\n  [{label}] Diagnostics (avg over {} queries):",
        sample.len()
    );
    eprintln!("    BMP:      {:.0}µs/query", bmp_avg_us);
    eprintln!("    MaxScore: {:.0}µs/query", ms_avg_us);
    eprintln!("    Winner:   {winner} ({speedup:.2}x faster)");
}

// ============================================================================
// Benchmark functions
// ============================================================================

fn bench_query_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );
    let docs = generate_sparse_docs(num_docs, 12345);
    let queries = generate_queries(num_queries, 67890);

    // Build both indexes
    eprintln!("Building indexes...");
    let bmp = build_index(&rt, &docs, SparseVectorConfig::splade_bmp(), "BMP");
    let maxscore = build_index(&rt, &docs, SparseVectorConfig::splade(), "MaxScore");

    eprintln!(
        "\nBuild times: BMP={:.1}ms, MaxScore={:.1}ms",
        bmp.build_time_ms, maxscore.build_time_ms
    );

    print_pruning_diagnostics(&rt, &bmp, &maxscore, &queries, "random");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    // Top-10 benchmark
    {
        let mut group = c.benchmark_group("sparse_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    // Top-100 benchmark
    {
        let mut group = c.benchmark_group("sparse_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark with clustered (topic-local) data — simulates real SPLADE workloads.
///
/// Documents are ordered by topic so BMP blocks contain topically similar docs.
/// This enables effective block pruning (the key to BMP's speedup).
fn bench_clustered(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== Clustered data benchmark ===");
    eprintln!(
        "Generating {} clustered docs and {} queries...",
        num_docs, num_queries
    );
    let clustered = generate_clustered_sparse_docs(num_docs, 54321);
    let queries = generate_clustered_queries(num_queries, &clustered.topic_dims, 67890);

    eprintln!("Building indexes (clustered data)...");
    let bmp = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-clustered",
    );
    let maxscore = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade(),
        "MaxScore-clustered",
    );

    eprintln!(
        "Build times: BMP={:.1}ms, MaxScore={:.1}ms",
        bmp.build_time_ms, maxscore.build_time_ms
    );

    print_pruning_diagnostics(&rt, &bmp, &maxscore, &queries, "clustered");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    {
        let mut group = c.benchmark_group("clustered_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    {
        let mut group = c.benchmark_group("clustered_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

fn bench_multi_ordinal(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let vectors_per_doc = 5; // 5 sparse vectors per document
    let num_queries = 100;

    eprintln!(
        "\n=== Multi-ordinal benchmark ({} vectors/doc) ===",
        vectors_per_doc
    );
    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );
    let docs = generate_sparse_docs(num_docs, 12345);
    let queries = generate_queries(num_queries, 67890);

    eprintln!("Building multi-ordinal indexes...");
    let bmp = build_index_multi_ordinal(
        &rt,
        &docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-multi",
        vectors_per_doc,
        99999,
    );
    let maxscore = build_index_multi_ordinal(
        &rt,
        &docs,
        SparseVectorConfig::splade(),
        "MaxScore-multi",
        vectors_per_doc,
        99999,
    );

    eprintln!(
        "Build times: BMP={:.1}ms, MaxScore={:.1}ms",
        bmp.build_time_ms, maxscore.build_time_ms
    );

    // Print BMP-specific multi-ordinal diagnostics
    {
        let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
        let bmp_searcher = rt.block_on(bmp_reader.searcher()).unwrap();
        if let Some(bmp_idx) = bmp_searcher
            .segment_readers()
            .first()
            .and_then(|r| r.bmp_index(bmp.sparse_field))
        {
            eprintln!("\n  [multi-ordinal BMP diagnostics]");
            eprintln!(
                "    blocks={}, block_size={}, total_postings={}, num_virtual_docs={}",
                bmp_idx.num_blocks,
                bmp_idx.bmp_block_size,
                bmp_idx.total_postings(),
                bmp_idx.num_virtual_docs,
            );
            let avg_postings_per_block =
                bmp_idx.total_postings() as f64 / bmp_idx.num_blocks as f64;
            eprintln!("    avg_postings/block={:.0}", avg_postings_per_block,);
        }
    }

    print_pruning_diagnostics(&rt, &bmp, &maxscore, &queries, "multi-ordinal");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    {
        let mut group = c.benchmark_group("multi_ord_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark with varying query lengths (50, 100, 200 non-zero dimensions).
///
/// Longer queries are typical for SPLADE v2 and other learned sparse models.
/// BMP's advantage should increase with query length because:
/// - More query terms → higher block UBs → better pruning discrimination
/// - MaxScore processes more posting lists per query
fn bench_long_queries(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== Long query benchmark ===");
    let docs = generate_sparse_docs(num_docs, 12345);

    eprintln!("Building indexes...");
    let bmp = build_index(&rt, &docs, SparseVectorConfig::splade_bmp(), "BMP-long");
    let maxscore = build_index(&rt, &docs, SparseVectorConfig::splade(), "MaxScore-long");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    // Query lengths to benchmark: 50, 100, 200 non-zero dims
    // BMP gets the full query; MaxScore stays at default 32 (dies with more)
    for target_dims in [50, 100, 200] {
        let queries = generate_queries_with_dims(
            num_queries,
            67890 + target_dims as u64,
            target_dims,
            target_dims / 4,
        );
        let avg_dims: f64 =
            queries.iter().map(|q| q.len() as f64).sum::<f64>() / queries.len() as f64;
        eprintln!(
            "  query_dims={}: avg actual dims = {:.1} (MaxScore capped to 32)",
            target_dims, avg_dims
        );

        let group_name = format!("long_q{}_top10", target_dims);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone())
                        .with_max_query_dims(target_dims);
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        // MaxScore uses default max_query_dims (32) — can't handle 200+ terms
        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark approximate retrieval with varying heap_factor (alpha).
///
/// Tests BMP and MaxScore with alpha = {1.0, 0.8, 0.6} to measure the
/// speed–accuracy trade-off. Alpha < 1.0 means more aggressive pruning:
/// - BMP: prune when `UB * alpha <= threshold`
/// - MaxScore: prune when `UB <= threshold / alpha`
///
/// Also measures recall@10 vs exact (alpha=1.0) to quantify quality loss.
fn bench_approximate(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== Approximate retrieval benchmark ===");
    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );
    let clustered = generate_clustered_sparse_docs(num_docs, 54321);
    let queries = generate_clustered_queries(num_queries, &clustered.topic_dims, 67890);

    eprintln!("Building indexes...");
    let bmp = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-approx",
    );
    let maxscore = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade(),
        "MaxScore-approx",
    );

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    let alphas: &[f32] = &[1.0, 0.8, 0.6];

    // Measure recall@10 for each alpha vs exact (alpha=1.0)
    {
        eprintln!("\n  Recall@10 vs exact (alpha=1.0):");

        // Exact BMP results (alpha=1.0)
        let exact_bmp: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| {
                let query = SparseVectorQuery::new(bmp.sparse_field, q.clone());
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                results.iter().map(|h| h.doc_id).collect()
            })
            .collect();

        // Exact MaxScore results (alpha=1.0)
        let exact_ms: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| {
                let query = SparseVectorQuery::new(maxscore.sparse_field, q.clone());
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                results.iter().map(|h| h.doc_id).collect()
            })
            .collect();

        for &alpha in alphas {
            // BMP recall
            let bmp_recall: f64 = queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let query =
                        SparseVectorQuery::new(bmp.sparse_field, q.clone()).with_heap_factor(alpha);
                    let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                    let approx: Vec<u32> = results.iter().map(|h| h.doc_id).collect();
                    let overlap = approx.iter().filter(|d| exact_bmp[i].contains(d)).count();
                    overlap as f64 / exact_bmp[i].len().max(1) as f64
                })
                .sum::<f64>()
                / queries.len() as f64;

            // MaxScore recall
            let ms_recall: f64 = queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let query = SparseVectorQuery::new(maxscore.sparse_field, q.clone())
                        .with_heap_factor(alpha);
                    let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                    let approx: Vec<u32> = results.iter().map(|h| h.doc_id).collect();
                    let overlap = approx.iter().filter(|d| exact_ms[i].contains(d)).count();
                    overlap as f64 / exact_ms[i].len().max(1) as f64
                })
                .sum::<f64>()
                / queries.len() as f64;

            eprintln!(
                "    alpha={:.1}: BMP recall={:.3}, MaxScore recall={:.3}",
                alpha, bmp_recall, ms_recall
            );
        }
    }

    // Benchmark latency at each alpha
    for &alpha in alphas {
        let group_name = format!("approx_a{}_top10", (alpha * 10.0) as u32);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone())
                        .with_heap_factor(alpha);
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                )
                .with_heap_factor(alpha);
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark BMP with and without SimHash-based block reordering.
///
/// Uses uniformly random data (worst case for BMP superblock pruning) to
/// isolate the effect of SimHash reordering. Without reordering, blocks
/// contain random mixtures of documents and superblock pruning skips
/// almost nothing. With SimHash reordering, the merger clusters similar
/// documents within the same blocks/superblocks, enabling effective pruning.
///
/// Both indexes are built from the same data, split into the same number
/// of segments, then force-merged. The only difference is that the SimHash
/// variant has a u64 simhash fast field that guides block reordering.
fn bench_simhash_reorder(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;
    let num_segments = 5;

    eprintln!("\n=== SimHash block-reorder benchmark ===");
    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );

    // Use random data — worst case for BMP without reordering
    let docs = generate_sparse_docs(num_docs, 12345);
    let queries = generate_queries(num_queries, 67890);

    eprintln!(
        "Building BMP without SimHash reordering ({} segments)...",
        num_segments
    );
    let bmp_plain = build_bmp_segmented(
        &rt,
        &docs,
        false,
        SparseVectorConfig::splade_bmp(),
        num_segments,
        "BMP-plain",
    );

    eprintln!(
        "Building BMP with SimHash reordering ({} segments)...",
        num_segments
    );
    let bmp_simhash = build_bmp_segmented(
        &rt,
        &docs,
        true,
        SparseVectorConfig::splade_bmp(),
        num_segments,
        "BMP-simhash",
    );

    eprintln!(
        "Build times: plain={:.1}ms, simhash={:.1}ms",
        bmp_plain.build_time_ms, bmp_simhash.build_time_ms
    );

    // Print BMP index diagnostics for both variants
    {
        let plain_reader = rt.block_on(bmp_plain.index.reader()).unwrap();
        let plain_searcher = rt.block_on(plain_reader.searcher()).unwrap();
        if let Some(bmp_idx) = plain_searcher
            .segment_readers()
            .first()
            .and_then(|r| r.bmp_index(bmp_plain.sparse_field))
        {
            eprintln!(
                "  [plain] blocks={}, superblocks={}",
                bmp_idx.num_blocks,
                bmp_idx.num_blocks.div_ceil(64),
            );
        }

        let sh_reader = rt.block_on(bmp_simhash.index.reader()).unwrap();
        let sh_searcher = rt.block_on(sh_reader.searcher()).unwrap();
        if let Some(bmp_idx) = sh_searcher
            .segment_readers()
            .first()
            .and_then(|r| r.bmp_index(bmp_simhash.sparse_field))
        {
            eprintln!(
                "  [simhash] blocks={}, superblocks={}",
                bmp_idx.num_blocks,
                bmp_idx.num_blocks.div_ceil(64),
            );
        }
    }

    // Warmup diagnostics: compare average query latency
    {
        let plain_reader = rt.block_on(bmp_plain.index.reader()).unwrap();
        let plain_searcher = Arc::new(rt.block_on(plain_reader.searcher()).unwrap());

        let sh_reader = rt.block_on(bmp_simhash.index.reader()).unwrap();
        let sh_searcher = Arc::new(rt.block_on(sh_reader.searcher()).unwrap());

        let sample = &queries[..queries.len().min(20)];

        let plain_start = Instant::now();
        for q in sample {
            let query = SparseVectorQuery::new(bmp_plain.sparse_field, q.clone());
            let _ = rt.block_on(plain_searcher.search(&query, 10)).unwrap();
        }
        let plain_avg_us = plain_start.elapsed().as_micros() as f64 / sample.len() as f64;

        let sh_start = Instant::now();
        for q in sample {
            let query = SparseVectorQuery::new(bmp_simhash.sparse_field, q.clone());
            let _ = rt.block_on(sh_searcher.search(&query, 10)).unwrap();
        }
        let sh_avg_us = sh_start.elapsed().as_micros() as f64 / sample.len() as f64;

        let speedup = plain_avg_us / sh_avg_us;
        eprintln!(
            "\n  [simhash-reorder] Diagnostics (avg over {} queries):",
            sample.len()
        );
        eprintln!("    BMP (plain):    {:.0}µs/query", plain_avg_us);
        eprintln!("    BMP (simhash):  {:.0}µs/query", sh_avg_us);
        if speedup >= 1.0 {
            eprintln!("    SimHash wins:   {:.2}x faster", speedup);
        } else {
            eprintln!("    Plain wins:     {:.2}x faster", 1.0 / speedup);
        }
    }

    let plain_reader = rt.block_on(bmp_plain.index.reader()).unwrap();
    let plain_searcher = Arc::new(rt.block_on(plain_reader.searcher()).unwrap());

    let sh_reader = rt.block_on(bmp_simhash.index.reader()).unwrap();
    let sh_searcher = Arc::new(rt.block_on(sh_reader.searcher()).unwrap());

    // Top-10 benchmark
    {
        let mut group = c.benchmark_group("simhash_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-plain", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_plain.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(plain_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-simhash", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_simhash.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(sh_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    // Top-100 benchmark
    {
        let mut group = c.benchmark_group("simhash_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-plain", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_plain.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(plain_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-simhash", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_simhash.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(sh_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_query_latency,
    bench_clustered,
    bench_multi_ordinal,
    bench_long_queries,
    bench_approximate,
    bench_simhash_reorder,
);
criterion_main!(benches);
