//! End-to-end search plumbing benchmark: a small multi-segment RAM index with
//! a text field, a multi-valued flat dense field and a sparse field, queried
//! through the same `Searcher` entry points the server uses.
//!
//! This keeps `combine_ordinal_results`, `VectorResultScorer`,
//! `TopKCollector`, `matched_positions` and chunk-level fusion hot, so
//! per-hit allocation and per-segment sorting changes in the shared plumbing
//! are measured where they land rather than in isolation.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hermes_core::directories::RamDirectory;
use hermes_core::dsl::{DenseVectorConfig, Document, SchemaBuilder, VectorIndexType};
use hermes_core::query::{
    DenseVectorQuery, FusionMethod, MultiValueCombiner, Query, SparseVectorQuery, TermQuery,
};
use hermes_core::{Field, Index, IndexConfig, IndexWriter, Searcher};
use rand::prelude::*;

const DIM: usize = 32;
const SEGMENTS: usize = 4;
const DOCS_PER_SEGMENT: usize = 800;
const SPARSE_VOCAB: u32 = 2_000;

struct Fixture {
    searcher: Arc<Searcher<RamDirectory>>,
    title: Field,
    embedding: Field,
    sparse: Field,
    // Keep the index alive for as long as the searcher is used.
    _index: Index<RamDirectory>,
}

fn unit_vector(rng: &mut StdRng) -> Vec<f32> {
    let mut vector: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect();
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|v| *v /= norm);
    vector
}

fn sparse_vector(rng: &mut StdRng, dims: usize) -> Vec<(u32, f32)> {
    let mut entries: Vec<(u32, f32)> = (0..dims)
        .map(|_| {
            (
                rng.random_range(0..SPARSE_VOCAB),
                rng.random_range(0.1f32..2.0),
            )
        })
        .collect();
    entries.sort_unstable_by_key(|&(dim, _)| dim);
    entries.dedup_by_key(|entry| entry.0);
    entries
}

async fn build_fixture() -> Fixture {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, false);
    let embedding = sb.add_dense_vector_field_with_config(
        "embedding",
        true,
        false,
        DenseVectorConfig {
            dim: DIM,
            index_type: VectorIndexType::Flat,
            quantization: hermes_core::dsl::DenseVectorQuantization::F32,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: hermes_core::dsl::IvfRoutingMode::Auto,
            nprobe: 0,
            unit_norm: true,
            soar: None,
        },
    );
    let sparse = sb.add_sparse_vector_field("sparse", true, false);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(hermes_core::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let words = [
        "quantum", "zebra", "habitat", "lattice", "harbor", "signal", "meadow", "orbit", "cipher",
        "glacier", "tundra", "voltage", "pollen", "basalt", "ember", "ledger",
    ];
    for segment in 0..SEGMENTS {
        for doc in 0..DOCS_PER_SEGMENT {
            let mut document = Document::new();
            let mut text = String::new();
            for _ in 0..6 {
                text.push_str(words[rng.random_range(0..words.len())]);
                text.push(' ');
            }
            text.push_str(&format!("segment{segment} doc{doc}"));
            document.add_text(title, text);
            // Every third document carries three chunks so the multi-value
            // combiner and per-ordinal positions are exercised.
            let chunks = if doc % 3 == 0 { 3 } else { 1 };
            for _ in 0..chunks {
                document.add_dense_vector(embedding, unit_vector(&mut rng));
                document.add_sparse_vector(sparse, sparse_vector(&mut rng, 24));
            }
            writer.add_document(document).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(
        searcher.segment_readers().len(),
        SEGMENTS,
        "fixture must span several segments"
    );
    Fixture {
        searcher,
        title,
        embedding,
        sparse,
        _index: index,
    }
}

fn bench_search_pipeline(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let fixture = runtime.block_on(build_fixture());
    let searcher = &fixture.searcher;

    let mut rng = StdRng::seed_from_u64(7);
    let dense_query = DenseVectorQuery::new(fixture.embedding, unit_vector(&mut rng))
        .with_combiner(MultiValueCombiner::log_sum_exp());
    let sparse_query = SparseVectorQuery::new(fixture.sparse, sparse_vector(&mut rng, 24))
        .with_combiner(MultiValueCombiner::Max);
    let text_query = TermQuery::text(fixture.title, "quantum");
    let hybrid: Vec<(&dyn Query, f32)> = vec![
        (&text_query, 1.0),
        (&dense_query, 1.0),
        (&sparse_query, 1.0),
    ];

    let mut group = c.benchmark_group("search_pipeline");
    group.throughput(Throughput::Elements(1));
    for &limit in &[10usize, 200] {
        group.bench_with_input(
            BenchmarkId::new("fused_hybrid", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(searcher.search_fused_with_count(
                                black_box(&hybrid),
                                limit,
                                limit,
                                FusionMethod::default(),
                                MultiValueCombiner::Max,
                            ))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_top_k", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(searcher.search_with_count(black_box(&dense_query), limit))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_top_k_positions", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(
                                searcher.search_with_positions(black_box(&dense_query), limit),
                            )
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("sparse_top_k", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(searcher.search_with_count(black_box(&sparse_query), limit))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();

    // The async/current-thread path (HTTP/WASM shape): no rayon, scorers are
    // built through `Query::scorer` and results flow through the bounded
    // async segment stream.
    let current_thread = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("search_pipeline_current_thread");
    group.throughput(Throughput::Elements(1));
    for &limit in &[10usize, 200] {
        group.bench_with_input(
            BenchmarkId::new("fused_hybrid", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        current_thread
                            .block_on(searcher.search_fused_with_count(
                                black_box(&hybrid),
                                limit,
                                limit,
                                FusionMethod::default(),
                                MultiValueCombiner::Max,
                            ))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_top_k", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        current_thread
                            .block_on(searcher.search_with_count(black_box(&dense_query), limit))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_search_pipeline);
criterion_main!(benches);
