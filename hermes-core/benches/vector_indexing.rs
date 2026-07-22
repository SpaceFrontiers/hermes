//! Search and merge benchmarks for the production dense ANN implementation.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hermes_core::dsl::IvfRoutingMode;
use hermes_core::structures::{
    CoarseCentroids, CoarseConfig, IVFPQConfig, IVFPQIndex, IvfPqQueryPlan, PQCodebook, PQConfig,
};
use rand::prelude::*;

const DIM: usize = 128;
const VECTOR_COUNT: usize = 10_000;
const CLUSTER_COUNT: usize = 256;
const K: usize = 10;

fn random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
        .collect()
}

fn build_index(vectors: &[Vec<f32>]) -> (CoarseCentroids, PQCodebook, IVFPQIndex) {
    let coarse = CoarseCentroids::train(
        &CoarseConfig::new(DIM, CLUSTER_COUNT).with_routing(IvfRoutingMode::Flat),
        vectors,
    );
    let codebook = PQCodebook::train(PQConfig::new(DIM), vectors, 25);
    let index = IVFPQIndex::build(
        IVFPQConfig::new(DIM, codebook.config.num_subspaces),
        &coarse,
        &codebook,
        vectors,
        None,
    );
    (coarse, codebook, index)
}

fn bench_ivf_pq_search(c: &mut Criterion) {
    let vectors = random_vectors(VECTOR_COUNT, DIM, 42);
    let query = random_vectors(1, DIM, 7).pop().unwrap();
    let (coarse, codebook, index) = build_index(&vectors);

    let mut group = c.benchmark_group("ivf_pq_search");
    for nprobe in [16, 32, 64, 128] {
        let plan = IvfPqQueryPlan::build(&coarse, &codebook, &query, nprobe, IvfRoutingMode::Flat);
        group.bench_with_input(BenchmarkId::from_parameter(nprobe), &nprobe, |b, _| {
            b.iter(|| black_box(index.search_distinct_documents(K, black_box(&plan))));
        });
    }
    group.finish();
}

fn bench_ivf_pq_merge(c: &mut Criterion) {
    let vectors = random_vectors(VECTOR_COUNT, DIM, 42);
    let (coarse, codebook, _) = build_index(&vectors);
    let segment_size = VECTOR_COUNT / 10;
    let indexes: Vec<IVFPQIndex> = vectors
        .chunks(segment_size)
        .map(|segment| {
            IVFPQIndex::build(
                IVFPQConfig::new(DIM, codebook.config.num_subspaces),
                &coarse,
                &codebook,
                segment,
                None,
            )
        })
        .collect();
    let refs: Vec<&IVFPQIndex> = indexes.iter().collect();
    let offsets: Vec<u32> = (0..indexes.len())
        .map(|segment| (segment * segment_size) as u32)
        .collect();

    c.bench_function("ivf_pq_merge_10_segments", |b| {
        b.iter(|| black_box(IVFPQIndex::merge(&refs, &offsets).unwrap()));
    });
}

criterion_group!(benches, bench_ivf_pq_search, bench_ivf_pq_merge);
criterion_main!(benches);
