//! Focused benchmarks for allocation-sensitive core structures.

use std::hint::black_box;
use std::path::Path;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hermes_core::directories::{Directory, DirectoryWriter, RamDirectory, SliceCachingDirectory};
use hermes_core::query::{Collector, ScoredPosition, SearchResult, TopKCollector};
use hermes_core::structures::fast_field::codec::{auto_read_batch, serialize_auto};
use hermes_core::structures::{BlockSparsePostingList, SparseBlock, WeightQuantization};

fn sparse_postings(count: usize) -> Vec<(u32, u16, f32)> {
    (0..count)
        .map(|index| {
            (
                index as u32 * 3,
                (index % 7) as u16,
                ((index * 17 % 101) as f32 + 1.0) / 101.0,
            )
        })
        .collect()
}

fn bench_top_k(c: &mut Criterion) {
    const CANDIDATES: usize = 100_000;
    const K: usize = 1_000;

    let mut group = c.benchmark_group("core_structures/top_k");
    group.throughput(Throughput::Elements(CANDIDATES as u64));
    group.bench_function("scores_only", |bencher| {
        bencher.iter(|| {
            let mut collector = TopKCollector::new(K);
            for index in 0..CANDIDATES {
                let score = ((index * 2_654_435_761usize) as u32) as f32;
                collector.collect(index as u32, black_box(score), &[]);
            }
            black_box(collector.into_results_with_count())
        });
    });
    group.finish();
}

fn bench_extract_ordinals(c: &mut Criterion) {
    let positions = (0..8)
        .map(|field| {
            let values = (0..128)
                .map(|index| {
                    let ordinal = (index % 16) as u32;
                    ScoredPosition::new((ordinal << 20) | index as u32, index as f32)
                })
                .collect();
            (field, values)
        })
        .collect();
    let result = SearchResult {
        doc_id: 7,
        score: 1.0,
        segment_id: 11,
        positions,
    };

    c.bench_function("core_structures/extract_ordinals", |bencher| {
        bencher.iter(|| black_box(black_box(&result).extract_ordinals()));
    });
}

fn bench_sparse_blocks(c: &mut Criterion) {
    let one_block = sparse_postings(128);
    let many_blocks = sparse_postings(128 * 64);
    let list =
        BlockSparsePostingList::from_postings(&many_blocks, WeightQuantization::Float16).unwrap();
    let decode_block = SparseBlock::from_postings(&one_block, WeightQuantization::Float16).unwrap();

    let mut group = c.benchmark_group("core_structures/sparse_block");
    for quantization in [
        WeightQuantization::Float32,
        WeightQuantization::Float16,
        WeightQuantization::UInt8,
        WeightQuantization::UInt4,
    ] {
        group.bench_with_input(
            BenchmarkId::new("build_128", format!("{quantization:?}")),
            &quantization,
            |bencher, &quantization| {
                bencher.iter(|| {
                    black_box(
                        SparseBlock::from_postings(black_box(&one_block), quantization).unwrap(),
                    )
                });
            },
        );
    }
    group.throughput(Throughput::Elements(many_blocks.len() as u64));
    group.bench_function("serialize_64_blocks", |bencher| {
        bencher.iter(|| black_box(black_box(&list).serialize().unwrap()));
    });
    group.bench_function("decode_f16_128", |bencher| {
        let mut output = Vec::with_capacity(128);
        bencher.iter(|| {
            decode_block.decode_weights_into(&mut output);
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_slice_cache_hit(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let ram = RamDirectory::new();
    let path = Path::new("cached.bin");
    let bytes = vec![0x5au8; 1024 * 1024];
    runtime.block_on(ram.write(path, &bytes)).unwrap();
    let cached = SliceCachingDirectory::new(ram, bytes.len());
    runtime
        .block_on(cached.read_range(path, 0..bytes.len() as u64))
        .unwrap();

    let mut group = c.benchmark_group("core_structures/slice_cache");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("one_mib_hit", |bencher| {
        bencher.iter(|| {
            black_box(
                runtime
                    .block_on(cached.read_range(path, 0..bytes.len() as u64))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_fast_field_blockwise(c: &mut Criterion) {
    const VALUES: usize = 64 * 1024;
    const BLOCK: usize = 512;

    let values: Vec<u64> = (0..VALUES)
        .map(|index| {
            let block = index / BLOCK;
            let offset = index % BLOCK;
            block as u64 * 1_000_000 + offset as u64 * (block as u64 % 7 + 1)
        })
        .collect();
    let mut encoded = Vec::new();
    serialize_auto(&values, &mut encoded).unwrap();
    assert_eq!(encoded[0], 3, "benchmark data must select blockwise codec");

    let mut group = c.benchmark_group("core_structures/fast_field_blockwise");
    group.throughput(Throughput::Elements(VALUES as u64));
    group.bench_function("serialize", |bencher| {
        let mut output = Vec::new();
        bencher.iter(|| {
            output.clear();
            serialize_auto(black_box(&values), &mut output).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("read_batch", |bencher| {
        let mut output = vec![0u64; VALUES];
        bencher.iter(|| {
            auto_read_batch(black_box(&encoded), 0, &mut output);
            black_box(&output);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_top_k,
    bench_extract_ordinals,
    bench_sparse_blocks,
    bench_slice_cache_hit,
    bench_fast_field_blockwise,
);
criterion_main!(benches);
