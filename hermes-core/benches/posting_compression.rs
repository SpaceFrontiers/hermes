//! Comprehensive benchmarks for posting list compression methods
//!
//! Compares: Bitpacked, SIMD-BP128, Elias-Fano, Partitioned EF, Roaring
//!
//! Tests multiple doc_id distributions:
//! - Sparse (1% density): rare terms
//! - Medium (10% density): typical terms
//! - Dense (50% density): common terms
//! - Clustered: docs grouped in ranges (locality)
//! - Sequential: consecutive doc_ids (best case)
//!
//! Metrics measured:
//! - Compression ratio (bytes per doc_id)
//! - Encoding speed (elements/sec)
//! - Decoding/iteration speed (elements/sec)
//! - Seek speed (seeks/sec)
//! - Serialization/deserialization speed

#![allow(dead_code)]

use comfy_table::{Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hermes_core::structures::{
    BitpackedPostingList, CompressedPostingList, EliasFanoPostingList, PartitionedEFPostingList,
    RoaringPostingList, SimdBp128PostingList,
};
use std::io::Cursor;

/// Distribution types for doc_id generation
#[derive(Clone, Copy, Debug)]
enum Distribution {
    /// Random gaps with given density (count/universe ratio)
    Sparse, // 1% density - rare terms
    Medium,     // 10% density - typical terms
    Dense,      // 50% density - common terms
    Clustered,  // Grouped in clusters (locality)
    Sequential, // Consecutive doc_ids (best case for delta)
}

impl Distribution {
    fn name(&self) -> &'static str {
        match self {
            Distribution::Sparse => "sparse_1pct",
            Distribution::Medium => "medium_10pct",
            Distribution::Dense => "dense_50pct",
            Distribution::Clustered => "clustered",
            Distribution::Sequential => "sequential",
        }
    }

    fn density(&self) -> f32 {
        match self {
            Distribution::Sparse => 0.01,
            Distribution::Medium => 0.10,
            Distribution::Dense => 0.50,
            Distribution::Clustered => 0.10, // overall density
            Distribution::Sequential => 1.0,
        }
    }
}

/// Generate synthetic posting list data with various distributions
fn generate_postings(count: usize, dist: Distribution) -> (Vec<u32>, Vec<u32>) {
    let mut doc_ids = Vec::with_capacity(count);
    let mut term_freqs = Vec::with_capacity(count);
    let mut rng_state = 12345u64;

    // Helper for LCG random
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng_state
    };

    match dist {
        Distribution::Sparse | Distribution::Medium | Distribution::Dense => {
            let density = dist.density();
            let universe = (count as f32 / density) as u32;
            let avg_gap = (universe / count as u32).max(1);
            let mut current_doc = 0u32;

            for _ in 0..count {
                let gap = ((next_rand() >> 33) as u32 % (avg_gap * 2)).max(1);
                current_doc += gap;
                doc_ids.push(current_doc);
            }
        }
        Distribution::Clustered => {
            // Create clusters of ~100 docs with gaps between clusters
            let cluster_size = 100;
            let num_clusters = count.div_ceil(cluster_size);
            let cluster_gap = 10000u32; // gap between clusters
            let mut current_doc = 0u32;

            for cluster in 0..num_clusters {
                let cluster_count = (count - cluster * cluster_size).min(cluster_size);
                for _ in 0..cluster_count {
                    let gap = ((next_rand() >> 33) as u32 % 10).max(1); // small gaps within cluster
                    current_doc += gap;
                    doc_ids.push(current_doc);
                }
                current_doc += cluster_gap; // big gap to next cluster
            }
        }
        Distribution::Sequential => {
            // Consecutive doc_ids starting from random offset
            let start = (next_rand() >> 33) as u32 % 1_000_000;
            for i in 0..count {
                doc_ids.push(start + i as u32);
            }
        }
    }

    // TF follows Zipf-like distribution (mostly 1s)
    for _ in 0..count {
        let tf = if (next_rand() >> 40) % 10 < 7 {
            1
        } else {
            ((next_rand() >> 45) % 5 + 1) as u32
        };
        term_freqs.push(tf);
    }

    (doc_ids, term_freqs)
}

/// Compression stats for a single test case
struct CompressionResult {
    raw_bytes: usize,
    bitpacked_bytes: usize,
    simd_bp128_bytes: usize,
    elias_fano_bytes: usize,
    partitioned_ef_bytes: usize,
    roaring_bytes: usize,
}

impl CompressionResult {
    fn bitpacked_ratio(&self) -> f64 {
        self.bitpacked_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    fn simd_bp128_ratio(&self) -> f64 {
        self.simd_bp128_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    fn elias_fano_ratio(&self) -> f64 {
        self.elias_fano_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    fn partitioned_ef_ratio(&self) -> f64 {
        self.partitioned_ef_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    fn roaring_ratio(&self) -> f64 {
        self.roaring_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
}

fn measure_compression(doc_ids: &[u32], term_freqs: &[u32]) -> CompressionResult {
    let raw_bytes = doc_ids.len() * 8; // 4 bytes doc_id + 4 bytes tf

    let bitpacked = BitpackedPostingList::from_postings(doc_ids, term_freqs, 1.0);
    let mut bp_buf = Vec::new();
    bitpacked.serialize(&mut bp_buf).unwrap();

    let simd_bp128 = SimdBp128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let mut simd_buf = Vec::new();
    simd_bp128.serialize(&mut simd_buf).unwrap();

    let elias_fano = EliasFanoPostingList::from_postings(doc_ids, term_freqs);
    let mut ef_buf = Vec::new();
    elias_fano.serialize(&mut ef_buf).unwrap();

    let partitioned_ef = PartitionedEFPostingList::from_postings(doc_ids, term_freqs);
    let mut pef_buf = Vec::new();
    partitioned_ef.serialize(&mut pef_buf).unwrap();

    let roaring = RoaringPostingList::from_postings(doc_ids, term_freqs);
    let mut roar_buf = Vec::new();
    roaring.serialize(&mut roar_buf).unwrap();

    CompressionResult {
        raw_bytes,
        bitpacked_bytes: bp_buf.len(),
        simd_bp128_bytes: simd_buf.len(),
        elias_fano_bytes: ef_buf.len(),
        partitioned_ef_bytes: pef_buf.len(),
        roaring_bytes: roar_buf.len(),
    }
}

fn bench_encoding_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding_speed");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("bitpacked", size), &size, |b, _| {
            b.iter(|| {
                black_box(BitpackedPostingList::from_postings(
                    &doc_ids,
                    &term_freqs,
                    1.0,
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| black_box(EliasFanoPostingList::from_postings(&doc_ids, &term_freqs)))
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| black_box(RoaringPostingList::from_postings(&doc_ids, &term_freqs)))
        });
    }

    group.finish();
}

fn bench_iteration_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_speed");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        let bitpacked = BitpackedPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);

        group.bench_with_input(BenchmarkId::new("bitpacked", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = bitpacked.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = elias_fano.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = roaring.iterator();
                iter.init();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

fn bench_seek_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("seek_speed");

    let size = 50000;
    let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);

    // Generate seek targets
    let seek_targets: Vec<u32> = (0..1000).map(|i| doc_ids[i * (size / 1000)]).collect();

    let bitpacked = BitpackedPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
    let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);

    group.throughput(Throughput::Elements(seek_targets.len() as u64));

    group.bench_function("bitpacked_seek", |b| {
        b.iter(|| {
            let mut iter = bitpacked.iterator();
            let mut sum = 0u64;
            for &target in &seek_targets {
                sum += iter.seek(target) as u64;
            }
            black_box(sum)
        })
    });

    group.bench_function("elias_fano_seek", |b| {
        b.iter(|| {
            let mut iter = elias_fano.iterator();
            let mut sum = 0u64;
            for &target in &seek_targets {
                sum += iter.seek(target) as u64;
            }
            black_box(sum)
        })
    });

    group.finish();
}

fn bench_unified_format(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_format");

    // Test automatic format selection
    let sizes = [
        (100, 1_000_000, "small_bitpacked"),
        (15_000, 10_000_000, "medium_elias_fano"),
        (50_000, 1_000_000, "large_roaring"),
    ];

    for (size, total_docs, name) in sizes {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("create", name), &size, |b, _| {
            b.iter(|| {
                black_box(CompressedPostingList::from_postings(
                    &doc_ids,
                    &term_freqs,
                    total_docs,
                    1.0,
                ))
            })
        });

        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, total_docs, 1.0);

        group.bench_with_input(BenchmarkId::new("iterate", name), &size, |b, _| {
            b.iter(|| {
                let mut iter = list.iterator();
                let mut sum = 0u64;
                while !iter.is_exhausted() {
                    sum += iter.doc() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

fn bench_deserialization_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialization_speed");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        // Serialize each format
        let bitpacked = BitpackedPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut bp_buf = Vec::new();
        bitpacked.serialize(&mut bp_buf).unwrap();

        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let mut ef_buf = Vec::new();
        elias_fano.serialize(&mut ef_buf).unwrap();

        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);
        let mut roar_buf = Vec::new();
        roaring.serialize(&mut roar_buf).unwrap();

        group.bench_with_input(BenchmarkId::new("bitpacked", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&bp_buf);
                black_box(BitpackedPostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&ef_buf);
                black_box(EliasFanoPostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&roar_buf);
                black_box(RoaringPostingList::deserialize(&mut cursor).unwrap())
            })
        });
    }

    group.finish();
}

fn bench_by_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("by_distribution");
    let size = 10000;

    let distributions = [
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
        Distribution::Clustered,
        Distribution::Sequential,
    ];

    for dist in &distributions {
        let (doc_ids, term_freqs) = generate_postings(size, *dist);
        group.throughput(Throughput::Elements(size as u64));

        // Encoding benchmarks
        group.bench_with_input(
            BenchmarkId::new("encode_bitpacked", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(BitpackedPostingList::from_postings(
                        &doc_ids,
                        &term_freqs,
                        1.0,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("encode_elias_fano", dist.name()),
            &size,
            |b, _| b.iter(|| black_box(EliasFanoPostingList::from_postings(&doc_ids, &term_freqs))),
        );

        group.bench_with_input(
            BenchmarkId::new("encode_roaring", dist.name()),
            &size,
            |b, _| b.iter(|| black_box(RoaringPostingList::from_postings(&doc_ids, &term_freqs))),
        );

        // Iteration benchmarks
        let bitpacked = BitpackedPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);

        group.bench_with_input(
            BenchmarkId::new("iterate_bitpacked", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = bitpacked.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("iterate_elias_fano", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = elias_fano.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("iterate_roaring", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = roaring.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

fn bench_all_formats_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_formats_encoding");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("bitpacked", size), &size, |b, _| {
            b.iter(|| {
                black_box(BitpackedPostingList::from_postings(
                    &doc_ids,
                    &term_freqs,
                    1.0,
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("simd_bp128", size), &size, |b, _| {
            b.iter(|| {
                black_box(SimdBp128PostingList::from_postings(
                    &doc_ids,
                    &term_freqs,
                    1.0,
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| black_box(EliasFanoPostingList::from_postings(&doc_ids, &term_freqs)))
        });

        group.bench_with_input(BenchmarkId::new("partitioned_ef", size), &size, |b, _| {
            b.iter(|| {
                black_box(PartitionedEFPostingList::from_postings(
                    &doc_ids,
                    &term_freqs,
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| black_box(RoaringPostingList::from_postings(&doc_ids, &term_freqs)))
        });
    }

    group.finish();
}

fn bench_all_formats_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_formats_iteration");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        let bitpacked = BitpackedPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let simd_bp128 = SimdBp128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let partitioned_ef = PartitionedEFPostingList::from_postings(&doc_ids, &term_freqs);
        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);

        group.bench_with_input(BenchmarkId::new("bitpacked", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = bitpacked.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("simd_bp128", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = simd_bp128.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = elias_fano.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("partitioned_ef", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = partitioned_ef.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = roaring.iterator();
                iter.init();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

/// Measure encoding speed (returns elements per microsecond)
fn measure_encoding_speed(doc_ids: &[u32], term_freqs: &[u32], iterations: usize) -> [f64; 5] {
    use std::time::Instant;

    let n = doc_ids.len();

    // Bitpacked
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(BitpackedPostingList::from_postings(
            doc_ids, term_freqs, 1.0,
        ));
    }
    let bp_time = start.elapsed().as_micros() as f64;
    let bp_rate = (n * iterations) as f64 / bp_time;

    // SIMD-BP128
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(SimdBp128PostingList::from_postings(
            doc_ids, term_freqs, 1.0,
        ));
    }
    let simd_time = start.elapsed().as_micros() as f64;
    let simd_rate = (n * iterations) as f64 / simd_time;

    // Elias-Fano
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(EliasFanoPostingList::from_postings(doc_ids, term_freqs));
    }
    let ef_time = start.elapsed().as_micros() as f64;
    let ef_rate = (n * iterations) as f64 / ef_time;

    // Partitioned EF
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(PartitionedEFPostingList::from_postings(doc_ids, term_freqs));
    }
    let pef_time = start.elapsed().as_micros() as f64;
    let pef_rate = (n * iterations) as f64 / pef_time;

    // Roaring
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(RoaringPostingList::from_postings(doc_ids, term_freqs));
    }
    let roar_time = start.elapsed().as_micros() as f64;
    let roar_rate = (n * iterations) as f64 / roar_time;

    [bp_rate, simd_rate, ef_rate, pef_rate, roar_rate]
}

/// Measure decoding/iteration speed (returns elements per microsecond)
fn measure_decoding_speed(doc_ids: &[u32], term_freqs: &[u32], iterations: usize) -> [f64; 5] {
    use std::time::Instant;

    let n = doc_ids.len();
    let bp = BitpackedPostingList::from_postings(doc_ids, term_freqs, 1.0);
    let simd = SimdBp128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let ef = EliasFanoPostingList::from_postings(doc_ids, term_freqs);
    let pef = PartitionedEFPostingList::from_postings(doc_ids, term_freqs);
    let roar = RoaringPostingList::from_postings(doc_ids, term_freqs);

    // Bitpacked
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = bp.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let bp_time = start.elapsed().as_micros() as f64;
    let bp_rate = (n * iterations) as f64 / bp_time;

    // SIMD-BP128
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = simd.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let simd_time = start.elapsed().as_micros() as f64;
    let simd_rate = (n * iterations) as f64 / simd_time;

    // Elias-Fano
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = ef.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let ef_time = start.elapsed().as_micros() as f64;
    let ef_rate = (n * iterations) as f64 / ef_time;

    // Partitioned EF
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = pef.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let pef_time = start.elapsed().as_micros() as f64;
    let pef_rate = (n * iterations) as f64 / pef_time;

    // Roaring
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = roar.iterator();
        iter.init();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let roar_time = start.elapsed().as_micros() as f64;
    let roar_rate = (n * iterations) as f64 / roar_time;

    [bp_rate, simd_rate, ef_rate, pef_rate, roar_rate]
}

fn format_rate(rate: f64) -> String {
    if rate >= 1000.0 {
        format!("{:5.1}M", rate / 1000.0)
    } else if rate >= 1.0 {
        format!("{:5.1}K", rate)
    } else {
        format!("{:5.2}", rate * 1000.0)
    }
}

fn find_best_idx(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn find_min_idx(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn bench_all_formats_compression_summary(_c: &mut Criterion) {
    let format_names = ["Bitpack", "SIMD-BP", "EF", "PEF", "Roaring"];
    let format_short = ["BP", "SIMD", "EF", "PEF", "Roar"];

    println!("\n🚀 POSTING LIST COMPRESSION BENCHMARK RESULTS");
    println!("Formats: Bitpack, SIMD-BP128, EF (Elias-Fano), PEF (Partitioned EF), Roaring\n");

    // ═══════════════════════════════════════════════════════════════════════════════
    // COMPRESSION RATIO TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut compression_table = Table::new();
    compression_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Distribution",
            "Size",
            "Bitpack",
            "SIMD-BP",
            "EF",
            "PEF",
            "Roaring",
            "Best",
            "Bits/Doc",
        ]);

    let distributions = [
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
        Distribution::Clustered,
        Distribution::Sequential,
    ];
    let sizes = [1000, 10000, 50000];

    let mut total_ratios = [0.0f64; 5];
    let mut count = 0;

    for dist in &distributions {
        for &size in &sizes {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let result = measure_compression(&doc_ids, &term_freqs);

            let ratios = [
                result.bitpacked_ratio(),
                result.simd_bp128_ratio(),
                result.elias_fano_ratio(),
                result.partitioned_ef_ratio(),
                result.roaring_ratio(),
            ];

            for i in 0..5 {
                total_ratios[i] += ratios[i];
            }
            count += 1;

            let best_idx = find_min_idx(&ratios);
            let best_bytes = match best_idx {
                0 => result.bitpacked_bytes,
                1 => result.simd_bp128_bytes,
                2 => result.elias_fano_bytes,
                3 => result.partitioned_ef_bytes,
                _ => result.roaring_bytes,
            };
            let bits_per_doc = (best_bytes * 8) as f64 / size as f64;

            compression_table.add_row(vec![
                dist.name(),
                &size.to_string(),
                &format!("{:.1}%", ratios[0]),
                &format!("{:.1}%", ratios[1]),
                &format!("{:.1}%", ratios[2]),
                &format!("{:.1}%", ratios[3]),
                &format!("{:.1}%", ratios[4]),
                format_short[best_idx],
                &format!("{:.2}", bits_per_doc),
            ]);
        }
    }

    let avg: Vec<f64> = total_ratios.iter().map(|r| r / count as f64).collect();
    let best_avg_idx = find_min_idx(&avg);
    compression_table.add_row(vec![
        "AVERAGE",
        "ALL",
        &format!("{:.1}%", avg[0]),
        &format!("{:.1}%", avg[1]),
        &format!("{:.1}%", avg[2]),
        &format!("{:.1}%", avg[3]),
        &format!("{:.1}%", avg[4]),
        format_short[best_avg_idx],
        &format!("Winner: {}", format_names[best_avg_idx]),
    ]);

    println!("📊 COMPRESSION RATIO (% of raw size - lower is better)");
    println!("{compression_table}");

    // ═══════════════════════════════════════════════════════════════════════════════
    // ENCODING SPEED TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut encoding_table = Table::new();
    encoding_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Distribution",
            "Size",
            "Bitpack",
            "SIMD-BP",
            "EF",
            "PEF",
            "Roaring",
            "Fastest",
        ]);

    let mut total_enc_rates = [0.0f64; 5];
    let mut enc_count = 0;

    for dist in &[
        Distribution::Medium,
        Distribution::Sparse,
        Distribution::Dense,
    ] {
        for &size in &[1000, 10000, 100000] {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let iterations = if size < 5000 {
                500
            } else if size < 50000 {
                100
            } else {
                50
            };
            let rates = measure_encoding_speed(&doc_ids, &term_freqs, iterations);

            for i in 0..5 {
                total_enc_rates[i] += rates[i];
            }
            enc_count += 1;

            let best_idx = find_best_idx(&rates);

            encoding_table.add_row(vec![
                dist.name(),
                &size.to_string(),
                &format_rate(rates[0]),
                &format_rate(rates[1]),
                &format_rate(rates[2]),
                &format_rate(rates[3]),
                &format_rate(rates[4]),
                format_short[best_idx],
            ]);
        }
    }

    let avg_enc: Vec<f64> = total_enc_rates
        .iter()
        .map(|r| r / enc_count as f64)
        .collect();
    let best_enc_idx = find_best_idx(&avg_enc);
    encoding_table.add_row(vec![
        "AVERAGE",
        "ALL",
        &format_rate(avg_enc[0]),
        &format_rate(avg_enc[1]),
        &format_rate(avg_enc[2]),
        &format_rate(avg_enc[3]),
        &format_rate(avg_enc[4]),
        format_short[best_enc_idx],
    ]);

    println!("\n⚡ ENCODING SPEED (ints/μs - higher is better)");
    println!("{encoding_table}");

    // ═══════════════════════════════════════════════════════════════════════════════
    // DECODING SPEED TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut decoding_table = Table::new();
    decoding_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Distribution",
            "Size",
            "Bitpack",
            "SIMD-BP",
            "EF",
            "PEF",
            "Roaring",
            "Fastest",
        ]);

    let mut total_dec_rates = [0.0f64; 5];
    let mut dec_count = 0;

    for dist in &[
        Distribution::Medium,
        Distribution::Sparse,
        Distribution::Dense,
    ] {
        for &size in &[1000, 10000, 100000] {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let iterations = if size < 5000 {
                2000
            } else if size < 50000 {
                500
            } else {
                100
            };
            let rates = measure_decoding_speed(&doc_ids, &term_freqs, iterations);

            for i in 0..5 {
                total_dec_rates[i] += rates[i];
            }
            dec_count += 1;

            let best_idx = find_best_idx(&rates);

            decoding_table.add_row(vec![
                dist.name(),
                &size.to_string(),
                &format_rate(rates[0]),
                &format_rate(rates[1]),
                &format_rate(rates[2]),
                &format_rate(rates[3]),
                &format_rate(rates[4]),
                format_short[best_idx],
            ]);
        }
    }

    let avg_dec: Vec<f64> = total_dec_rates
        .iter()
        .map(|r| r / dec_count as f64)
        .collect();
    let best_dec_idx = find_best_idx(&avg_dec);
    decoding_table.add_row(vec![
        "AVERAGE",
        "ALL",
        &format_rate(avg_dec[0]),
        &format_rate(avg_dec[1]),
        &format_rate(avg_dec[2]),
        &format_rate(avg_dec[3]),
        &format_rate(avg_dec[4]),
        format_short[best_dec_idx],
    ]);

    println!("\n🔄 DECODING SPEED (ints/μs - higher is better)");
    println!("{decoding_table}");

    // Legend
    println!("\n📖 DISTRIBUTION LEGEND:");
    println!("  • sparse_1pct  : 1% density - rare terms");
    println!("  • medium_10pct : 10% density - typical terms");
    println!("  • dense_50pct  : 50% density - common terms");
    println!("  • clustered    : docs grouped in ranges");
    println!("  • sequential   : consecutive doc_ids");
}

criterion_group!(
    benches,
    bench_encoding_speed,
    bench_iteration_speed,
    bench_seek_speed,
    bench_unified_format,
    bench_deserialization_speed,
    bench_by_distribution,
    bench_all_formats_encoding,
    bench_all_formats_iteration,
    bench_all_formats_compression_summary,
);
criterion_main!(benches);
