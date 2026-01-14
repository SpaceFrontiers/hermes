//! Comprehensive benchmarks for posting list compression methods
//!
//! Compares: Bitpacked, Elias-Fano, Roaring
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

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hermes_core::structures::{
    BitpackedPostingList, CompressedPostingList, EliasFanoPostingList, RoaringPostingList,
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
            let num_clusters = (count + cluster_size - 1) / cluster_size;
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
    elias_fano_bytes: usize,
    roaring_bytes: usize,
}

impl CompressionResult {
    fn bitpacked_ratio(&self) -> f64 {
        self.bitpacked_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    fn elias_fano_ratio(&self) -> f64 {
        self.elias_fano_bytes as f64 / self.raw_bytes as f64 * 100.0
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

    let elias_fano = EliasFanoPostingList::from_postings(doc_ids, term_freqs);
    let mut ef_buf = Vec::new();
    elias_fano.serialize(&mut ef_buf).unwrap();

    let roaring = RoaringPostingList::from_postings(doc_ids, term_freqs);
    let mut roar_buf = Vec::new();
    roaring.serialize(&mut roar_buf).unwrap();

    CompressionResult {
        raw_bytes,
        bitpacked_bytes: bp_buf.len(),
        elias_fano_bytes: ef_buf.len(),
        roaring_bytes: roar_buf.len(),
    }
}

fn bench_compression_ratio(_c: &mut Criterion) {
    println!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                    COMPRESSION RATIO BY DISTRIBUTION & SIZE                      ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ Distribution   │  Size  │   Raw   │ Bitpacked │ Elias-Fano │  Roaring  │  Best   ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );

    let distributions = [
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
        Distribution::Clustered,
        Distribution::Sequential,
    ];
    let sizes = [1000, 10000, 50000];

    // Accumulators for averages
    let mut total_bp = 0.0;
    let mut total_ef = 0.0;
    let mut total_roar = 0.0;
    let mut count = 0;

    for dist in &distributions {
        for &size in &sizes {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let result = measure_compression(&doc_ids, &term_freqs);

            let bp_ratio = result.bitpacked_ratio();
            let ef_ratio = result.elias_fano_ratio();
            let roar_ratio = result.roaring_ratio();

            total_bp += bp_ratio;
            total_ef += ef_ratio;
            total_roar += roar_ratio;
            count += 1;

            // Find best
            let best = if bp_ratio <= ef_ratio && bp_ratio <= roar_ratio {
                "BP"
            } else if ef_ratio <= roar_ratio {
                "EF"
            } else {
                "Roar"
            };

            println!(
                "║ {:14} │ {:6} │ {:7} │ {:6.1}%   │  {:6.1}%   │  {:6.1}%  │  {:5}  ║",
                dist.name(),
                size,
                result.raw_bytes,
                bp_ratio,
                ef_ratio,
                roar_ratio,
                best
            );
        }
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════════╣"
        );
    }

    // Print averages
    let avg_bp = total_bp / count as f64;
    let avg_ef = total_ef / count as f64;
    let avg_roar = total_roar / count as f64;

    println!(
        "║ AVERAGE        │   ALL  │    -    │ {:6.1}%   │  {:6.1}%   │  {:6.1}%  │         ║",
        avg_bp, avg_ef, avg_roar
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Summary insights
    println!("\n📊 INSIGHTS:");
    println!("  • Bitpacked: Best for medium-density random distributions");
    println!("  • Elias-Fano: Best for sparse data (large gaps between doc_ids)");
    println!("  • Roaring: Best for dense/sequential data (bitmap containers)");
    println!("  • Sequential doc_ids compress extremely well with all methods");
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

fn bench_comprehensive_summary(_c: &mut Criterion) {
    println!("\n");
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                           COMPREHENSIVE POSTING LIST BENCHMARK SUMMARY                               ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let distributions = [
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
        Distribution::Clustered,
        Distribution::Sequential,
    ];
    let sizes = [1000, 10000, 50000];

    // Header
    println!(
        "║                                                                                                      ║"
    );
    println!(
        "║  COMPRESSION RATIO (% of raw size, lower is better)                                                  ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ Distribution   │  Size  │ Bitpacked │ Elias-Fano │  Roaring  │ Bits/Doc (BP) │ Bits/Doc (EF) │ Best  ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for dist in &distributions {
        for &size in &sizes {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let result = measure_compression(&doc_ids, &term_freqs);

            let bp_ratio = result.bitpacked_ratio();
            let ef_ratio = result.elias_fano_ratio();
            let roar_ratio = result.roaring_ratio();

            // Bits per doc_id
            let bp_bits_per_doc = (result.bitpacked_bytes * 8) as f64 / size as f64;
            let ef_bits_per_doc = (result.elias_fano_bytes * 8) as f64 / size as f64;

            let best = if bp_ratio <= ef_ratio && bp_ratio <= roar_ratio {
                "BP"
            } else if ef_ratio <= roar_ratio {
                "EF"
            } else {
                "Roar"
            };

            println!(
                "║ {:14} │ {:6} │   {:5.1}%   │    {:5.1}%   │   {:5.1}%   │     {:5.1}      │     {:5.1}      │ {:5} ║",
                dist.name(),
                size,
                bp_ratio,
                ef_ratio,
                roar_ratio,
                bp_bits_per_doc,
                ef_bits_per_doc,
                best
            );
        }
    }

    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║                                                                                                      ║"
    );
    println!(
        "║  MEMORY OVERHEAD ANALYSIS (for 10K docs)                                                             ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let size = 10000;
    for dist in &distributions {
        let (doc_ids, term_freqs) = generate_postings(size, *dist);
        let result = measure_compression(&doc_ids, &term_freqs);

        let raw_kb = result.raw_bytes as f64 / 1024.0;
        let bp_kb = result.bitpacked_bytes as f64 / 1024.0;
        let ef_kb = result.elias_fano_bytes as f64 / 1024.0;
        let roar_kb = result.roaring_bytes as f64 / 1024.0;

        println!(
            "║ {:14} │ Raw: {:5.1} KB │ BP: {:5.1} KB │ EF: {:5.1} KB │ Roaring: {:5.1} KB                        ║",
            dist.name(),
            raw_kb,
            bp_kb,
            ef_kb,
            roar_kb
        );
    }

    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║                                                                                                      ║"
    );
    println!(
        "║  RECOMMENDATIONS BY USE CASE                                                                         ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║                                                                                                      ║"
    );
    println!(
        "║  • RARE TERMS (sparse, <1% density):     Elias-Fano - best compression, good seek                    ║"
    );
    println!(
        "║  • TYPICAL TERMS (medium, ~10% density): Bitpacked - balanced speed/compression                      ║"
    );
    println!(
        "║  • COMMON TERMS (dense, >30% density):   Roaring - bitmap containers excel                           ║"
    );
    println!(
        "║  • SEQUENTIAL DOC_IDS:                   Any format - all compress well                              ║"
    );
    println!(
        "║  • CLUSTERED DATA:                       Roaring - run-length encoding helps                         ║"
    );
    println!(
        "║  • FAST ITERATION PRIORITY:              Bitpacked - cache-friendly blocks                           ║"
    );
    println!(
        "║  • FAST SEEK PRIORITY:                   Elias-Fano - O(1) random access                             ║"
    );
    println!(
        "║                                                                                                      ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
}

criterion_group!(
    benches,
    bench_compression_ratio,
    bench_encoding_speed,
    bench_iteration_speed,
    bench_seek_speed,
    bench_unified_format,
    bench_deserialization_speed,
    bench_by_distribution,
    bench_comprehensive_summary,
);
criterion_main!(benches);
