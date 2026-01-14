//! Main entry point for posting compression benchmarks

use criterion::{criterion_group, criterion_main};

mod common;
mod deserialization;
mod distribution;
mod encoding;
mod iteration;
mod seek;
mod summary;

use deserialization::bench_deserialization_speed;
use distribution::{bench_by_distribution, bench_unified_format};
use encoding::{bench_all_formats_encoding, bench_encoding_speed};
use iteration::{bench_all_formats_iteration, bench_iteration_speed};
use seek::bench_seek_speed;
use summary::bench_all_formats_compression_summary;

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
