//! Real range materialization plus isolated closure/code-generation probes.
//! Build/open/merge and membership assertions are outside timed loops.

use std::cmp::Ordering;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hermes_core::query::{DocBitset, HeapEntry, Query, RangeQuery};
use hermes_core::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use hermes_core::structures::fast_field::codec::bitpacked_read_batch;
use hermes_core::{Document, Field, RamDirectory, SchemaBuilder};

const DOCUMENTS: u32 = 65_536;

fn value(doc: u32) -> u64 {
    // Permutation of [0, 65536); shuffled selectivity within each block.
    u64::from(doc.wrapping_mul(40_503) & 65_535)
}

async fn range_fixture(blocks: u32) -> (SegmentReader, Field) {
    let dir = RamDirectory::new();
    let mut sb = SchemaBuilder::default();
    let field = sb.add_u64_field("value", false, false);
    sb.set_fast(field, true);
    let schema = Arc::new(sb.build());
    let mut sources = Vec::new();
    for block in 0..blocks {
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();
        for local in 0..DOCUMENTS / blocks {
            let mut doc = Document::new();
            doc.add_u64(field, value(block * (DOCUMENTS / blocks) + local));
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        sources.push(
            SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                .await
                .unwrap(),
        );
    }
    let reader = if blocks == 1 {
        sources.pop().unwrap()
    } else {
        let id = SegmentId::new();
        SegmentMerger::new(Arc::clone(&schema))
            .merge(&dir, &sources, id, None)
            .await
            .unwrap();
        SegmentReader::open(&dir, id, schema, 0).await.unwrap()
    };
    assert_eq!(
        reader.fast_field(field.0).unwrap().num_blocks(),
        blocks as usize
    );
    (reader, field)
}

fn bench_range(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("rust_hot_paths/range_bitset");
    group.sample_size(20);
    group.throughput(Throughput::Elements(u64::from(DOCUMENTS)));
    for blocks in [1, 16] {
        let (reader, field) = runtime.block_on(range_fixture(blocks));
        for upper in [655u64, 32_767] {
            let query = RangeQuery::u64(field, Some(0), Some(upper));
            let actual = query.as_doc_bitset(&reader).unwrap();
            let predicate = query.as_doc_predicate(&reader).unwrap();
            let original = DocBitset::from_predicate(DOCUMENTS, &*predicate);
            for doc in 0..DOCUMENTS {
                let expected = value(doc) <= upper;
                assert_eq!(actual.contains(doc), expected);
                assert_eq!(original.contains(doc), expected);
            }
            let fixture = format!("{blocks}_blocks_max_{upper}");
            group.bench_function(BenchmarkId::new("production", &fixture), |b| {
                b.iter(|| black_box(black_box(&query).as_doc_bitset(black_box(&reader)).unwrap()));
            });
            group.bench_function(BenchmarkId::new("predicate_control", &fixture), |b| {
                b.iter(|| {
                    let predicate = black_box(&query)
                        .as_doc_predicate(black_box(&reader))
                        .unwrap();
                    black_box(DocBitset::from_predicate(DOCUMENTS, &*predicate))
                });
            });
        }
    }
    group.finish();
}

// These symbols deliberately remain identifiable in optimized assembly. Do not
// add inline(never) to production callbacks just to reproduce these experiments.
#[inline(never)]
fn count_static(values: &[u64], predicate: impl Fn(u64) -> bool) -> usize {
    values.iter().filter(|&&v| predicate(v)).count()
}

#[inline(never)]
fn count_dynamic(values: &[u64], predicate: &dyn Fn(u64) -> bool) -> usize {
    values.iter().filter(|&&v| predicate(v)).count()
}

#[inline(never)]
fn heap_order(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    left.cmp(right)
}

fn bench_dispatch(c: &mut Criterion) {
    let values: Vec<_> = (0..DOCUMENTS).map(value).collect();
    let upper = black_box(32_767);
    let predicate = |v| v <= upper;
    assert_eq!(count_static(&values, predicate), 32_768);
    assert_eq!(count_dynamic(&values, &predicate), 32_768);
    let mut group = c.benchmark_group("rust_hot_paths/decoded_predicate");
    group.sample_size(20);
    group.throughput(Throughput::Elements(u64::from(DOCUMENTS)));
    group.bench_function("generic_closure", |b| {
        b.iter(|| black_box(count_static(black_box(&values), predicate)));
    });
    group.bench_function("opaque_dyn_fn", |b| {
        b.iter(|| black_box(count_dynamic(black_box(&values), black_box(&predicate))));
    });
    group.finish();

    let left = HeapEntry {
        doc_id: 7,
        score: 1.0,
        ordinal: 0,
    };
    let right = HeapEntry { doc_id: 8, ..left };
    assert_eq!(
        heap_order(black_box(&left), black_box(&right)),
        Ordering::Less
    );
    eprintln!(
        "layout bytes: HeapEntry={}, DocBitset={}, SharedThreshold={}, ColumnBlock={}, FastFieldReader={}",
        size_of::<HeapEntry>(),
        size_of::<DocBitset>(),
        size_of::<hermes_core::query::SharedThreshold>(),
        size_of::<hermes_core::structures::fast_field::ColumnBlock>(),
        size_of::<hermes_core::structures::fast_field::FastFieldReader>()
    );
}

fn bench_byte_aligned_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_hot_paths/bitpacked_batch");
    group.sample_size(20);
    group.throughput(Throughput::Elements(256));
    for bits in [8u8, 16, 32, 64] {
        // Valid bitpacked payload, excluding the auto-codec tag: min + bpv + data.
        let mut encoded = 17u64.to_le_bytes().to_vec();
        encoded.push(bits);
        let mut expected = Vec::new();
        let mask = u64::MAX >> (64 - bits);
        for i in 0..256u64 {
            let raw = i.wrapping_mul(0x9e37_79b9_7f4a_7c15) & mask;
            encoded.extend_from_slice(&raw.to_le_bytes()[..usize::from(bits / 8)]);
            expected.push(raw.wrapping_add(17));
        }
        let mut output = [0u64; 256];
        bitpacked_read_batch(&encoded, 0, &mut output);
        assert_eq!(output.as_slice(), expected);
        group.bench_function(BenchmarkId::from_parameter(bits), |b| {
            b.iter(|| {
                bitpacked_read_batch(black_box(&encoded), black_box(0), &mut output);
                black_box(&output);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_range,
    bench_dispatch,
    bench_byte_aligned_decode
);
criterion_main!(benches);
