//! Deterministic two-source merge, with fixture construction outside timing.
//! RAM isolates merge CPU/allocation costs; this does not measure cold disk I/O.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hermes_core::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use hermes_core::{Document, RamDirectory, SchemaBuilder};

fn bench_segment_merge(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("segment_merge");
    group.sample_size(20);
    for count in [4_096u32, 65_536] {
        for missing in [false, true] {
            let dir = RamDirectory::new();
            let mut sb = SchemaBuilder::default();
            let field = sb.add_u64_field("sequence", false, false);
            sb.set_fast(field, true);
            sb.set_multi(field, true);
            let schema = Arc::new(sb.build());
            // Simulate an older source without this optional column. Keep
            // identical field IDs and types while disabling its fast storage.
            let mut old = SchemaBuilder::default();
            assert_eq!(old.add_u64_field("sequence", false, false), field);
            old.set_multi(field, true);
            let old_schema = Arc::new(old.build());
            let sources = runtime.block_on(async {
                let mut readers = Vec::new();
                for source in 0..2 {
                    let source_schema = if missing && source == 0 {
                        Arc::clone(&old_schema)
                    } else {
                        Arc::clone(&schema)
                    };
                    let mut builder =
                        SegmentBuilder::new(source_schema, SegmentBuilderConfig::default())
                            .unwrap();
                    for i in 0..count {
                        let mut doc = Document::new();
                        if !missing || source != 0 {
                            doc.add_u64(field, u64::from(i) * 17);
                            doc.add_u64(field, u64::from(i) * 31);
                        }
                        builder.add_document(doc).unwrap();
                    }
                    let id = SegmentId::new();
                    builder.build(&dir, id, None).await.unwrap();
                    readers.push(
                        SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                            .await
                            .unwrap(),
                    );
                }
                assert_eq!(readers[0].fast_field(field.0).is_none(), missing);
                readers
            });
            let merger = SegmentMerger::new(schema);
            // Overwrite the same unpublished output so iterations do not
            // retain an unbounded collection of output segments in RAM.
            let output = SegmentId::new();
            let (meta, _) = runtime
                .block_on(merger.merge(&dir, &sources, output, None))
                .unwrap();
            assert_eq!(meta.num_docs, 2 * count);
            group.throughput(Throughput::Elements(u64::from(count) * 2));
            group.bench_with_input(
                BenchmarkId::new(
                    if missing {
                        "missing_fast_column"
                    } else {
                        "copy_fast_columns"
                    },
                    count,
                ),
                &count,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            runtime
                                .block_on(merger.merge(&dir, &sources, output, None))
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_segment_merge);
criterion_main!(benches);
