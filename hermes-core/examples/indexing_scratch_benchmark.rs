//! Focused full-text ingestion benchmark, excluding flush/ANN/merge work.
//! Run with `cargo run -p hermes-core --release --example indexing_scratch_benchmark`.
//! Preload a growing vocabulary, then index 1,000 documents with eight chunks
//! and an unpositioned text field. Report elapsed ingestion time, not setup.

use hermes_core::dsl::PositionMode;
use hermes_core::segment::{SegmentBuilder, SegmentBuilderConfig};
use hermes_core::tokenizer::SimpleTokenizer;
use hermes_core::{Document, SchemaBuilder};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    for custom in [false, true] {
        for vocabulary in [1_000, 10_000, 100_000] {
            let mut schema = SchemaBuilder::default();
            let body = schema.add_text_field("body", true, false);
            schema.set_positions(body, PositionMode::TokenPosition);
            schema.set_chunked(body, true);
            let title = schema.add_text_field("title", true, false);
            let mut builder =
                SegmentBuilder::new(Arc::new(schema.build()), SegmentBuilderConfig::default())
                    .unwrap();
            if custom {
                builder.set_tokenizer(body, Box::new(SimpleTokenizer));
            }
            let mut seed = Document::new();
            seed.add_text(
                body,
                (0..vocabulary)
                    .map(|i| format!("seed{i} "))
                    .collect::<String>(),
            );
            // Clear the large seed's scratch before timing short chunks.
            seed.add_text(title, "warmup");
            builder.add_document(seed).unwrap();
            let docs: Vec<_> = (0..1_000)
                .map(|i| {
                    let mut doc = Document::new();
                    for chunk in 0..8 {
                        doc.add_text(body, format!("anchor repeat repeat fresh{i} chunk{chunk}"));
                    }
                    doc.add_text(title, "anchor title");
                    doc
                })
                .collect();
            let start = Instant::now();
            for doc in docs {
                builder.add_document(doc).unwrap();
            }
            let elapsed = start.elapsed();
            assert_eq!(builder.num_docs(), 1_001);
            println!(
                "custom={custom} vocabulary={vocabulary} docs=1000 chunks_per_doc=8 elapsed_ms={:.3} docs_per_sec={:.0}",
                elapsed.as_secs_f64() * 1_000.0,
                1_000.0 / elapsed.as_secs_f64()
            );
        }
    }
}
