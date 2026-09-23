//! Read-only diagnostics: exact collector versus exhaustive scorer enumeration.
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};
use hermes_core::directories::MmapDirectory;
use hermes_core::query::{Collector, CountCollector, TopKCollector, collect_segment};
use hermes_core::tokenizer::{Purpose, TokenizerRegistry};
use hermes_core::{Index, IndexConfig};
use serde_json::{Value, json};

pub async fn run(path: &Path, input: &Path, config: IndexConfig) -> Result<()> {
    let index = Index::open(MmapDirectory::new(path), config).await?;
    let searcher = index.reader().await?.searcher().await?;
    let field = searcher
        .schema()
        .get_field("body")
        .context("missing body")?;
    let id = searcher.schema().get_field("id").context("missing id")?;
    let tokenizer = TokenizerRegistry::new()
        .get(super::corpus::BODY_TOKENIZER)
        .context("invalid tokenizer")?;
    let parser = searcher.query_parser();
    let source = std::io::BufReader::new(std::fs::File::open(input)?);
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in source.lines() {
        let mut value: Value = serde_json::from_str(&line?)?;
        if let Some(text) = value.get("text").and_then(Value::as_str) {
            let tokens: Vec<_> = tokenizer
                .tokenize_with(text, None, Purpose::Exact)
                .into_iter()
                .map(|t| json!({"term":t.text,"position":t.position}))
                .collect();
            writeln!(output, "{}", json!({"text":text,"tokens":tokens}))?;
            continue;
        }
        value["limit"] = json!(0);
        let request =
            super::query::Envelope::from_value(value.clone())?.parse(&parser, field, &tokenizer)?;
        let check_ranked = value["audit_ranked"].as_bool().unwrap_or(false);
        anyhow::ensure!(
            !check_ranked || searcher.num_segments() == 1,
            "rank audit requires one segment"
        );
        let mut oracle = TopKCollector::new(100);
        let mut optimized = 0;
        let mut exhaustive = 0u64;
        let mut sample = Vec::new();
        for segment in searcher.segment_readers() {
            let mut collector = CountCollector::new();
            collect_segment(segment, request.query.as_ref(), &mut collector).await?;
            optimized += collector.count();
            let column = segment.fast_field(id.0).context("missing id column")?;
            let mut scorer = request.query.scorer_sync(segment, 0)?;
            while scorer.doc() != hermes_core::structures::TERMINATED {
                if sample.len() < 10 {
                    sample.push(
                        column
                            .get_text(scorer.doc())
                            .context("missing external id")?
                            .to_owned(),
                    );
                }
                if check_ranked {
                    oracle.collect(scorer.doc(), scorer.score(), &[]);
                }
                exhaustive += 1;
                scorer.advance();
            }
        }
        let ranked = if check_ranked {
            let actual = searcher
                .search_with_offset_and_count_sync(request.query.as_ref(), 100, 0)?
                .0;
            let expected = oracle.into_sorted_results();
            anyhow::ensure!(
                actual.len() == expected.len()
                    && actual.iter().zip(&expected).all(
                        |(a, b)| a.doc_id == b.doc_id && a.score.to_bits() == b.score.to_bits()
                    ),
                "ranked results differ from exhaustive oracle"
            );
            let column = searcher.segment_readers()[0]
                .fast_field(id.0)
                .context("missing id column")?;
            actual.iter().map(|hit| Ok(json!({"id": column.get_text(hit.doc_id).context("missing id")?, "score_bits": hit.score.to_bits()}))).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        writeln!(
            output,
            "{}",
            json!({"query":value["query"],"class":value["class"],
            "plan":request.query.to_string(),"optimized":optimized,"exhaustive":exhaustive,"sample":sample,"ranked":ranked})
        )?;
        output.flush()?;
    }
    Ok(())
}
