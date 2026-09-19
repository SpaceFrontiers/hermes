# Hermes documentation

Start with the [repository quick start](../README.md#quick-start) and
[contribution guide](../CONTRIBUTING.md). This index covers every guide in this
directory. Package READMEs below document each executable or library.

The schema and operational guides describe current behavior. Research notes,
implementation ledgers, and dated benchmark tables also retain proposed or
rejected work; read their status and measurement context before treating a
claim as implemented or as current performance.

## Getting started and interfaces

- [Benchmark guide](benchmarks.md)

## Search engineering and performance

- [Search system engineering contract](search-system-contract.md)
- [Current full-text benchmark comparison](search-benchmark-current.md)
- [Core/server review](search-performance-review.md)
- [Search Benchmark, the Game comparison](search-benchmark-game.md)
- [Wikipedia benchmark results and evidence](search-benchmark-results.md)
- [Score-bound follow-up and exact-ranking evidence](search-benchmark-ratio-results.md)
- [Lucene 11 performance research](lucene-11-performance-research.md): pinned source findings and Hermes experiments.
- [Bulk-scoring follow-up](search-benchmark-bulk-results.md) — full-corpus exact-count batching results and remaining gaps.
- [Bounded posting validation reuse](search-benchmark-validation-results.md): correctness, resource policy and measurements.
- [Dictionary block and allocation experiments](search-benchmark-dictionary-results.md)
- [L1 candidate scoring handoff and rejected lookup design](handoffs/2026-09-05-l1-candidate-scoring.md)
- [Rust hot-path review](rust-hot-path-review.md)

## Schema, text search, and query behavior

- [Hermes Schema Definition Language (SDL)](schema.md)
- [Query language](query-language.md): required/prohibited clauses, precedence, and strict parsing.
- [Dynamic per-document stemming and wire-level phrase queries](dynamic-tokenizer-and-phrase.md)
- [Chunked text fields: BM25 over passages with ordinals](chunked-text-fields.md)
- [BM25 over equal-length chunks](chunked-bm25.md)
- [Lexical vertical: positions, pruning, reordering, tokenization](lexical-vertical.md)
- [Posting block codecs](posting-codecs.md)
- [Compact text storage and quantized norms](compact-text-format.md): versioned opt-in formats, normalization and compatibility.
- [MaxScore text reordering: design and implementation status](maxscore-text-reordering.md)
- [Hermes Web UX Configuration DSL](ux-config.md)

## Storage, operations, and distributed search

- [Hermes broker](broker.md)
- [Segment Lifecycle and Recovery](segment-lifecycle.md)
- [Row deletion, upserts, and compaction](row-deletion.md)
- [Content-hash deduplication](content-deduplication.md)
- [Index diagnostics](diagnostics.md)
- [Prometheus Metrics](metrics.md)
- [Document store v3](document-store-v3.md)
- [Owned byte views and their safety contract](owned-byte-views.md)
- [Posting block execution and Tantivy comparison](search-block-execution.md)
- [Standalone RGB benchmark and mapping diagnosis](search-rgb-benchmark.md)
- [RGB execution repair and measured results](search-rgb-repair.md)
- [Hot-Metadata Pinning (meta/data residency split)](hot-metadata-pinning.md)
- [Cold IO for merges (hot-metadata-pinning Phase 2)](cold-io.md)
- [Merge-Time BP Reordering](merge-time-reorder.md)
- [Budgeted (Partial) BP Reordering](budgeted-reorder.md)
- [Reordering Performance Review](reordering-performance-review.md)
- [Block-Level Reorder with Stats-Guided Granularity](block-level-reorder.md)

## Vector retrieval and compression

- [Streaming ScaNN index](scann-streaming-index.md)
- [Single-copy binary ANN storage and streaming merge](binary-vector-storage.md)
- [FastScan layout v2 for float ScaNN leaves](fast-scan-layout-v2.md)
- [TurboQuant (TQ) — training-free dense ANN codec](turboquant-quantization.md)
- [Unified Dense IVF Architecture](unified-vector-quantization.md)
- [Seismic sparse indexing](seismic-sparse-index.md) — optional third sparse algorithm alongside default BMP and MaxScore; exact values, copy merges and bounded maintenance.
- [Compact Seismic summaries](seismic-compact-summaries.md) — measured lossless directory compression, memory/latency tradeoffs and deferred locality experiments.
- [Seismic forward dimension compression](seismic-forward-compression.md) — lossless U24 and aligned gap encoding with full U32 IDs.
- [Forward values in BMP search passes: research and experiment](bmp-forward-search.md)
- [BMP forward values and format compatibility](bmp-forward-index.md)
- [BMP LSP/0 and Maximum-Grid Compression](bmp-grid-compression.md)
- [Algebraic float reductions](algebraic-float-reductions.md)
- [Seismic: research assessment (2026-07-09)](seismic-research.md)

## Language models, training, and evaluation

- [LLM inference and training code map](llm-code-map.md)
- [LLM compute architecture](uni-stack-inference.md)
- [MAL parser architecture](mal-single-parser.md)
- [Tokenizer backend compatibility](tokenizer-backends.md)
- [Upstream dependency pins](upstream-dependencies.md)
- [LLM Visualization Lab](llm-visualization-lab.md)
- [Training workflows and task contracts](training-objectives-and-curricula.md)
- [Generation evaluation](generation-eval.md)
- [Candidate backfill and linear L1 ranking](candidate-rescoring.md)
- [Passage evidence for document nominations](document-nominated-passages.md)
- [Large-candidate-pool retrieval evaluation](retrieval-pool-eval.md)
- [Configurable MoE design](moe-design.md)
- [SOTA LLM design (2024–2026) + shared retrieval embeddings — research notes](llm-design-and-rag-embeddings.md)
- [RL Training Pipeline for Agentic Search over Hermes](rl-search-training.md)

## Accelerator implementation and measurements

- [MoE A100 performance](moe-performance.md)
- [Attention kernels](fused-attention.md)
- [Fused chunked cross-entropy (GPU)](fused-cross-entropy.md)
- [Segment-parallel selective scan](segmented-selective-scan.md)
- [BF16 residual stream (CUDA training)](bf16-residual-stream.md)
- [Native cuBLAS GEMM dispatch: proof results and verdict](cublas-gemm-dispatch.md)
- [Kernel size-generality and tuning surface](kernel-tuning-surface.md)

## Package guides

- [hermes-broker](../hermes-broker/README.md)
- [hermes-client-python](../hermes-client-python/README.md)
- [hermes-client-typescript](../hermes-client-typescript/README.md)
- [hermes-core](../hermes-core/README.md)
- [hermes-llm](../hermes-llm/README.md)
- [hermes-mal](../hermes-mal/README.md)
- [hermes-mal-python](../hermes-mal-python/README.md)
- [hermes-model-lab](../hermes-model-lab/README.md)
- [hermes-proto](../hermes-proto/README.md)
- [hermes-server](../hermes-server/README.md)
- [hermes-tokenizer](../hermes-tokenizer/README.md)
- [hermes-tool](../hermes-tool/README.md)
- [hermes-train](../hermes-train/README.md)
- [hermes-wasm](../hermes-wasm/README.md)
- [hermes-web](../hermes-web/README.md)

## Keeping documentation current

Run `uv run scripts/check_docs.py` from the repository root to validate local
links and heading anchors, guide coverage here, and the benchmark inventory.
External links need a separate network review. For results, follow the
[benchmark reporting requirements](benchmarks.md#recorded-results-and-reporting).

- [Text index formats and memory comparison](text-format-comparison.md)

- [Query work diagnostics](query-work-diagnostics.md): separate traversal volume from per-unit cost.
- [Measured query-work diagnosis](search-work-diagnosis.md): pruning, payload and scoring cost findings.
- [Pruning and scoring setup fixes](search-pruning-fixes.md): count-aware norm setup, prepared bounds and packed decoding.

- [Search merge review](search-merge-review.md) — September 17 cleanup, format reuse, BMP applicability and validation.
