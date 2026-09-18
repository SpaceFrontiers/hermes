# Compact text formats and byte norms — September 16, 2026

## Outcome and scope

Implemented opt-in compact posting/position directories and byte4 document
norms with query-local BM25 normalization lookups. Both configuration defaults
remain false. RGB is disabled throughout. File savings are 96.589 MiB of
postings, 185.028 MiB of positions and another 4.80 MiB of norm payload.
Size and RSS savings do not establish a query-speed win. See the final
[same-run latency, memory and ranking tables](../../search-benchmark-current.md),
[format contract](../../compact-text-format.md), and
[review findings](../../search-performance-review.md).

The final reader shares the compact descriptors' borrowed L0 slice and reuses
one header/payload lookup during decoding and content checks. No payload codecs,
block boundaries, document order, copied-merge policy or scoring formula change
as part of that cleanup. The norm-gather loop prototype is rejected and absent
from the final source. Byte norms intentionally change score/ranking precision;
compact directories with exact norms preserve every result bit.

## Archive

results.zip (local archive `results.zip`) is content-deduplicated. [manifest.json](manifest.json)
maps logical files to stored `archive_path`, bytes and SHA-256; it also records
the complete archive's size/hash. Identical logical files may resolve to one
physical ZIP member. Every member and the completed archive are verified.
Binaries, raw perf.data, indexes and raw document-order arrays are excluded;
binary/file/order hashes, source snapshots and text profiles are retained.

Key logical directories under `compact-text/`:

- Top-level `source.tar.gz` is the first implementation used to build x86
  indexes; `source-query-v2.tar.gz` built the ARM indexes. `source-final.tar.gz`
  is the first fully measured reader (v3), **not** the final selected source.
- `reader-v4/source.tar.gz` is the final 335-file source overlay, SHA-256
  `fa9e4bf6c205058c84f35717015c4527e1aa3d19196e11b0dd18a79c52ff2260`.
  Its manifest matches the main workspace and cloud build. The final overlay
  changes 22 files from the preserved September 16 baseline; the last reader
  cleanup changes only two files from v3.
- `arm-latency/` and `compact-text-evidence/` hold the first implementation's
  paired runs, audits and memory snapshots. `profiles-v3/` and
  `compact-text-profiles/` hold its CPU-clock text profiles. Hardware counters
  remain unsupported on this VM. Profiles are diagnostics, not latency trials.
- `reader-v4/` and `compact-text-reader-v4-evidence/` hold final ARM/x86 paired
  results, raw samples, exact oracles, checks and the final x86 residency audit.
- `arm-gather-latency/`, `gather-prototype.patch` and associated build logs record
  the rejected norm-code gather experiment. It was not promoted to x86 or main.
- `buffer-prototype/` records an unpromoted two-line decoder-buffer reuse
  experiment. Its ARM changes were small overall and metadata-only COUNT moved
  too; the measurements do not isolate a decoder speedup. It passes 220 posting
  tests and all three ARM oracles but is absent from the selected source.
- The helper `text_layout_audit.rs` fingerprints term keys, block geometry,
  codec/width/count fields and every encoded payload, excluding norms/bounds
  and physical addressing metadata. `traversal_score_oracle.rs` compares the
  optimized paths with a collector that disables top-k pruning.
- Top-level `harness/` records the initial broker discovery timeout, passing serial rerun,
  the v3 full check and the final v4 full check. Initial helper quoting and
  stale test-field compile errors and their corrections are also retained.

## Protocol and invariants

Full corpus: 5,032,104 Wikipedia documents; 962 official queries plus 714
standalone terms. Cascade Lake, Rust 1.98.1 / LLVM 22.1.8, native CPU flags,
release LTO, four build jobs and one query CPU (2). ARM uses 100,000 documents
on an Apple M4 with the same compiler/flags; shared Mac load limits small-effect
claims. Fresh release compilation is recorded for each reader.

Seven rotated complete passes follow at least ten seconds of warmup per
engine/command. The final x86 run compares the preserved baseline, the final
reader on old bytes, compact/exact, compact/byte norms, Tantivy and the first
byte-norm reader in one run. The last control isolates the reader cleanup;
results are not multiplied across runs. Proof-cache budget is 262,144 bytes;
term-cache limits are 8,192 blocks / 4,194,304 bytes. Rounded payloads and ratio/L1
bounds are held fixed; impact envelopes and RGB are disabled. Builds, full-byte
scans, profiles and memory audits do not overlap latency on the same host.

Separate fresh-process memory runs execute three passes per command and retain
`smaps`, rollups, mapping attribution and process status. RSS, anonymous memory
and index file bytes are reported separately. Final source/index hashes are
verified after the experiments. Indexing logs and peak resource measurements
are retained, but sequential indexing runs are not a paired ingestion benchmark.

All three layouts have identical document order, encoded payloads and block
geometry on each architecture. Full-corpus payload/geometry hash:
`bc82eebf366f209f4485dd90932d2c304c1cd4935b13bb027a5c515acd4067d3`.
The compact/exact oracle matches the preserved legacy oracle. Each reader and
layout passes all 1,676 queries against exhaustive results at k=10/100/1000,
including raw score bits, ordering and exact counts. Final v4 oracle outputs
match v3 for every layout. Ranking overlap between byte and exact norms is
reported separately; it is not a judged relevance test.

## Checks and limits

Final `python3 scripts/check_search.py full`: 1,816 passed, 25 ignored, plus
four real-server tests passed. Formatting, Clippy, native no-sync, portable core
and documentation builds pass. Portable compilation retains the existing
`set_document_units` warning. Final x86 core: 1,599 passed, 16 ignored; the
compact/norm integration test passes. The final focused posting suite passes
220 tests. WASM is not rebuilt, following the user's instruction. Cold and
concurrent performance and judged relevance remain unmeasured.

The owned VM is stopped after verified evidence download. Changes are uncommitted.
