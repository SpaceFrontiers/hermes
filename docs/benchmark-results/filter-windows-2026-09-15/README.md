# Filtered collection windows — September 15, 2026

This continuation preserves bounded collection windows through the existing
`PredicatedScorer`. The selected implementation requires the driver's
`supports_filtered_windows` capability, defaulting to false. Eligible Boolean
composites opt in; leaf filtering retains scalar traversal. Predicate checks,
MUST score addition order, forward-only consumption and positions are preserved.
No cache, encoding, approximation default or alternate executor is introduced.

The first prototype forwarded all window capabilities. Its standalone-term
regressions motivated the conservative composite-only capability. Its sources
and results remain in the archive, explicitly separate from the final version.

On the full-corpus fixed-mask workload, official top-100 plus exact count is
28.6% faster and count 25.3% faster; all seven passes improve. ARM gains are
14.6–15.6% and 18.2–19.7%. Ordinary x86 top-10/top-1000 regress 1.1%, and masked
phrase count regresses 6.7%. Those tradeoffs remain recorded; the ordinary
ranked-search gap is still open. Process RSS is essentially unchanged.

See the [current comparison](../../search-benchmark-current.md) and
[performance review](../../search-performance-review.md) for measured results,
selection and remaining work.

## Evidence layout

- results.zip (local archive `results.zip`) contains source overlays, exact runner scripts,
  per-query samples, ordered ID/raw-score/count oracles, disassembly, index
  hashes, process RSS and failure/recovery logs.
- [manifest.json](manifest.json) records each logical file's byte length,
  SHA-256 and ZIP `archive_path`. Identical files share stored entries; restoring
  logical paths requires reading their `archive_path` and checking the hash.
- native-check.zip (local archive `native-check.zip`) and
  [native-check-manifest.json](native-check-manifest.json) retain the final
  native harness, portable compilation and source/binary identities. All 1,789
  tests pass (25 ignored), as do Clippy, native-without-sync and portable core.

`cloud/filter-windows-gated-evidence/` and `local/gated/` contain final source
and validation. Final ordinary ARM samples use the `local/gated-arm-` prefix.
The other two cloud directories and `local/masked-arm-*` record the rejected
unconditional-forwarding prototype. Executables and corpus payloads are omitted
from this compact archive; executable hashes and source overlays are retained.

## Reproduction

Restore the frozen source overlay on repository base
`ce2c96b945fccc4bac58ccc45bb4bc23b809773e`, verifying every source-manifest hash.
The prior cleanup snapshot supplies the control. Final changes are confined to
`query/docset.rs`, `query/traits.rs`, `query/boolean.rs` and regression tests.
`local/gated/final-source.tar.gz` pins the complete final source overlay.

The cloud scripts restore source into isolated worktrees, build with Rust
1.98.1, native CPU flags and release LTO, verify immutable indexes, run tests
and exact oracles, and then time complete rotated workload passes. Adapt their
workspace paths and create-only output directories before rerunning. Restore
the corpus/index fixtures separately from the preserved benchmark workspace;
they are not included here. The full-corpus host is the preserved Cascade Lake
VM with 5,032,104 documents and query/driver affinity to CPU 2.

Ordinary runs use all 962 official queries and 714 supplemental terms, all four
commands, and the same frozen Tantivy binary. Cloud runs use seven rotated
complete passes after at least ten seconds of warmup per engine/command. ARM
runs use eleven rotated samples per query and fifteen seconds of warmup on
canonical and merged Apple M4 fixtures. Geometric means use per-query medians.
Correctness oracles and compiler work are outside timing.

The separate fixed-mask helper admits physical IDs where `doc_id % 8 != 0`.
It constructs the bitmap before timing, uses the public query/predicate APIs,
and disables the unfiltered dictionary-count shortcut. It measures exhaustive
COUNT and top-100 plus exact count only, with seven rotated complete passes and
ten seconds of warmup. Both versions must produce byte-identical ordered
top-1000 IDs, raw score bits and counts for all 1,676 queries before timing.
No persisted index bytes are changed. The real deletion integration test also
checks positions and native/async ranked collection across an empty batch.

The first SSH run lacked Rust on PATH and stopped before building; loading the
Rust toolchain environment resolved it. An intermediate native check was
intentionally interrupted when the leaf regression prompted the capability
refinement. A first refined cloud build read a stale upload snapshot and failed
compilation; rebuilding the bundle directly from the validated source and checking
its input hashes resolved it. Intermittent cloud API connection failures required
download retries. Final validation is recorded separately from those attempts.

## Measurement limits

The fixed mask covers one periodic 12.5% exclusion pattern; clustered and
highly selective filters remain unmeasured. It isolates filtered collection;
it is not a deletion-publication,
masked ranked-pruning, cold-cache, concurrent-ingest or tail-latency benchmark.
RSS is the process high-water mark including mapped pages and heap. The wrapper
adds no scratch allocation; eligible score-window collection uses the existing
16 KiB collector allocation. Leaf filtering retains its previous scalar path.
The full RPC/browser suites and original upstream-runner confirmation of
post-OR changes were not rerun for this query-only change. WASM
was not rebuilt under the standing user instruction; portable core is checked.
