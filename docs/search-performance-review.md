# Core/server review — 2026-09-05

Review base: `dc09bb3594424910f29f3854deaf1ac58c7fc0f1`.
This is a review of core/server entry points, merge representations, metadata
residency, and related broker behavior, followed by a focused alignment pass.
It is not a claim that every algorithm in the search stack has been audited.

## Implemented findings

| Priority    | Finding and trigger                                                                                                                                                       | Change and evidence                                                                                                                                                                                                                      |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1          | `GetTextStats` bypassed search shape validation and admission; deeply nested queries and concurrent statistics calls could reach expensive work outside the search limits | Reuse the query-shape walker and shared search permit before index open. Two RPC-level regressions first returned `NotFound` where `InvalidArgument`/`ResourceExhausted` were required; now they enforce the boundary and permit release |
| P1          | `HERMES_PIN_METADATA_BUDGET_MB=u64::MAX` overflowed multiplication, panicking in debug or wrapping in release; malformed values silently disabled pinning                 | Checked conversion and actionable warnings; separate-process regression covers unset, zero, valid, malformed, negative, and overflowing values without mutating global test environment                                                  |
| P2          | A merge synthesized absent fast columns with document-sized vectors, ran codec estimation, serialized, and copied the result back out                                     | Emit the existing constant/empty codecs directly; one bounded missing payload per column and no second block-directory/payload-placeholder arrays                                                                                        |
| Maintenance | Search orchestration, shape policy, hydration accounting, and hundreds of tests lived in one 1,862-line file                                                              | Extract `search_service/validation.rs`, `response.rs`, and RPC/budget tests; preserve `SearchLimits` and `QueryShapeLimits` re-exports. The service retains the request orchestration                                                    |

The shared [contract](search-system-contract.md), root `AGENTS.md`, executable
`scripts/check_search.py`, harness self-tests, and CI ownership check now connect
the design requirements to a repeatable workflow. Existing documentation was
corrected where it equated `native` with `sync` or described advisory cold I/O
as a general cache-bypass guarantee.

Statistics validation preserves the broker's flattened text-term container:
the aggregate node/clause budgets still apply, while the per-Boolean scoring
fanout is specific to search. An additional regression first rejected 129
flattened terms, then passed after this distinction; an oversized aggregate
remains rejected. This avoids turning the new admission checks into a broker
compatibility regression.

The missing-column optimization preserves the format: 9 bytes for a nonempty
single-value absent column, 23 for a multi-value absent column. Tests compare
the complete encoded block against the existing builder for numeric/text,
single/multi-value, and several document counts. A complete three-source merge
reopens with missing sources on both sides of real numeric/text values. Another
test emits a billion-document absent block without a document-sized allocation.
Pinning coverage now compares exact doc IDs, score bits, and scored count before
and after moving metadata into heap-backed storage.

## Deployed indexing follow-up: position scratch and commit recovery

A userspace CPU-clock sample of release 1.8.121 on the social shard captured
4,850 indexing-worker samples, of which at least 4,676 (96.4%) were in the loop
clearing every retained position vector before each text field/chunk. The
scratch map accumulated the segment's vocabulary, including terms absent from
the current field; even unpositioned fields paid for that scan.

`SegmentBuilder` now records distinct terms with nonempty position scratch and
clears only those on the next field/chunk. Cleanup is O(previous field's unique
position terms), not O(segment vocabulary). Existing per-term vector capacities
are reused. Additional scratch is one `Spur` per distinct positioned term in
the largest field/chunk (four bytes per entry, plus Vec capacity slack); the
existing segment-wide map still retains its allocations. Regression coverage
checks growing vocabulary, repeated terms, empty/unpositioned fields, both
tokenizer paths, all three position modes, and isolation between chunks/docs.

The focused fixture is `hermes-core/examples/indexing_scratch_benchmark.rs`:
1,000 eight-chunk documents after preloading 1K/10K/100K terms, excluding flush,
ANN and merge work. Exploratory before/after runs used Rust 1.98.0 and the same
release flags, but other CPU-heavy work ran on the shared Mac; these are not
controlled throughput estimates or a production speedup claim. The fixture's
large seed field also retains a large term-frequency table, so it includes
more than the position-cleanup cost. Repeat on an idle machine and measure
end-to-end ingest/maintenance separately before asserting sustained capacity.

The same production review found all three document shards paused after a
300-second flush timeout, despite their builds eventually finishing. Retrying
`Commit` recovered the retained generation (1,744,007 committed documents
across the three shards). Core now exposes `CommitFlushTimeout` as a distinct
error. The server owns an accepted commit through flush, publication, reader
reload and timeout retries even if its client disconnects. Shutdown waits for
that writer guard before closing segment-build admission. Build/publication
errors are not retried in a loop. See [the lifecycle contract](segment-lifecycle.md)
for recovery and forced-termination limits. This does not alter sparse reorder
policy, ANN construction, scoring, or persisted formats.

## Rust and low-level continuation

The [Rust hot-path review](rust-hot-path-review.md) extends this pass with
optimized assembly, closure/virtual-call probes, decoder benchmarks, measured
native layouts, and ranked follow-up experiments. It implements batch range
materialization and safe byte-range validation that allows byte-aligned decoders
to vectorize. Range comparison semantics are shared by scorer, probes and scans;
no persisted format, ranking arithmetic, or architecture default changes.

On the same M4, 256-value 16/32/64-bit decode batches measured approximately
8.8×/3.8×/4.7× faster, with 696 additional bytes of decoder machine code and no
new heap scratch. Three same-binary range comparisons showed about 6.5–8.1×
lower materialization time; a fourth had an unstable control. The linked review
records the shared-machine noise, lost initial distributions, fresh baselines,
assembly, native/async/WASM validation, and the distinction between measured
fixes and proposed work on dispatch, scratch, metadata layout and header walks.

The harness now accepts `--bench rust_hot_paths` and a Criterion `--filter`.
New distributions default to `.context/search-harness/criterion` so compiler
output cleanup does not remove them. Earlier evidence paths below describe the
original runs, which used Cargo's target directory.

## Benchmark evidence

Fixture: `hermes-core/benches/segment_merge.rs`; two RAM segments, each with
4,096 or 65,536 documents and one multi-value numeric fast field. The control
copies both source columns. The missing case simulates one older source without
the optional column. Source building/opening is outside timing. Each iteration
merges all segment components into the same unpublished output, preventing
unbounded retained RAM outputs. No ANN training or reordering is involved.

Host: Apple M4, 10 logical CPUs, aarch64 macOS 15.6.1; Rust 1.98.0 / LLVM 22.1.8;
Cargo release benchmark profile, default `sync` features, no custom RUSTFLAGS.
Criterion: 20 samples, 3-second warmup and approximately 5-second measurement
per case. A second baseline run replaced an initially noisy control measurement;
the 65,536-document missing case was stable at roughly 315–317 microseconds.
This is a shared development machine, so small control differences need caution.

| Case (documents per source) | Before, µs | After, µs | Interpretation                   |
| --------------------------- | ---------: | --------: | -------------------------------- |
| Copy both columns (4,096)   |      41.83 |     41.94 | No statistically detected change |
| One missing column (4,096)  |      47.20 |     31.48 | About 1.5× faster                |
| Copy both columns (65,536)  |      58.02 |     58.08 | No statistically detected change |
| One missing column (65,536) |     314.78 |     51.24 | About 6.1× faster                |

These are Criterion's reported central timing estimates. For the large missing
case, the before interval was 313.08–317.51 µs and after was 49.93–52.21 µs;
Criterion's separate comparison estimator reported a 83.90–84.98% reduction.
Both controls reported no significant change (`p=0.16` and `p=0.42`). The result
supports removing the document-sized synthesis cost; it is not a claim of a
6× speedup for ordinary merges or production search latency.

Commands used:

```sh
python3 scripts/check_search.py bench --save-baseline review-before
# After the implementation and correctness checks:
python3 scripts/check_search.py bench --baseline review-before
```

Raw commands, environment, dirty-source fingerprint, and timing output:
`.context/search-harness/20260904T204806.155345Z-bench/` (before) and
`.context/search-harness/20260904T210348.998886Z-bench/` (after). Criterion's
distributions are under `target/criterion/segment_merge/`. The fixture was added
before the baseline; the core implementation at baseline was still the review
base. Compare using the same added fixture when reproducing against that base.

Memory improvement follows directly from the removed allocation: the old
multi-value path retained at least `4*(N+1)` bytes of offsets and another
`8*(N+1)` bytes while encoding them, plus capacity slack/codec scratch. At
65,536 missing documents, those two arrays alone occupied about 768 KiB. The
replacement encodes 23 bytes using bounded temporary storage. This is a code-
derived allocation bound, not an RSS measurement. Existing source block views,
output bytes, store block metadata, and reader validation still have their own
costs; the entire merge is not constant-space.

## Remaining findings and proposed experiments

These items are not silently treated as compliant. Their changes need the
additional behavior/format or production-workload validation listed here.

1. **P1 — Fusion drops cross-shard statistics and the text deadline.** In
   [search_service.rs](../hermes-server/src/search_service.rs), the fusion arm
   calls `search_fused_with_count`; only the ordinary-query arm constructs and
   passes `stats_override` and `deadline`. The broker deliberately extracts text
   leaves from fusion in `partition::text_stats_query` and sends merged stats.
   Unequal shard term distributions can therefore change lexical contributions
   inside fused ranking, and a fusion request's text budget is ineffective.
   Add a budget/statistics-aware fused entry point shared by native/async paths,
   aggregate truncation, and test two shards with unequal term distributions.
   Until supported, explicitly rejecting unsupported options is an alternative
   that requires a deliberate API compatibility decision. This is a code-path
   finding; no distributed fusion fix or performance claim is made here.

2. **P1 — Standalone document hydration remains outside search admission and
   response accounting.** `get_document` loads all stored fields and clones them
   into protobuf values without the permit/budget used by `search`. Transport
   byte limits are checked too late to bound this transient heap amplification.
   Define a document-fetch budget and admission policy, then reuse response
   accounting with endpoint-level tests for many concurrent large documents.
   Measure peak RSS and rejection/retry behavior. Applying the search hydration
   limit unchanged could reject documents that the standalone API currently
   serves, so that behavior needs an explicit decision.

3. **P2 — First text fast-field access builds an allocation-heavy global
   dictionary.** `FastFieldReader::build_text_state` in
   [fast_field/mod.rs](../hermes-core/src/structures/fast_field/mod.rs) clones
   source strings into a `BTreeMap`, builds ordinal maps with another lookup
   pass, then serializes the global dictionary. Its “k-way/O(total_entries)”
   comment overstated the implementation and is now corrected. Prototype a heap of borrowed
   dictionary cursors, streaming unique terms and ordinal remaps directly into
   contiguous output. Benchmark first text access after multi-segment merges
   (1/4/16/64 blocks, high and low overlap), retained/peak heap, and warm lookups.
   Require byte-equivalent sorted dictionaries and identical multi-value
   ordinals, including empty/missing blocks. This is a stronger general workload
   candidate than optimizing rare absent columns alone.

4. **P2 — Extend bounded cold range copying to stores/fast fields if storage
   profiling warrants it.** BMP/ANN paths already use kernel-assisted range
   copying. `StoreMerger::append_store` reads each compressed block and writes
   its bytes; fast-field merge writes mapped data/dictionary slices directly.
   Prototype bounded local-file copies with short-copy/error handling and
   cancellation checks. Measure Linux page faults, read/write bytes, merge
   throughput and concurrent query p99 on data larger than RAM. The RAM benchmark
   here cannot justify such a change, and `copy_file_range` benefit depends on
   filesystem support. Dictionary-compressed stores still require recompression;
   removing it needs a versioned per-block dictionary-reference design.

5. **P2 — Audit physical-page ownership and aggregate pin accounting.** Pin
   budgets count logical section bytes and apply per segment/generation. Actual
   `mlock` rounds to pages, and the independent `HeapPinGuard` owners can cover
   allocations sharing a page. Linux memory locks do not stack: an overlapping
   `munlock` can release residency required by another live owner. The OS behavior
   is specified in [mlock(2)](https://man7.org/linux/man-pages/man2/mlock.2.html);
   the frequency of shared-page overlap here remains unmeasured. Add a Linux
   generation-overlap stress fixture, compare reported bytes to `VmLck`, and
   evaluate shared page-range ownership or dedicated page-aligned metadata
   arenas before promising a physical process-wide budget. Include old readers
   held across merges in residency sizing.

## Release gates

Validation completed on this host:

- `python3 scripts/check_search.py full`: all eight steps passed. This includes
  formatting, Clippy for core/server/broker/tool, 1,406 passing Rust tests,
  native-without-sync and minimal-core compile checks, API docs with warnings
  denied, and two additional broker tests against real server subprocesses.
- After the final statistics-container compatibility change: all 54 server
  tests, server Clippy, a fresh server build, and both real-server broker tests
  passed again. The new test adds one distinct regression to the full-run count.
- Five harness self-tests passed, including dependency aliases/target-specific
  boundaries, missing documentation, subprocess failure, and timeout cleanup.
  Python Ruff checks/formatting and `git diff --check` passed.
- WASM/browser, Linux mlock/cold-I/O, GPU, and x86 performance were not run in
  this pass. The new encoding helper is native-gated; public protocol formats
  and generated clients did not change.

Local raw evidence is retained in
`.context/search-harness/20260904T205148.716251Z-full/`, with the final server
checks in `.context/server-final-{clippy,test,build,e2e}.log`. The intentionally
failing regressions are in `.context/server-regression-before.log`,
`.context/pin-before.log`, and `.context/stats-flatten-before.log`.

Keep defaults unchanged until a representative production corpus and concurrent
ingest/merge workload show acceptable p95/p99, memory, recall, and throughput.
Run Linux cold-I/O/mlock measurements and x86 AVX2 comparison separately; this
Mac/RAM run cannot establish those results. The focused harness does not replace
the workspace's GPU, client generation, or WASM integration jobs.
