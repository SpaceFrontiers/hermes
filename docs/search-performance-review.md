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

## Deployed BM25 latency review (1.8.122, 2026-09-05)

This follow-up is diagnostic only: no production settings, index data or search
implementation were changed. All four servers and the broker were healthy on
1.8.122. The document index contained 10,339,428 documents and 177,707,964 text
chunks; `machine` and `learning` occurred in 2,247,060 and 3,371,715 chunks.
Indexing/maintenance and other searches remained active. Segment count changed
from 20 to 15 during the observation window (approximately 11:50–12:00 UTC).

Sequential gRPC probes requested top 40, loaded only `id`, used the same query
text for BM25 and server-tokenized sparse search, and had a 30-second RPC limit.
These are small live samples, not controlled benchmarks, recall comparisons or
production percentile estimates. Times below are median `timings.search_us`
over three repetitions: the broker reports the maximum backend phase time,
excluding its statistics round trip and the client's SSH/network overhead.

| Query                | Plain content BM25 | BM25 + separate phrase bonus | Bounded proximity rescoring | Sparse |
| -------------------- | -----------------: | ---------------------------: | --------------------------: | -----: |
| `history of germany` |              85 ms |                       186 ms |                       68 ms |  53 ms |
| `machine learning`   |             140 ms |                       493 ms |                      116 ms |  21 ms |

For `machine learning`, plain BM25 ranged from 134–233 ms, phrase-bonus search
from 416–858 ms, and sparse from 16–118 ms. The phrase-bonus request exposed
217,093 intermediate hits versus 548 for plain BM25; these are executor results
seen by collection, **not** posting-visit counters. ID loading took 5–19 ms in
the plain case and 7–12 ms with the phrase bonus. A nine-term query did not show
a consistent BM25/sparse disadvantage: two plain-BM25 samples were 169/260 ms
versus sparse 173/281 ms. No equivalent dense query embedding was supplied, so
there is no matched dense latency or recall claim from this review.

Observed request shapes included `SHOULD(Match(content), Boost(Phrase(content)))`.
Plain text already selects the windowed block-max MaxScore executor; enabling
WAND/MaxScore from scratch is not the missing optimization. Production logs
also contain slow fusion requests, but currently do not identify each fused
subquery's cost, so their full latency cannot be attributed to BM25 alone.

### Confirmed issues and recommended order

1. **Fix the text pruning-factor contract.** The converter accepts
   `MatchQuery.heap_factor > 1`, and `BooleanQuery::with_text_heap_factor`
   preserves that value. `MaxScoreExecutor::new` then clamps it to `[0.01, 1]`:
   every supported approximate text value becomes exact `1`. On one document
   shard, factors 1 and 100 returned identical ordered IDs, scores and `seen`
   counts in both paired runs; scoring was 104–107 ms in all four requests.
   Normalize the public text convention to the executor's sparse convention
   explicitly and add an RPC-to-executor regression. Keep approximation opt-in
   and measure recall before selecting a default.

2. **Make budgets effective through scorer construction.** With a 1 ms text
   budget on the same shard, `machine learning` plain BM25 reported truncation
   after 10 ms of search, but the phrase-only path spent 385 ms and reported
   `truncated=false`. A single-term `machine` query spent 67 ms and also reported
   no truncation. The phrase-bonus request spent 282 ms before reporting
   truncation: its eager phrase construction had already run. Thread the budget
   through phrase verification and single-term chunked execution; do not present
   the RPC deadline as cancellation of already-started blocking CPU work.
   The existing fusion budget/statistics finding below also applies.

3. **Optimize phrase verification without changing match semantics.**
   `build_chunked_phrase_scorer` drains all matching chunks into a vector and
   folds all matching documents, even when the phrase only supplies an optional
   bonus. `PositionStream::read_into` re-decodes its position block on each
   access; its scratch retains capacity, not a keyed decoded-block cache.
   Phrase matching also restarts linear position scans for each starting
   position. Prototype per-term bounded position cursors/caches and monotone
   positional intersection, then a lazy, top-k-aware composition that preserves
   mandatory phrase constraints. The existing bounded `proximity_weight` stage
   is an optional ranking alternative, **not** an equivalent replacement: on
   `history of germany` its top 40 overlapped the separate-phrase version in
   only 25 results (35 for `machine learning`). Benchmark rank-safe kernel
   changes separately from any candidate-limited ranking change.

4. **Tighten existing chunk bounds before changing codecs or scoring.**
   `LengthLookup::length` writes raw chunk lengths into block minima, while
   scoring uses `ChunkMap::bm25_length`, floored at the nominal chunk length.
   List/block/group bounds use the raw minimum, so they can be conservatively
   loose. Applying the same floor when computing query-time upper bounds is a
   format-preserving, rank-safe candidate, subject to exact-top-k oracle tests.
   Cache bounds per active block/group rather than recomputing the same BM25
   divisions for successive windows; Lucene's
   [MaxScoreCache](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/MaxScoreCache.html)
   uses this separation. Cross-segment pruning for chunked results additionally
   needs a floor backed by **distinct documents**, not a raw top-chunk heap.
   None of these proposed kernel speedups has been measured in this review.

The sampled document shard had zero CPU quota throttling, no swap or OOM events,
and zero recent memory-pressure averages, but some CPU contention. These checks
do not rule out page-cache misses or quantify per-query peak scratch. No
instruction-level CPU profile or controlled cold-cache/cross-architecture
benchmark was obtained. Preserve scoring, formats and production defaults until
the proposed changes pass the harness and a representative quality/performance
comparison. Local raw probes and the reproducible sequential client are retained
in `.context/bm25-remote-probes-20260905.json` and
`.context/bm25_remote_probe.py`.

Validation for this diagnostic/documentation follow-up:
`python3 scripts/check_search.py check` passed all four steps on Rust 1.98.1,
including 1,284 core tests, 56 server tests, broker/tool tests, Clippy, formatting
and the native-without-sync boundary. Evidence is in
`.context/search-harness/20260905T120227.417351Z-check/`. The additional `full`
harness, WASM and performance benchmarks were not rerun: no search implementation
was changed, and these existing tests do not establish coverage of the newly
identified bugs.

## BM25 execution fixes following the deployed review

Implemented against `5cecf189` with Rust 1.98.1. The preceding section records
the observation phase, before these fixes; publication/deployment are separate
release operations.

- Text and sparse now share the executor's reciprocal convention: factors below
  1 enable extra pruning (`threshold / factor`). RPC zero/unset or 1 selects
  exact search; negative, non-finite and >1 RPC values fail validation, without
  legacy translation. Single-token conversion no longer bypasses explicitly
  requested text pruning, and nested query flattening preserves tuning. The
  shared executor's existing 0.01 factor floor caps the effective multiplier at 100. Exact defaults and sparse tuning are unchanged.
- Nested scorers keep the deadline/truncation state while discarding the outer
  score floor. Term/phrase construction, phrase intersection, phrase bitsets,
  text executor entry and generic collection check it. Cancelled bitsets never
  escape as complete negative filters. An expired request can return no hits;
  cancelled chunked phrase construction does not start its all-hit sort/fold.
  Legacy non-phrase materializers have boundary checks, not preemption inside
  their work; outstanding I/O is still not interrupted.
- Phrase and proximity readers retain one decoded position block per term.
  Phrase slop/offset matching uses monotone position cursors rather than
  restarting each list scan for each starting position. Frequency, repeated
  terms, offset gaps, independent slop intervals and chunk boundaries stay intact.
- Text list/block/group score bounds apply the scoring chunk-length floor.
  Two one-entry bound caches per text cursor remove repeated BM25 divisions;
  no encoder, merge writer, persistent format or score formula changed.

The failing regressions reproduced equal exact/approximate rankings, ignored
expired construction budgets, and raw-length bounds before their respective
fixes. Coverage also includes converter-to-executor behavior for one/multiple
terms on plain/chunked fields, opposite text/sparse factor conventions, exact
top-k oracles, nested floor isolation, cancellation after an initial phrase
match, generous-budget score/ordinal/negative-filter equivalence, position
frequency oracles, and cached reads across copied short blocks and backward
seeks. Reading preserves the original encoded position bytes.

### Same-fixture local measurements

`hermes-core/examples/bm25_execution_benchmark.rs`: 20,000 RAM documents,
80,000 positioned chunks, top 40 with ordinals, one indexing/search thread,
current-thread async entry, default release flags on the same Apple Silicon
Mac, Rust 1.98.1 (`48a229cea`). Each run has a warmup plus ten timed searches.
The corpus deliberately repeats terms 40–60 times in most chunks: it stresses
the positional kernel and is not representative of every production query.

| Query               | Before median | After medians, three runs |
| ------------------- | ------------: | ------------------------: |
| Plain BM25          |       1.43 ms |              1.30–1.40 ms |
| Phrase              |      36.48 ms |            11.53–11.66 ms |
| BM25 + phrase bonus |      38.53 ms |            13.57–13.96 ms |

The before/after ordered document IDs, score bits, ordinal score bits and seen
counts are identical in all runs. The final repeats ran after this task's
compilers/tests stopped; the shared Mac was not CPU-isolated. These are local
synthetic improvements (about 3.1x phrase and 2.8x phrase-bonus), not production
latency/recall estimates or evidence to change ranking defaults. Plain BM25 is
effectively unchanged on this tied-score fixture; no standalone speedup is
claimed for tighter bounds here.

Process peak RSS from `/usr/bin/time -l`, including fixture construction, was
109.5 MB before and 108.9–113.4 MB after. This is not isolated query scratch or
evidence of reduced memory. Each position cache holds at most 128 u32 values
(512 bytes) per term plus small metadata; the score-bound caches add 16 bytes
per text cursor. These measurements precede lazy folding (below). Per-document
position buffers remain. Fusion deadline/statistics propagation (see remaining
findings), and production/cold-cache/recall evaluation are still follow-ups,
not completed by this patch.

Raw local evidence: `.context/bm25-before*.log`, `.context/bm25-final-{1,2,3}*.log`.
Full validation passed in `.context/search-harness/20260905T124157.496034Z-full/`,
including both real-server broker tests; the final collector-boundary adjustments
passed `check` in `20260905T124513.498918Z-check/` (1,295 core tests passed,
12 ignored, plus server/broker/tool/integration tests). Final WASM build and all
four JS tests passed after `npm ci`. GPU/full-workspace and production performance
checks were not run.

### Incremental lazy-folding validation

Verified doc-ordered chunk maps now yield one folded document at a time, with
one matching-chunk lookahead. Reordered maps retain stable eager aggregation.
Both paths match the old fold's document IDs, score bits and ordinal encounter
order, including missing chunks and zero scores. A counted-cursor test proves
construction consumes only one document, and a seek skips directly to a late
document; the index-level test verifies matches beyond the requested top-k.
Expired iteration clears the current score/ordinals and marks truncation.

Three paired runs of the same 20K-document fixture above compared a preserved
pre-lazy binary with the final binary, both Rust 1.98.1 release, same Apple M4.
All this task's compilers/tests had stopped, but the shared Mac still had other
application/test activity, so these remain exploratory warm-cache measurements.

| Query               | Before lazy folding, medians | After lazy folding, medians |
| ------------------- | ---------------------------: | --------------------------: |
| Plain BM25          |                 1.29–1.37 ms |                1.33–1.42 ms |
| Phrase              |               11.73–12.15 ms |              11.11–11.12 ms |
| BM25 + phrase bonus |               13.77–13.87 ms |              12.87–12.90 ms |

Ordered IDs, exact score bits, ordinal score bits and seen counts match in all
three pairs. Lazy folding adds roughly 5–9% phrase and 6–7% phrase-bonus savings
on this fixture, not another multi-fold kernel speedup. Plain BM25 is not helped
by phrase folding. Process peak RSS (including ingestion) was 107.2–113.6 MB
before and 112.5–116.0 MB after: no process-memory reduction is established.
The structural query-scratch reduction is from all matching chunks/documents
to one document's ordinals; the opener adds a four-byte-per-chunk sequential
order check and one retained boolean. Reordered segments do not get this folding
benefit. No persisted bytes or merge writer changed.

Final `full` harness passed all eight steps in
`.context/search-harness/20260905T130936.230637Z-full/`: 1,299 core tests passed,
12 ignored, 58 server tests, broker/tool/integration tests, both real-server
broker tests, Clippy, portable/native boundaries and API docs. WASM release plus
four JS tests and regenerated TypeScript plus four client tests passed. The
documentation check passed via `uv run scripts/check_docs.py` (plain Python
lacked its declared Markdown dependency). npm reported two existing WASM dev
dependency audit findings; dependency upgrades are outside this patch.
Raw paired evidence: `.context/bm25-lazy-{before,after}-{1,2,3}*.log`.

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

## Candidate scoring and coordinator review (2026-09-05)

The post-implementation review traced the named query contract through native,
async and WASM execution, immutable ordinal lookup publication, branch nomination,
score backfill, document combiners, shard export, broker selection and clients.
Core owns the only linear scorer and fusion algorithm; RPC adapters validate and
translate. Existing extraction and chunk limits are unchanged.

Findings resolved in this change:

- A multi-field branch could cast two RRF votes for one logical passage, changing
  the winner. The failing `one_branch_cannot_vote_twice_for_the_same_passage_across_fields`
  regression now passes; core deduplicates by logical passage before assigning ranks.
- Averaging or summing exported top passages cannot reproduce a document's full
  reduction. Shards now export all scored rows for AVG/SUM and enough top rows
  for MAX/weighted-top-k. Both levels execute the same core formula and verify
  score agreement before the broker selects the global page.
- The broker rejects missing shard responses/statistics, incompatible response
  versions, duplicate document/branch identities, unexpected scopes, nonfinite
  features and incomplete combiner inputs. CPU work retains admission permits
  after client cancellation. Concurrent decode bytes and feature matrices are bounded.
- Plain-text fields missing required length metadata now advertise that they
  are unprepared instead of claiming readiness and failing only during backfill.
- Integration with the independent text-pruning change preserves lazy ordered
  phrase iteration and cancellation, while point backfill and ordinary phrase
  retrieval share positional verification, global BM25 statistics and field parameters.

The full harness passed all eight stages after integration with 1.8.123, including
1,309 core unit tests, 62 server tests, 49 broker unit tests, 13 mock-broker
integration tests and both real-server broker tests. Evidence is in
`.context/search-harness/20260905T133245.093943Z-full/`. Python client round trips
(8 tests) and TypeScript build/wire tests (6 tests) passed before the upstream
merge; generated clients were refreshed against the combined protocol afterward.

These are correctness results, not production recall or latency measurements.
The L1 model remains opt-in. Training must measure held-out teacher candidate
recall and passage survival with a frozen corpus, along with latency, response
bytes and concurrent indexing load. Legacy nested fusion and fusion with the
vector reranker retain their existing shard execution and do not advertise
`global_rrf_v1`. Ordered old segments remain readable without a new sidecar;
legacy reordered fields explicitly require preparation before L1 use.

The Linux CI broker-only build exposed an x86 feature-boundary regression:
`pack_group_bmi2` was compiled when its native/WASM writer caller was absent.
Its cfg now matches the caller, retaining runtime BMI2 dispatch in writer builds.
The failing CI run is `33969669241`; this does not change encoded bytes or scoring.
