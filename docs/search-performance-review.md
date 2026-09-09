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

## Reordering continuation

The [reordering performance review](reordering-performance-review.md) traces
record/block BMP and chunked-text BP, removes repeated per-posting gain
arithmetic with an optional budgeted term cache, and records byte/permutation
and assembly evidence. It also records outstanding text memory/convergence
gaps, rank scratch, transpose sorting and scheduling experiments. The kernel
change preserves arithmetic order and existing formats; it does not resolve
the text-path findings or establish production performance. Synthetic BP
speedups are 1.11–2.33× on Apple M4 and 1.20–2.93× on an Intel Xeon 8581C
with default and AVX2 builds. Before/after permutations and sparse/query bytes
match within each tested platform; the original coarse permutation already
differs between macOS/ARM and Linux/x86. Both x86 builds passed 1,299 core
library tests, and peak process RSS stayed around 95–96 MiB on that fixture.

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
The first fix accidentally gated the read-side decoder; CI `33971140308` caught
that error. The corrected gate applies only to the writer. The broker's minimal
core now also passes an explicit `x86_64-apple-darwin` compile with Rust 1.98.1
and warnings denied (`.context/phrase-boost/l1-x86-broker-check.log` in the
Azeroth integration workspace).

The continuation caught lossy `GetIndexInfo` schema rendering: BMP sparse fields
were reported without `format: bmp`, dimensions, grid/block settings, mass
cropping or reordering. Re-parsing that report changed the apparent format to
MaxScore even though persisted production metadata was BMP. The behavior-named
round-trip regression fails before the fix and now preserves those settings.
This changes diagnostics and schema fingerprints, not persisted storage or
scoring. Production sparse format was verified against `metadata.json` on all
four shards; full-text continues to use its existing MaxScore execution.

Validation for the schema-reporting fix: all eight full-harness stages passed
with `RUST_TEST_THREADS=1` (evidence: `.context/search-harness/20260905T150826.682593Z-full/`).
Two earlier parallel runs timed out in different mock-broker discovery tests;
the recovery test passed in isolation and the complete serial suite passed.
A separate two-real-shard BMP fixture nominated only through dense search,
backfilled BM25/phrase/sparse/document features including negative and zero
values, and matched an independent full-union oracle for broker MAX/AVG/SUM
top-K with bounded raw passage export. Its script and results are retained in
the Azeroth integration workspace under `.context/l1-training/`.

## L1 completion and owned forward values (Quebec, 2026-09-05)

Reviewed and integrated `origin/handoff/l1-scoring-2026-09-05` at `2d21cb0e`.
Rebased onto `origin/main` at `1844d9af`, preserving its budgeted BP gain cache
and the earlier broker/text merge resolutions. This section supersedes the draft sidecar and complete-rescoring descriptions
above; historical evidence remains attributed to its original workspace.

Resolved findings:

- L1 recomputed organic scores and treated every missing value as an implicit
  zero contribution. It now preserves branch/document and branch/passage scores
  exactly, probes only missing cells, and supports optional `backfill` (default
  true). `missing_values` supplies learned raw defaults before transforms;
  raw exports retain absence. Actual zero/negative scores are never imputed.
  Core and broker share the same formula. The original coefficient contract was `linear_v2` /
  `feature_export_v2`, with candidate-scoring capability 2.
- Sparse MaxScore was unavailable for backfill. Bounded skip-index probes now
  establish presence and score selected documents/ordinals with the existing
  quantized block decoder. Full-text BM25/MaxScore and phrases already use the
  existing posting and position readers. Sparse probes/reads and lazy text
  materialization have request-wide admission budgets; no missing cells means
  no address/payload probes, including on legacy unprepared fields. Async BM25
  statistics now read only dictionary frequencies, fixing posting payload reads
  that bypassed candidate-scoring admission on lazy backends.
- Rejected `.lookup` sidecars introduced duplicate addressing and lifecycle
  state. BMP V20 stores quantized logical forward values inside `.sparse`;
  CHNK V3 adds only physical slots to the existing chunk metadata. Compatible
  merge copies payload and streams metadata remapping. Record and block BP copy
  forward bytes unchanged. Record BP graph construction and selected-record
  rewrite consume those forward values; block BP retains block-level inputs.
  V19 remains readable; all-legacy ordinary merges retain V19, mixed V19/V20
  merges reject, and explicit budgeted reorder migrates old values. No new
  publication, cleanup protocol or preparation RPC is introduced.
- Text merge could silently lose addressing by combining legacy unordered and
  prepared maps. It now rejects incompatible combinations before writing,
  retains the legacy version for pure legacy maps, and validates V3 ordering,
  permutations and section bounds. Explicit reorder upgrades even small legacy
  text segments with no BP plan, preserving original unsaturated token totals.
- Duplicate sparse dimensions were accepted by existing BMP ingestion but a
  new forward validator rejected them; the legacy point scorer also returned
  only one duplicate impact. Forward values preserve every retained impact,
  both point paths sum them, and accumulation overflow fails explicitly. Repeated
  query dimensions also retain all independently quantized contributions.
- The TypeScript wrapper discarded new L1 options despite correct generated
  bindings. A failing wrapper-level regression now verifies actual request
  forwarding of false and learned defaults. Python and TypeScript bindings
  preserve optional-boolean presence and absent raw feature keys. Both wrappers
  reject legacy linear responses rather than silently accepting ignored options.
- Nomination union construction cloned lists before checking retained budgets.
  The shared union builder accepts borrowed lists and checks each hit/ordinal
  before cloning. Original branch lists remain available for organic scores.

Behavior-named regressions reproduce V3 corruption acceptance, legacy duplicate
impact loss, repeated-query contribution loss, TypeScript option loss, legacy
response acceptance, premature lazy text payload reads, and invalid backfill/all-passage
policy acceptance before their fixes; the latter now fails before admission. Additional tests
cover quantized forward/inverted equality, real missing ordinals, byte identity
through copy merge and both BP modes, explicit V19 migration budgets, partial
writer failure, cancellation during payload output, and sparse MaxScore ordinals
spanning multiple blocks. Existing generation failure/cleanup tests exercise
these writers through their original lifecycle owner.

### Same-binary format/path comparison

Host: Apple M4, macOS 15.6.1 (24G90), aarch64, Rust 1.98.1 / LLVM 22.1.8,
default release flags. Fixed RAM fixture: 16,384 single-valued vectors, 4,096
dimensions, 64 retained entries/vector, 128-slot BMP blocks, 4-bit grids.
Legacy V19 is obtained by removing only the V20 forward section from the same
encoded fixture. This compares the retained old and new paths in one binary;
it is not a historical ingestion benchmark. Candidate scores agree exactly;
BP retains identical document, term and posting counts without budget truncation.
Existing BP rewrite tests separately compare all quantized values.

Run the ignored `measure_forward_candidate_scoring_and_bp_graph` core test in
release mode alone, with `--ignored --nocapture --test-threads=1`. It fixes BP
at four workers and a 128 MiB budget. Eleven samples measure 100 scoring calls
per sample (128 candidates, 64 query dimensions) and one BP graph construction.
No other CPU-heavy checks ran during this measurement.

| Operation             | V19 median (range)           | V20 median (range)        |
| --------------------- | ---------------------------- | ------------------------- |
| Candidate scoring     | 110.438 µs (110.162–115.661) | 12.427 µs (12.359–12.706) |
| BP graph construction | 2.759 ms (2.717–2.909)       | 2.180 ms (2.144–2.217)    |
| Encoded BMP bytes     | 6,944,624                    | 12,449,664                |

The forward section adds 5,505,040 bytes here (79.3% over V19). Its exact cost
is 5 bytes/retained posting + 16 bytes/vector + 16 bytes/field. Both forward
payload and directory remain evictable, with no new pinning allocation.
Graph CSR allocations remain 4,194,304 term bytes + 131,080 offset bytes on both
paths; existing physical maps and graph admission limits remain in force.
The entire measurement process peaked at 83,836,928 RSS bytes / 41,779,800 bytes
macOS memory footprint, including fixture construction and both source blobs.
This is not a per-operation or cold-mmap memory measurement. New-ingestion blob
construction took 121.580 ms, without a measured old-ingestion comparison.

Normal forward merge holds O(source count) plans, copies at most 4 MiB per
I/O call and streams fixed-size directory rows; it allocates no vector/posting
permutation. Explicit V19 migration separately admits 12 bytes/real vector of
permutation/offset scratch. New ingestion retains its original posting input
longer, adding only dimension cursors and 8-byte/vector offsets; it does not
materialize a second posting collection. Text merge patches doc IDs in 64 KiB
batches and streams slot remapping; explicit text rewrite admits its columns,
largest 4-byte/chunk permutation and retained BP plans before allocation.

These measurements support the selective scoring improvement on this fixture,
not a production latency, recall or universal BP speedup claim. Production
cold/warm p95/p99 with concurrent ingestion, reordered large corpora, Linux
kernel-copy/mlock behavior and x86 runtime measurements remain unmeasured.
No ranking, BP-granularity or SIMD defaults changed from these measurements.
Raw measurement output is `.context/l1-performance.log` in this workspace.

The follow-up [BMP forward-search research](bmp-forward-search.md) includes primary
literature, reproducible phase-two and whole-block kernel experiments, and explicit
safety conditions for selective completion, filters and threshold seeding. Current
Hermes measurements favor very small survivor sets, not replacing whole-block
inverted evaluation. Retrieval defaults remain unchanged.

Follow-up measurements include production per-block term masks and already-parsed
phase-two blocks. They supersede the initial unmasked-header microbenchmark:
one-survivor completion is about 2.6 times faster, while full 32-slot forward
scoring is 11.3–11.6 times slower on that fixture. A separate record-BP mmap
experiment finds a 42–44% one-survivor forward-kernel reduction from a dense query table plus
fused validation, but a 423,516-byte table costs about 3.1 µs to allocate/prepare.
Only ignored experiments use that kernel; no new production cache or search
policy is enabled. See the linked research for setup, memory and locality limits.

### Validation of the rebased integration

All eight full-harness stages passed with `RUST_TEST_THREADS=1` after the rebase:
`.context/search-harness/20260905T174703.766254Z-full/`. This includes 1,324 core
unit tests (17 manual experiments ignored), 64 server tests, 50 broker unit tests,
13 broker integration tests, core integration/doctests, and both real-server
broker tests. The earlier required `check` mode also passed; `full` repeats and
extends all of its stages. The real L1 fixture tests disabled backfill, learned
missing defaults, retained organic values and shared broker/shard inference.

Supplementary checks passed:

- Native without sync: nine candidate-scoring/model tests, including MaxScore
  and legacy addressing; `.context/l1-rebased-native-async.log`.
- WASM release build and four runtime tests in two files;
  `.context/l1-rebased-wasm-tests.log` records the runtime results.
- Python and TypeScript client unit tests: nine each;
  `.context/l1-rebased-python-tests.log`, `.context/l1-rebased-ts-tests.log`.
- x86_64-apple-darwin minimal-core cross-check with warnings denied;
  `.context/l1-x86-minimal.log`. This is not an x86 performance measurement.
- Final format/diagnostic comment cleanup: core all-target Clippy with warnings
  denied, `git diff --check`, and search ownership/document contracts passed.
- All four manual measurement runs (candidate/BP, phase two, whole query, mmap
  query-table/BP locality) passed their integer/shape oracles after the rebase.

No required validation remains unrun and no environment failure blocked these
checks. Production cold-corpus latency, concurrent load, Linux residency/copy
behavior, and x86 runtime performance remain outside the measured evidence.

### Retired selective forward-search experiment

The prototype integrated per-slot forward completion into actual BMP traversal,
including bounded reads, exact score/ordinal comparisons and real RPC tests.
It has now been removed from production search, together with its query option,
adapters, counters and format-only uniqueness certificate. The measurements
below are historical evidence for that decision, not a current search feature.
The prototype sources/patch are archived in `.context/retired-forward-search/`.

The [whole-query experiment](bmp-forward-search.md#whole-query-traversal-and-record-bp-experiment)
now includes real traversal before and after record BP, exact score/ordinal
comparisons, p50/p95/p99 and memory/fault evidence. At 4K dimensions/depth 10/cap 2,
warm median latency improved from 169.46 to 162.92 µs (3.9%), with essentially
unchanged tails. After BP, small survivor sets were rare; forcing whole-block
completion at depth 10 regressed median latency by 1.6–1.7 times. Reclamation hints
also removed the warm benefit. Forward completion is no longer integrated with search;
dense query tables and fused validation remain isolated kernel experiments.

Historical prototype validation (before removing its search integration):

- Required `check` passed at `.context/search-harness/20260905T182911.605688Z-check/`.
- All eight `full` stages passed at
  `.context/search-harness/20260905T183614.969305Z-full/`: 1,332 core unit tests
  (18 manual experiments ignored), 65 server tests, 50 broker unit tests, 13 broker
  integration tests, core integration/doctests and three real-server broker tests.
  The new real RPC test checks the opt-in and rejects an oversized setting.
- Final dispatch guard/lookup-counter cleanup passed seven focused native
  regressions, seven native-without-sync regressions and core all-target Clippy
  with metrics and warnings denied. Evidence is in `.context/l1-traversal-final-*`
  and `.context/l1-traversal-native-async.log`.
- Final WASM release build and five runtime tests passed; Python and TypeScript
  client suites passed ten tests each. Logs use `.context/l1-traversal-*`.
- The whole-query benchmark passed all comparison oracles, including actual
  forward scoring after record BP. Peak process RSS / footprint were
  96,354,304 / 30,851,720 bytes, including setup and both index layouts.

All required checks ran successfully. Controlled cold-disk, concurrent production
load and x86 runtime performance remain unmeasured. The workspace remains rebased
on `origin/main` at `1844d9af`; defaults are not changed from this synthetic ARM
fixture.

### Optional BMP forward storage and final search policy

`bmp_forward_index` is a per-field schema boolean, default true. SDL and persisted
JSON retain explicit false; server schema export preserves it. Disabled ingestion
skips forward construction and releases input postings before grid output.
Ordinary merge and both BP output modes omit the forward section when disabled.
Mixed V19/V20 sources can then copy their compatible inverted blocks into V19;
enabled mixed-version merges still reject an implicit migration. A field excluded
from explicit reorder remains a byte-identical copy, preserving that contract.

BMP retrieval has no forward-index integration or query switch. BP's per-vector
graph and rewrite reads and L1 missing-cell backfill always use stored forward
values when available, without a crossover heuristic. Block BP continues to use
its compact block graph and copies unchanged payloads. Corrupt forward values
fail their L1/record-BP consumers; ordinary search never reads that payload.
Ordered V19 maps still support targeted L1 posting probes. Unordered V19 requires
enabling forward storage and explicit reorder/rebuild, or disabled backfill with
organic scores and learned missing defaults. Full-text/MaxScore are unaffected.

Eight focused regressions pass, including full inverted-byte identity with
storage disabled, mixed-version copy merge, standalone and merge-time BP,
missing/zero/organic L1 scores, schema persistence and the search/L1/BP read
boundary. Logs: `.context/l1-storage-optional-tests.log`. The initial ingestion
test failed before the option was wired (`.context/l1-storage-optional-red.log`).
The space saving is exactly the omitted forward section: 5 bytes per retained
posting, 16 bytes per vector and a 16-byte trailer. No new latency claim is made.

Final validation:

- Required `check` passed:
  `.context/search-harness/20260905T191742.125172Z-check/`.
- All eight `full` stages passed:
  `.context/search-harness/20260905T192107.952734Z-full/`, including 1,333 core unit
  tests (17 manual experiments ignored), 64 server tests, 50 broker unit tests,
  13 broker integration tests, core integration/doctests and two real-server RPC
  tests. This includes optional-storage schema round-tripping in server SDL.
- All eight storage/boundary regressions also passed on native without sync:
  `.context/l1-storage-native-async.log`.
- WASM release build and all five runtime tests passed, including equal search
  results with storage enabled/disabled. Python and TypeScript regenerated
  bindings and their nine client tests each passed. Logs use
  `.context/l1-storage-{wasm,python,ts}-*`.
- Documentation links, search ownership contracts, formatting and whitespace
  checks passed. No required check remains unrun or environmentally blocked.

The earlier production-corpus, controlled cold-storage, concurrent-load and x86
runtime performance limitations still apply; this storage option makes no new
performance claim or change to BMP search behavior.

## Current BMP format and exclusion filters (2026-09-05 follow-up)

BMP now has one accepted/emitted envelope. Optional forward storage is an
explicit section with enabled or disabled state; disabling it no longer writes
an older format. The published enabled representation is unchanged. Disabled
fields add a 16-byte marker. Older readers/writers and version-based merge
selection are removed. The earlier V19/V20 comparisons above are historical
measurements from the migration release, not current compatibility guarantees.
Deploy against rebuilt indexes or after every live BMP blob passes the current
format audit. The operator chose a fresh rebuild instead of waiting for the
one-time forward materialization and BP pass over existing production segments.

The common fusion filter exposed an exclusion-only Boolean bug: no positive
clause produced an empty scorer and an unsupported bitmap, so small segments
returned nothing and large segments could reject the materialization fallback.
The Boolean owner now supplies a neutral document universe, subtracts excluded
matches and preserves entirely empty Boolean semantics. Absent indexed exclusion
terms produce complete empty bitmaps instead of an unsupported result. Native
bitmap materialization is O(segment words + exclusion postings), with the same
fusion bitmap budget; async Boolean scoring streams the complement. Regressions
cover pre-selection exclusion in RRF, L1 and feature export, cross-partition
exclusions, absent terms, tail-bit bounds and exhausted scorers.

Validation:

- Required `check` passed: `.context/search-harness/20260905T204939.904681Z-check/`.
- All eight `full` stages passed:
  `.context/search-harness/20260905T205133.222273Z-full/`, including 1,339 core
  unit tests (17 manual experiments ignored), 65 server tests, 50 broker unit
  tests, 13 broker integration tests, and both real-server RPC tests.
- Native without sync passed the exclusion regression and all 11 selected
  forward-storage tests. WASM release build and all five runtime tests passed.
  Supporting logs are in the parent workspace's
  `.context/bmp-current-{async,wasm-build,wasm-test}.log`.
- The enabled fixture is byte-identical to release 1.8.125: 12,713 bytes,
  FNV-1a-64 `5ccfbcdae3690623`. Disabled storage preserves every inverted byte.
- The first broad run exposed an existing log-capture race with unrelated
  parallel tests. The capture now accepts only its owning test thread; both
  complete runs then passed. Production logging is unchanged.

No new search-latency or throughput claim is made by this cleanup.

## Deletion and maintenance admission (2026-09-06)

Production recreation exposed three lifecycle ordering bugs. Deletion evicted
the registry handle but did not stop the manager before waiting for issued
handles, allowing old Reorder work to retain maintenance capacity. A handler
that already held the index could then wait for its writer behind the delete
lease, forming a second handle-drain cycle. Registry open also swept alleged
orphans before acquiring the OS writer lock, which could delete another
process's unpublished output. Behavior-named regressions reproduced all three.

Deletion now stops manager admission and signals cancellation before the handle
drain; reopening checks the deletion marker before waiting for the lease. Open
uses the existing locked writer opener before loading metadata or cleaning up;
the returned index and writer share one segment manager. This also closes the
stale-snapshot window if another writer finishes during open. Actual blocking work, publication
and deferred deletion still drain before directory removal.

Reorder's shared-writer entry point commits admitted input, releases the writer
lock during maintenance, and uses the existing manager and primary-key refresh
path. Its retained writer Arc preserves the OS lock. Manual BP now uses the
configured background CPU pool. Tests hold all BP capacity while committing
new documents, preserve committed and pending primary keys across replacement,
and cancel queued maintenance without releasing the other index's permit.

The observed fresh-index stall was maintenance waiting, not multi-minute BMP
encoding: new-field BP/rewrite phases logged about 0.2–0.7 seconds while
publication retried 120-second timeouts. After recovery/recreation, all three
documents shards were committing again (94,442 documents at 21:09 UTC). These
are incident observations, not a controlled throughput benchmark.

Focused regression evidence is in the parent workspace's
`.context/lifecycle-{delete,reopen,writer-owner}-red.log`,
`.context/lifecycle-maintenance-tests.log` and
`.context/lifecycle-registry-tests.log`. The lifecycle-only `check` and all eight
`full` stages passed (1,341 core, 68 server, 50 broker unit, 13 broker integration
and two real-server tests), as did native-without-sync maintenance regressions.
One initial broad run hit a mock broker's ephemeral-port collision; rerunning
with `RUST_TEST_THREADS=1` passed without changing production or test behavior.

The API also uses an explicit `AllQuery` inside exclusion filters. The existing
wire variant now maps to a core query sharing the all-document cursor and
bounded bitmap. Regressions cover both common-filter spellings in all fusion
modes, missing metadata, bitmap tail bounds and forward-only cursor seeking.
The wire conversion regression first failed with the unimplemented-query error.
Final combined validation passed:

- `check`: `.context/search-harness/20260905T213650.334212Z-check/`.
- All eight `full` stages:
  `.context/search-harness/20260905T213942.454422Z-full/` (1,344 core tests,
  17 manual experiments ignored, 70 server, 50 broker unit, 13 broker integration
  and two real-server RPC tests).
- Native without sync: both match-all regressions and all three maintenance
  regressions passed. WASM release build and all five runtime tests passed.
  Logs: parent workspace `.context/final-engine2-{async,wasm-build,wasm-test}.log`.
- The lock-before-metadata regression failed before the final opener change;
  both registry ownership regressions then passed. Red evidence:
  `.context/lifecycle-open-snapshot-red.log` in the parent workspace.
  An interim full run was interrupted to make that final change; its process
  group cancellation reported an OS error, so only the complete final run above
  is used as validation evidence.

## Search correctness and execution review (2026-09-06)

Reviewed against `5286d8c5` (1.8.126), tracing the shared query-language parser,
Index/Searcher planning, common filters, Boolean/BMP/MaxScore execution and L1
candidate readers. The fixes stay in those owners; there are no index-format,
wire-format, scoring-formula or approximate-default changes.

### Confirmed and fixed

- `emb:sparse({...})` dropped the field's `query<lsp_gamma: N>` setting.
  Explicit zero therefore became the depth-derived approximate default. The
  core parser now carries `Some(0)` and positive caps through unchanged, while
  an omitted setting remains `None`. This shared parser serves native, async,
  tool and WASM callers. The structured RPC converter already inherited the
  setting. Regressions reproduce the lost zero and a nine-document query
  ignoring a one-superblock cap; parsed results now match an explicitly
  configured core query. The schema reference also now describes the existing
  3000/4000/depth gamma schedule correctly.
- Boolean optimization could flatten a common-filter wrapper, or a nested
  Boolean with required/excluded clauses, into unfiltered sparse terms.
  Scoring decomposition now keeps those constraints opaque. A separate BMP
  planning hook preserves query-global superblock selection for common-filter
  wrappers and Boolean queries with local pure filters. A two-segment test
  checks that a cap of one still admits only one superblock across the index.
- Common eligibility reached some nested BM25/BMP and sparse MaxScore plans
  only after their top-k heaps, allowing disallowed high scores to crowd out
  eligible hits. Local Boolean eligibility now intersects outer eligibility;
  sparse MaxScore receives both eligibility and the request deadline. Known-hit
  regressions cover direct, optimized Boolean and nested-filter plans.
- Common-filter materialization used fresh default options, losing deadlines,
  and continued into scoring even after the intersection became empty. It now
  shares the existing truncation state, discards incomplete bitmaps and stops
  before remaining filters or scoring payloads. The direct synchronous scorer
  also delegates to the implemented filtered path. Work remains bounded by the
  existing bitmap/fallback limits; emptiness checks scan existing bitmap words.
- BMP L1 backfill bounded candidate count but did not admit variable-size
  selected payloads against a byte budget. Forward offsets or selected inverted
  block ranges now reserve bytes before payload validation/scoring, sharing the
  existing 256 MiB lazy-text allowance across segments, features and components.
  The reader performs an O(selected values) metadata pass with constant scratch.
  A zero-budget regression failed before the fix for both forward-storage modes.
  Dead version-dependent logging in the current-format-only BMP reader was
  removed while checking this path.

The eligibility and admission invariants are recorded in
[candidate rescoring](candidate-rescoring.md#eligibility-and-bounded-nomination).
No writer or codec changed, so this pass does not claim a format rewrite or a
new byte-identity experiment.

### Local measurement

The empty-filter experiment uses the same RAM index of 16,384 documents, each
containing `alpha beta gamma` and a numeric eligibility value of zero. It asks
for ten `alpha` hits with eligibility equal to one and verifies an empty result
outside timing. Index construction is excluded. Each process measures eleven
batches of 100 end-to-end searches; three before/after pairs alternate order.
Both binaries use the same Apple M4, Rust 1.98.1 / LLVM 22.1.8 and Cargo release
flags, with no concurrent build during measurement.

| Measurement                                       |                   Before |                    After |
| ------------------------------------------------- | -----------------------: | -----------------------: |
| Median batch latency, three processes (µs/search) | 90.091 / 87.678 / 89.499 | 17.672 / 14.169 / 13.285 |
| Median of those medians (µs/search)               |                   89.499 |                   14.169 |
| Process peak RSS range (MiB)                      |              19.34–20.33 |              18.77–19.69 |
| Eligibility bitmap (bytes)                        |                     2048 |                     2048 |

This is a roughly 6.3× improvement for empty eligibility, not an overall search
speedup or a tail-latency claim. RSS includes the fixture and test process and
does not isolate per-query scratch. The first baseline process incurred 544
page faults; the other five reported zero. Queries use a warmed RAM fixture,
and ordinary desktop activity was present. Cold storage, loaded concurrency,
nonempty-filter latency and a controlled x86 before/after remain unmeasured.
Reproduction: run the ignored release test
`query::filtered::tests::empty_common_filter_benchmark` under `/usr/bin/time -l`.
Raw runs and environment metadata are in this workspace's
`.context/common-filter-benchmark-{before,after}-{1,2,3}.log` and
`.context/common-filter-benchmark-environment.json`.

### Deployed observations and remaining performance work

Read-only SSH/Kubernetes sampling observed the existing 1.8.126 broker and four
servers. No patch was deployed, requests replayed or settings changed. Between
approximately 04:52:42 and 04:59:36 UTC, broker counter deltas recorded 37
successful document searches averaging 448.1 ms and 12 social searches averaging
5.13 ms, with no new recorded errors. These are sparse, mixed live requests
(about 0.12 requests/second combined), not a throughput benchmark. The first
broker summary exported document p50/p95/p99 of 47.1/1303.0/1303.0 ms; these
rolling summary quantiles are not quantiles of the counter-delta interval.

On document shard `s2`, the same interval contained seven `sparse_vectors` LSP
plans averaging 90.0 ms and 56 segment BMP executions averaging 218.7 ms.
Mean execution components included 164.3 ms in block scoring, 49.3 ms in
prefetch, 3.64 ms in the D grid and 1.06 ms in document mapping. Four short-document
sparse plans averaged 25.3 ms; their 28 segment executions averaged 289.7 ms,
including 268.4 ms in block scoring. These component means identify profiling
targets; segment work can run concurrently and must not be summed as request
wall time. Sampled LSP counter deltas averaged gamma 3000, but metrics do not
identify the request syntax or prove those requests used a schema gamma of zero.

Two process/cgroup snapshots on `s2`, 337 seconds apart, showed RSS about
63.6 GiB (46.9–47.1 GiB anonymous and 16.5–16.7 GiB file-backed), 1.75 GiB locked
and no swap. Cgroup usage rose from 68.6 to 73.3 GiB; its historical peak was
206.0 GiB, while process peak RSS was 168.7 GiB. The interval added 7.20 million
major faults and 7.99 million file refaults, with CPU consumption averaging
5.53 cores and no quota throttling. Recent memory-pressure averages were low
and OOM counters were zero. These are whole-container observations, not
per-query allocations; neither faults nor the historical peaks can be assigned
to search versus background work from these samples alone.

Remaining work is to profile block scoring and selected-range I/O on a fixed
production-like corpus, correlating faults with per-request and maintenance
activity, then measure warm/cold latency and recall at the same explicit gamma.
The slow log includes candidate exports taking 3.1–6.5 seconds, some with empty
segment heaps and zero thresholds; selective-filter/candidate-depth behavior
deserves that controlled follow-up. The documented bounded ANN nomination can
still underfill a selective common filter. This pass preserves that policy;
changing it needs a recall/work comparison. No production default was tuned.

`kubectl top` failed because the cluster Metrics API is unavailable. Direct
Prometheus scrapes plus `/proc` and cgroup files supplied the observations
instead. Raw evidence is retained under `.context/review-prod-*`, with derived
tables in `.context/review-production-{analysis,resources-analysis,bmp-breakdown}.json`.

### Validation

- Required `python3 scripts/check_search.py check` passed: formatting, Clippy,
  1,356 core unit tests (18 manual experiments ignored), 70 server tests,
  50 broker unit tests, 13 broker integration tests, tool/integration/doc tests
  and the native-without-sync compile boundary. Run evidence:
  `.context/search-harness/20260906T045858.391501Z-check/`.
- Native without sync passed eight filter regressions, all 22 query-language
  tests and the BMP admission regression: `.context/review-native-async.log`.
- WASM release compilation and all seven runtime tests passed, including
  query-language searches with exhaustive and bounded schema gamma. Evidence:
  `.context/review-wasm-{npm-install,build,test}.log`.
- The extended `full` harness/real-server RPC tests, a Linux mlock experiment,
  controlled cold-cache and cross-architecture comparisons were not run in
  this pass. Lifecycle/RPC implementations and residency policy were unchanged.

## Search attribution, traces and symbolic L1 (Dushanbe, 2026-09-06)

This follow-up replaces the coefficient-only L1 API with required `l1.formula`
(capability 3, `formula_v1`). Removed coefficient fields have reserved protobuf
names/tags. `backfill` and raw missing defaults remain, with zero as the default
for a missing formula variable. Core owns the bounded compiled expression and
passage/document inference; server and broker translate and validate. Expressions
compile once per request, bind at most 17 indexed inputs, and have no global cache.
Python and TypeScript bindings and examples use the formula-only contract.

Resolved correctness findings and added observability:

- Native nomination without supplied text statistics bypassed the ordinary
  searcher's query-global BM25 statistics. A two-segment regression first
  reproduced different scores/ranking; nomination now uses the same statistics
  owner. Optional diagnostics preserve ordinary hit identities and score bits.
- `include_rrf_scores` returns a separate score and per-branch votes based on
  complete organic nominations, excluding backfill/score-only features. Broker
  attribution uses global ranks, including nominations absent from final hits.
- `tracing=false` preserves the default path. Opt-in traces retain bounded
  per-shard/per-branch nominations, raw scores and ordinals, query trees/common
  filters, nomination depth/counters and shard selection, across pagination.
  Candidate/ordinal and retained/encoded byte budgets apply before cloning.
- RRF in L1 must be evaluated inside the passage formula, before its document
  combiner. Global RRF can change both the best passage and winning document,
  so the broker obtains the whole bounded union and every scored passage.
  Shards use a constant formula for this export; the broker applies the actual
  formula with global votes. Regressions cover below-local-top-k winners,
  discarded passages, negative/zero multipliers, logarithms/division of RRF,
  incomplete exports, old backends and invalid formulas before admission.

### Fixed-fixture performance evidence

Apple M4 arm64, Rust 1.98.1, identical debug compiler flags and warm local gRPC.
The saved pre-formula server and new server used the same persisted 1,000-document
seed-7 fixture, two BM25 branches, nomination depth 40 and top 20, with raw feature
exports. Each mode started a fresh process, warmed up 20 times and measured 100
sequential calls; no concurrent builds ran. RSS is the maximum sampled whole
process RSS, not isolated scratch or an OS peak. These desktop debug samples are
smoke measurements, not production throughput or evidence to change defaults.

| L1 request                                            | p50 ms | p95 ms | Response bytes | Sampled RSS KiB |
| ----------------------------------------------------- | -----: | -----: | -------------: | --------------: |
| Previous coefficients, 0.2 title + 0.8 body           |  4.050 |  4.319 |          1,826 |          36,672 |
| Equivalent compiled formula                           |  4.020 |  4.136 |          1,825 |          36,784 |
| 0.2 title + 0.8 log1p(body) + 3 RRF, with attribution |  4.521 |  4.689 |          5,934 |          38,976 |

The equivalent formula returned identical document IDs and f32 score bits.
All three modes matched independently computed f64 arithmetic with checked f32
rounding. The tiny timing difference between coefficients and the equivalent
formula is inconclusive; the nonlinear/RRF mode performs additional ranking
and attribution work. Raw evidence and fixture script are in
`.context/formula-performance-samples.jsonl` and `.context/measure_l1_formulas.py`.

A separate run on the same fixture/configuration compared legacy fusion with
optional diagnostics using the new binary:

| Diagnostics     | p50 ms | p95 ms | Response bytes | Sampled RSS KiB |
| --------------- | -----: | -----: | -------------: | --------------: |
| Off             |  3.419 |  3.579 |          1,371 |          35,648 |
| RRF attribution |  3.710 |  4.773 |          5,517 |          36,880 |
| Trace           |  3.627 |  3.963 |          6,184 |          36,528 |
| Both            |  3.876 |  4.328 |          6,656 |          36,512 |

All four modes returned identical IDs and score bits. Traces retained all 80
organic branch nominations; RRF diagnostics matched the fusion scores exactly.
Evidence: `.context/formula-diagnostics-samples.jsonl` and
`.context/measure_search_diagnostics.py`. Earlier production observations above
remain applicable; this follow-up did not tune ANN/LSP defaults or deploy to the
sampled installation. Cold-cache, sustained-load, large-expression/many-passage
and x86 comparisons remain future performance work.

### Validation evidence

The final required `check` passed all four steps:
`.context/search-harness/20260906T073146.024481Z-check/`.
The full search harness passed all eight steps, including real-server broker RPCs:
`.context/search-harness/20260906T072445.352810Z-full/`. Eleven candidate-scoring
regressions also passed with native async execution (no sync feature), and the
WASM release build plus all eight runtime tests passed. Both regenerated client
suites passed eleven tests each. Evidence includes `.context/formula-native-async.log`,
`.context/formula-wasm-{build,tests}.log`, and
`.context/formula-{python,typescript}-tests.log`.

## Filtered body queries at small limits — 2026-09-06

Confirmed against `fa92ac62` (1.8.128). With ten higher-scoring `article`
documents and one lower-scoring `book`, a required chunked text term combined
with `kind:book` returned no hits at limit 1 and the correct book at limit 20.
The generic Boolean planner constructed the body's bounded scorer before
applying its document predicates. Required nested disjunctions and negative
fast-field filters had the same failure. The public query-language reproduction
is `body:machine AND kind:book` with a chunked `body` and raw fast `kind`.

The generic plans now push available predicates into shared eligibility before
constructing their text children. They reuse selective bitset construction
where supported and scan eligible fast-field values otherwise, retaining the
original scoring/verifier clauses and the existing 16 MiB bitmap ceiling.
The regression checks small/large-limit IDs, exact score bits and ordinals;
another covers required disjunctions on plain text. Fast text equality is used
only where it matches indexed-term semantics: single-valued raw text. Analyzed
and multi-valued indexed fields retain their posting-list semantics.

Actual WASM execution of this query also reproduced a MaxScore panic from an
unconditional `std::time::Instant::now()` used for diagnostic logging. Both
MaxScore loops now use the existing portable `observe::WallTimer`.
Red evidence: `.context/filtered-body-red.log`,
`.context/filtered-body-fast-text-red.log`, and
`.context/filtered-body-wasm-red.log`.

### Controlled fixture and limits of the measurements

Apple M4 arm64, Rust 1.98.1, identical debug server build commands and the same
persisted 10,000-document fixture (50 eligible books, one body chunk per
document), top 1. Each case opened the fixture in a fresh process, warmed 20
queries and measured 100 sequential RPCs. Index construction and the full-limit
correctness reference were outside timing; builds did not overlap sampling.

| Query shape                       | Before p50/p95 ms | After p50/p95 ms | Before/after sampled RSS KiB | Before/after correct top 1 |
| --------------------------------- | ----------------: | ---------------: | ---------------------------: | -------------------------- |
| Required body term + kind         |     1.090 / 1.232 |    1.475 / 2.005 |              34,640 / 36,352 | No / Yes                   |
| Required nested body OR + kind    |     1.679 / 2.816 |    1.422 / 1.737 |              35,328 / 35,456 | No / Yes                   |
| Required body term, excluded kind |     1.536 / 2.926 |    1.580 / 1.746 |              37,392 / 34,432 | No / Yes                   |
| Existing filtered SHOULD control  |     1.742 / 1.996 |    1.367 / 1.539 |              34,688 / 35,728 | Yes / Yes                  |

All fixed responses matched the full-limit reference IDs and score bits.
The control's movement makes the timing comparison inconclusive; this is a
correctness fix with bounded extra eligibility work, not a throughput claim.
RSS is sampled process residency, not a measurement of peak scratch. The
bitmap for this fixture occupies 1,256 bytes. Evidence and reproduction:
`.context/filtered-body-{before,after}.jsonl` and
`.context/measure_filtered_body.py`.

This fixes predicate pushdown, not every source of candidate loss. Filters
without a document predicate retain the existing verifier paths. The documented
two-times chunk nomination cap can still retain several chunks of one document
and underfill a document result window; this pass does not change that policy,
ANN/LSP defaults, stored formats, or production configuration. Cold-cache,
large-corpus, concurrency and x86 performance comparisons remain unmeasured.

### Validation and remaining async discrepancy

The final required harness passed all four steps:
`.context/search-harness/20260906T133139.703927Z-check/`. The WASM release build
and all nine runtime tests passed, as did documentation checks and all three
new native-without-sync regression tests. Logs:
`.context/filtered-body-{final-check,wasm-build,wasm-tests,docs}.log` and
`.context/filtered-body-native-async-regressions.log`. No lifecycle, RPC, or
wire definitions changed; the additional `full` lifecycle/RPC harness was not
rerun for this patch. The measured probes did exercise a real local server.

A broader native-without-sync chunked suite passed 16 of 17 tests and exposed
an existing discrepancy in `filters_and_phrases_push_into_chunked_text_maxscore`:
the async fallback includes the required phrase's ordinal 0, returning `[0, 1]`
where the sync bitset-filter path returns `[1]`. The saved 1.8.127 test binary
(`hermes_core-14836e5caef7dcbc`, from the previous review) fails the same test
identically, confirming this is not introduced by the patch. Async phrase-filter
materialization/scoring parity remains a separate correctness follow-up.
Evidence: `.context/filtered-body-native-async.log` and
`.context/filtered-body-preexisting-async-phrase.log`.

## Missing one-word phrase filters — 2026-09-06

Confirmed against `56bcae64` (1.8.129): an indexed `PhraseQuery` with one
absent term returned `None` from native bitmap materialization. `None` means
unsupported, so common filtering attempted its scorer fallback and rejected
segments above 200,000 documents. A 200,001-document regression reproduced
the reported "common filter cannot be materialized" error. Structured phrase
requests preserve `PhraseQuery` even with one token; the query-language parser
normally lowers a single-token quote on one field to `TermQuery`, which does
not exercise this phrase path.

The phrase owner now distinguishes a successful posting lookup with no list
from an unavailable materialization. Missing indexed terms produce the empty
bitmap already allocated by this path. Read errors still return an unavailable
bitmap and fall back to explicit errors; they do not become empty matches.
Non-indexed fields retain the scorer fallback because fast-column values can
match even without postings. No parser, wire or persisted representation changed.

The regression covers direct async and sync scorers plus public index search,
plain/chunked text, unpopulated indexed fields, and positive, OR and negated
filters. A smaller fixture also checks present/absent terms against indexed and
fast-only fields across the native-without-sync boundary. Red and green evidence:
`.context/missing-phrase-{red,green}.log`.

The cost remains one document bitmap (25,008 payload bytes for the large
regression), plus the existing term lookup and matching-posting traversal.
The 16 MiB bitmap and 200,000-document scorer-fallback bounds are unchanged.
No latency or RSS benchmark was run for this correctness patch, and no
performance improvement is claimed. Portable builds still have the documented
fallback bound; the async phrase-ordinal discrepancy recorded above remains
outside this fix.

Validation passed: the four-step required harness at
`.context/search-harness/20260906T154504.235296Z-check/`, all ten active common-filter
tests with native async execution (one manual benchmark ignored), and the WASM
release build plus all nine runtime tests. Documentation checks also passed.
Logs: `.context/missing-phrase-{check,native-async,wasm-build,wasm-tests,docs}.log`.
The additional `full` lifecycle/RPC harness was not rerun because this patch
changes neither lifecycle nor RPC code.

## Long L1 phrase features — 2026-09-06

Confirmed against `cf170fbb` (1.8.130): a 64-token phrase succeeds in both
ordinary search and candidate scoring, while 65 tokens fail candidate-plan
validation with "invalid L1 phrase feature or missing positions". The server
converter admits up to 256 tokens by default. L1 reused the 64-term nomination
limit even though the shared phrase scorer uses dynamic positional cursors.
The same plan validation runs for learned ranking and raw feature collection.

Phrase feature validation now has its own 256-token bound, matching the default
server conversion budget. It preserves every term and existing offsets/slop.
An oversized core request fails with the feature name, actual term count and
maximum before statistics or candidate resolution, including an empty candidate
set. Field/position checks, nonphrase feature limits, nomination limits and
wire/storage formats are unchanged.

The core regression uses synthetic `term0` through `term255` on plain and
chunked fields. At 64, 65 and 256 tokens, raw exports and formula scores match
ordinary phrase search bit-for-bit. Documents with a wrong 65th term or missing
256th term score zero, so accepting the request cannot hide truncation. A
257-token candidate request still fails. The broker reproducer uses the same
four synthetic documents over two real server processes and checks formula
ranking and collection against ordinary distributed phrase scores.

Reproduction commands using synthetic data:

```sh
cargo test --locked -p hermes-core --lib long_phrase_features_keep_every_term_in_ranking_and_collection
cargo build --locked -p hermes-server --bin hermes-server
cargo test --locked -p hermes-broker --test e2e_real_server broker_ranks_and_exports_long_phrase_features_without_dropping_terms -- --ignored
```

The algorithm is unchanged: candidate phrase scoring keeps one posting cursor
and reusable position buffer per term, seeks only nominated physical targets,
and retains existing read and scored-value admission budgets. Increasing the
accepted phrase length increases that bounded work; this is a correctness fix,
not a throughput claim. No before/after latency or memory benchmark was run
for this patch. Red/green core evidence is in
`.context/long-phrase-{red,green}.log`. The previously recorded native-async
required-phrase ordinal discrepancy remains outside this change.

The required `check` passed all four steps with normal test concurrency:
`.context/search-harness/20260906T180558.238875Z-check/`. All eight full-harness
steps passed with `RUST_TEST_THREADS=1 python3 scripts/check_search.py full`:
`.context/search-harness/20260906T181122.939063Z-full/`, including all three
real-server broker tests. Eight candidate-scoring tests also passed with native
async execution. Logs: `.context/long-phrase-{final-check,full-serial,native-async,broker}.log`.

Parallel validation intermittently timed out in two existing broker integration
tests while waiting ten seconds for initial index discovery:
`client_deadline_propagates_and_absence_means_untimed` in the first `check`, and
`partitioned_stream_routes_each_flush_by_primary_key` in the parallel `full`.
Neither exercises candidate phrase scoring. The focused deadline retry and a
fresh default-concurrency `check` passed; the final `full` used one test thread.
No timeout or production setting was changed. The cause of these timeouts remains
unresolved; original failures are retained in `.context/long-phrase-{check,full}.log`
and the focused retry in `.context/long-phrase-broker-deadline-retry.log`.

The WASM release build and all nine existing runtime tests passed, along with
documentation checks and pre-commit hooks. Evidence:
`.context/long-phrase-{wasm-build,wasm-tests,docs,final-precommit}.log`.

## Index-configurable L1 phrase limits — 2026-09-06

Following the fixed 256-token limit in 1.8.131 (`2bf99929`), the creation schema
now accepts `max_l1_phrase_terms`. The requested default is restored to **64**;
indexes opt into longer features explicitly, for example
`index documents { max_l1_phrase_terms: 256 field body: text [indexed<token_position>] }`.
The optional positive-u32 setting is owned by core `Schema` and persisted under
`schema.max_l1_phrase_terms` in `metadata.json`. SDL, JSON creation, Rust builders,
server/broker creation and WASM creation share it. Index info returns the
effective value in SDL. No protobuf or segment format changes are involved.
Absent settings retain the prior metadata bytes and load as 64; explicit zero,
negative, fractional and out-of-range values fail creation/deserialization.

Before the change, the new default-limit regression incorrectly accepted a
65-term phrase with no nominated candidates (`.context/phrase-cap-red.log`).
It now checks the configured boundary in ranking and collection after reopen,
including custom limits 1, 65 and 300. The existing long-phrase fixture explicitly
sets 300 and compares complete plain/chunked phrase features and formula
predictions against ordinary search at 64, 65, 256 and 300 terms. Wrong 65th terms
and missing 256th terms still score zero. A two-shard RPC fixture checks both
default-64 and configured-256 indexes, reported schemas, and invalid creation.
WASM runtime coverage checks default/configured metadata across reopen and a
second commit, plus rejected zero limits.

The setting uses one inline `Option<NonZeroU32>` with no heap allocation;
validation reads it once per phrase component. Phrase probing is unchanged:
per-phrase cursor scratch and posting/position probes grow linearly with retained
terms. The shared 256 MiB candidate payload-read budget, scored-value budget,
nomination limits, and separate server token budget (default 256) still apply.
No latency/RSS benchmark was run for this configuration change and no performance
improvement is claimed. The previously recorded native-async phrase-ordinal
discrepancy and intermittent parallel broker startup timeouts remain separate
findings.

The initial required check exhausted local disk space while linking tests
(`.context/phrase-cap-check.log`). Removing only this workspace's rebuildable
incremental caches predating the task freed 45 GiB. The next normal-concurrency
check reached broker tests but timed out in the existing
`partitioned_create_and_commit_fan_out_to_every_partition` test, waiting ten
seconds for initial index discovery (`.context/phrase-cap-final-check.log`).
This is the same unresolved discovery failure recorded above; no production
behavior or test timeout was changed. Subsequent harness runs use one test thread.
The first serial full run passed all 1,369 core unit tests, then could not
execute an integration binary: the workspace's entire `target` directory
disappeared during the run, beyond the earlier bounded incremental-cache cleanup.
Its log is `.context/phrase-cap-full-serial.log`. Validation was restarted with
`CARGO_TARGET_DIR=$PWD/.context/l1-phrase-cap-build` to isolate build artifacts.

The isolated full run passed its first seven steps, including all search-stack
tests, native-without-sync and portable compilation, API docs, and the server
build (`.context/search-harness/20260906T184531.789002Z-full/`). Its standalone
broker step caught a native-gated helper used only by the new index-info test
assertion; replacing that helper with the portable SDL parser fixed the feature
boundary. Rerunning that exact final step passed all three real-server tests,
including both phrase-cap configurations (`.context/phrase-cap-broker.log`).
The WASM release build and all 12 runtime tests passed in the separate
`.context/l1-phrase-cap-wasm-build` directory; logs are
`.context/phrase-cap-wasm-{build,install,tests}.log`.
All nine candidate-scoring tests also passed with native async execution
(`cargo test --locked -p hermes-core --no-default-features --features native --lib
query::candidate_scoring::tests`), including the 300-term exact-score checks:
`.context/phrase-cap-native-async.log`.

## Row deletion and compaction — 2026-09-07

Implementation base: `7a4b380c1dedfebac5be7a02b7924b93eadb9051`
(`origin/main`, 1.8.132). The [row-deletion contract](row-deletion.md) documents
format 7, immutable visibility, atomic full-document upserts, exact live-key
checks behind the existing Bloom filter, and single-segment compaction. The
implementation operates on encoded fields, so indexed-only values survive.
ANN codebooks, assignments, fingerprints, and surviving codes are retained.

### Correctness and lifecycle evidence

- `python3 scripts/check_search.py check` passed all four steps; evidence:
  `.context/search-harness/20260907T074632.727500Z-check/`.
- `python3 scripts/check_search.py full` passed all eight steps, including
  1,381 core tests, 77 server tests, broker/tool tests, portable/native-without-sync
  compilation, docs, and all three real-server broker E2E tests; evidence:
  `.context/search-harness/20260907T075229.998139Z-full/`.
- Follow-up regressions and final Clippy cover indexed-only numeric fields,
  binary vectors, JSON/byte storage, missing/multi-value fast fields, all-deleted
  dense/sparse segments, stacked PK dictionaries, stale visibility rejection,
  and global-before-local maintenance admission. The physical-copy merger
  rejects masked readers instead of silently discarding their visibility.
  Logs: `.context/deletion-final-{rows,pk,capacity,clippy}.log`.
  The five end-to-end deletion tests also pass with
  `--no-default-features --features native` (async search without `sync`);
  evidence: `.context/deletion-final-native-async.log`.
- Native tests preserve old searchers through commit and compaction, reopen
  cached Bloom filters, allow deleted keys to be reused, reject live/pending
  duplicates, and cover abort, publication failure, cancelled commits, and
  cancelled compactions with blocking writers. Shutdown drains the latter.
  PK reopen coverage includes Unicode keys and distinct case variants.
- ANN tests compare complete compacted bytes with canonical encoders for TQ,
  IVF-TQ, binary IVF, ScaNN AH, and ScaNN binary. Sparse block tests compare raw
  weights for Float32, Float16, UInt8, and UInt4. End-to-end BMP/MaxScore tests
  compare surviving scores and chunk/value ordinals.
- `hermes-wasm/build.sh`, `npm ci`, and `npm test -- --run` passed; the final
  rebuild and 13 tests also passed after portable warning cleanup. A persisted
  indexed-only text index reopens with tombstones, hides deleted hits/hydration,
  preserves an old reader, and rejects a corrupt mask. Logs are under
  `.context/deletion-wasm-*.log`.
- A local CLI smoke test ran create/upsert/update/delete/compact/merge/reopen;
  only the replacement remained, with one physical row and no mask. Evidence:
  `.context/deletion-cli-smoke.log` and `.context/deletion-cli-smoke-verify.log`.
  The initial inspection script used the wrong metadata filename; the final
  verifier reads `metadata.json` and passes.

### Matched microbenchmarks

Before and after used the same unchanged `segment_merge` and `search_pipeline`
fixtures, Rust 1.98.1, release/bench flags, and Apple M4 / 32 GiB / macOS 15.6.1.
`RUSTFLAGS` was unset. Criterion used 20 samples, 0.5 s warmup and 1 s measurement.
The binaries were built separately and copied before execution; baseline source
was a detached checkout of the SHA above. No agent-owned compiler/test ran
during measurements, but other work on the shared machine was not controlled.

The first pass ran before then after. A second pass reversed order and repeated
all merge cases plus the three search cases below. Numbers are Criterion point
estimates in microseconds; large variation prevents production latency claims.
These fixtures measure clean segments with no deletion masks. They do not
measure large-corpus deletion scans, dirty compaction, cold disk I/O, or recall.

| Fixture                              | Before, pass 1 | After, pass 1 | Before, reverse pass | After, reverse pass |
| ------------------------------------ | -------------: | ------------: | -------------------: | ------------------: |
| Copy fast columns, 2 × 4,096 rows    |          26.46 |         37.15 |                30.39 |               40.46 |
| Missing fast column, 2 × 4,096 rows  |          27.03 |         28.16 |                27.40 |               28.93 |
| Copy fast columns, 2 × 65,536 rows   |          74.28 |         68.96 |                77.49 |               51.20 |
| Missing fast column, 2 × 65,536 rows |          52.62 |         52.32 |                58.85 |               55.65 |
| Dense top 10, multi-thread search    |          51.88 |         36.56 |                54.64 |               36.08 |
| Hybrid top 200, multi-thread search  |         561.52 |        497.69 |               498.47 |              340.07 |
| Dense top 200, current-thread search |         191.41 |        203.17 |               175.92 |              156.55 |

`/usr/bin/time -l` measured whole-process peak RSS, including fixture building
and Criterion, rather than allocator-only scratch. Merge RSS was 55.8 → 64.7 MiB
in pass 1 and 60.0 → 61.0 MiB in the reverse pass. Full-search RSS was
84.7 → 91.0 MiB; the matched three-case repeat was 74.8 → 76.2 MiB. These
measurements cannot isolate mask residency because the fixtures have no masks.
The exact mask file cost is `24 + 8 * ceil(rows / 64)` bytes; reader masks use
one bit per physical row, with separate PK live-ordinal masks and compressed
row-statistic columns accounted as described in the design.

Raw logs: `.context/deletion-bench-{before,after}-{segment_merge,search_pipeline}.log`,
`.context/deletion-bench-repeat-{before,after}-{segment_merge,search_pipeline}.log`,
and `.context/deletion-bench-environment.txt`. Criterion samples are under
`.context/deletion-bench-run/target/criterion/`. The reverse pass includes the
final physical-copy visibility guard; subsequent changes only add tests/docs.

### Remaining review findings and limits

- The small clean-copy merge fixture regressed in both passes (about 10 µs,
  33–40%). The larger copy fixture improved, and current-thread search changed
  direction between passes. This was unresolved at that stage; the follow-up
  below profiles the empty-dictionary cost and removes it. These microbenchmarks
  do not establish a production throughput change.
- The initial implementation compacted dirty merges automatically and wrote a
  scratch segment first. The follow-up below removes that default cost: ordinary
  merges retain masks, and explicit compaction rewrites each final output directly.
  No cold-storage, large-corpus throughput, or cross-architecture measurement was run.
- Compaction has explicit bounded scratch admission. Very large row maps or
  high-frequency positioned terms can exceed the supplied budget and return an
  error; there is no silent truncation or unbounded fallback.
- Bloom filters retain deleted keys as false positives. Heavy churn can reduce
  their selectivity; exact live-key checks preserve correctness. Adaptive Bloom
  rebuilding and long-running churn throughput were not measured here.
- Format 7 requires rebuilding older indexes. Updates replace full documents
  and permit one pending insertion/update per key per commit. RPC mutation
  endpoints/cross-shard atomic updates are outside this native-core/CLI change.
  BM25 statistics remain physical until compaction.

## Explicit compaction and performance review — 2026-09-07

The follow-up changes ordinary merge to retain tombstones. `ForceMerge.compact`
(and the Python/TypeScript helpers and CLI `merge --compact`) compacts each final
output once, including a singleton. The existing optimizer selects segments from
physical/deleted metadata counts at a configurable ratio (default 0.30; 0 disables).
It shares task slots, CPU pools, global/local merge capacity and the optimizer BP
gate. One automatic compaction is allowed globally, followed by a 60-second
completion cooldown. This requires the existing optimizer to be enabled; it does
not create another worker pool. Compaction scratch defaults to 256 MiB.

### Review findings and implemented improvements

- **Default merge cost:** encoded payloads remain on their ordinary copy/remap
  paths. Only mask words are remapped (`O(physical_rows / 64)`). Address-preserving
  single-source reorder/ANN rewrites reuse the exact immutable mask file. Explicit
  compaction writes directly from final sources; it does not compact intermediate
  merge outputs or reconstruct indexed-only values from the document store.
- **Search and primary keys:** clean searches borrow `None` without allocating a
  deletion bitmap; dirty searches borrow the generation's mask and apply it before
  top-k/pruning admission. A final visibility wrapper protects generic scorers.
  PK Bloom bits are never cleared. Exact dictionary membership is checked against
  a precomputed live-ordinal bitmap, keeping insert checks free of per-key row
  scans. The bitmap is rebuilt once per changed visibility generation.
- **Compaction CPU:** fast columns previously encoded every 4,096-row chunk twice
  to produce a leading directory, allocating a temporary value vector per row.
  They now retain a bounded prefix of encoded chunks and use existing value
  iterators. The uncached suffix uses the same deterministic second-pass encoder.
  Cache capacity (including vector capacities) is capped at a quarter of remaining
  scratch; directory entries reserve another quarter and chunk encoding half.
- **Compaction memory:** physical row maps now reserve `4 * (physical + live)`
  bytes instead of `8 * physical`, using validated deletion counts. They reject
  unexpected extra survivors before reallocating. At 50% deletion this saves 25%
  of row-map storage; at 90% it saves 45%. Chunk maps retain a conservative bound
  where the live virtual-record count is not known in advance.
- **Empty dictionary overhead:** a macOS `sample` profile found FST registry
  initialization dominating the small numeric-only merge fixture in both main
  and this branch (957/940 top-of-stack samples respectively in the two captures).
  The owning block-index encoder now caches only its canonical empty encoding,
  bounded by a test to 128 bytes. It retains no FST build registry and leaves the
  nonempty path unchanged. Tests compare complete bytes and empty lookup behavior.
- **Ordering and statistics:** compaction stably filters physical and per-field
  BMP order, rebuilds affected block/statistic metadata, and retains BP history.
  Previously reordered surviving BMP layouts lose convergence; compaction neither
  resets nor consumes the lineage's attempt budget. Index-info aggregates counts
  in one pass; the broker computes a weighted ratio from summed counts.
- **Lifecycle cost/correctness:** measurement found that CLI exit could interrupt
  retired-file cleanup. Row mutation/merge commands now stop workers, release
  writer snapshots, and drain core cleanup, including on maintenance errors.
  ForceMerge now uses Commit's admission rule: cancellation before obtaining the
  writer starts no task; admitted work owns the writer through reader refresh.
  Detached failures are logged. Both issues have failing-before/passing-after
  regression tests (`compaction-{cli-drain,rpc-admission}-{red,green}.log`).

### Matched measurements

All figures below use the same Apple M4 / 32 GiB Mac, Rust 1.98.1, default
sync/native features and unchanged fixtures between each pair. `RUSTFLAGS` and
`CARGO_ENCODED_RUSTFLAGS` were unset. No task-owned compiler or test ran during
measurements; other applications on this shared machine were not controlled.
These are synthetic CPU/allocator fixtures, not production latency or recall.

The `segment_merge` benchmark now includes `row_compaction/mixed_fast_columns`:
one primary-key text column plus eight numeric columns, missing values, one
multi-value column, 50% deleted rows, and a 32 MiB compaction budget. Fixture
building/deletion/validation is outside timing. Each iteration calls the actual
compactor and overwrites one unpublished RAM output. The baseline binary was
preserved before the chunk cache, iterator and row-map improvements; a final
binary also includes the empty-FST cache. Criterion used 20 samples, 0.5-second
warmup and 1-second requested measurement (extended for slow iterations).

| Physical rows |    Before | Chunk/map changes | Before, reverse repeat | Chunk/map changes, reverse repeat |     Final |
| ------------- | --------: | ----------------: | ---------------------: | --------------------------------: | --------: |
| 4,096         | 3.1107 ms |         1.8308 ms |              3.1188 ms |                         1.7829 ms | 1.8179 ms |
| 65,536        | 113.25 ms |         60.287 ms |              113.25 ms |                         62.288 ms | 60.138 ms |

This is about **42–47% less compaction time**, with the improvement surviving
reversed execution order. Complete `.fast` outputs match byte-for-byte at both
sizes: 153,906 and 2,477,059 bytes. Captures and SHA-256 digests are recorded in
`.context/compaction-perf-evidence.json` and `compaction-perf-{before,final}-*.fast`.
The fixture's first 65K setup attempt hit `QueueFull`; the harness was corrected
to wait for admission, and both compared binaries use that corrected setup.

Peak whole-process RSS **increased**: 145.2 → 163.6 MiB in the first pair and
133.6 → 146.1 MiB in the reverse pair; final was 161.4 MiB. RSS includes index
building, source readers, allocator retention and output buffers, so it does not
isolate scratch. The encoded cache trades bounded memory for CPU, within the
existing cap. Physical map allocation is separately bounded by the exact formula
above; a regression verifies a mostly-deleted map fits that smaller budget and
refuses growth beyond its admitted survivor count.

The earlier small clean-merge regression was investigated rather than dismissed.
With a 1-second warmup and 3-second measurement, the unchanged 2 × 4,096 numeric
fixture measured 27.801 µs on `origin/main` versus 34.104 µs before the empty-FST
fix. Final measured 5.328 µs; the reversed main repeat was 36.501 µs. The broad
spread demonstrates machine noise, but removal of repeated empty-registry work
is clear. A final all-case run measured 4.772/4.579 µs for 4K copy/missing columns
and 27.701/17.056 µs for 64K copy/missing columns. This benefit applies to empty
term dictionaries; it is not a claim that populated text merges improve equally.
The sampled profiles are attribution evidence, not the source of timing numbers.

A separate matched **debug CLI** fixture used two 4,096-row segments with a stored
primary key, numeric fast field, indexed-only 129-token text, and 50% deletion.
Three runs rotated execution order and checked all 4,096 surviving keys after
each command. After adding cleanup draining, median default merge was 0.10 s
(range 0.10–0.65), explicit compaction 0.39 s (0.39–0.40), and the initial implicit
compaction implementation 0.40 s (0.38–0.47). Median RSS was 29.8/31.7/30.8 MiB,
respectively. The old command could leave retired files, so its command time does
not include an equivalent cleanup guarantee. Both final modes left only owned
segment files. This run predates the chunk/FST optimizations; it establishes the
cost distinction of the API flag, not final release throughput.

Raw evidence: `.context/compaction-perf-{before,after,final}.log`,
`.context/compaction-perf-repeat-{before,after}.log`,
`.context/compaction-clean-{main,current,final,main-repeat}.log`,
`.context/compaction-clean-sample-{main,current}.txt`, and
`.context/compaction-cost/final/`. Build logs and binary hashes accompany the
fixture captures. Ignore Criterion's automatic cross-run percentage overlays;
the table above compares the recorded point estimates from the named binaries.

### Remaining limits and follow-up measurements

- No large-corpus ANN deletion/compaction, cold-storage throughput, x86/AVX2,
  sustained churn, or production p99 measurement was run. Defaults were not tuned
  from these fixtures. The 30% trigger and cooldown are configurable policy.
- Visibility-only refresh currently reopens the affected segment's compact
  metadata, including validation, while payload mappings remain evictable.
  Sharing more immutable decoded metadata across visibility generations needs a
  separate measured change to reader ownership; this review does not claim that
  a deletion costs only the mask write.
- Deletion still scans affected PK columns. Persisted Bloom filters can lose
  selectivity under heavy churn, although exact live-key checks remain correct.
  Neither adaptive Bloom rebuilding nor a new reverse PK-to-row index was added.
- Oversized compaction maps, terms, positions or column values fail before
  publication when the supplied scratch budget cannot cover them. Global pacing
  also conservatively cools down admission-skipped attempts; busy workloads can
  defer compaction until a later scan. Failure retries use existing capped
  exponential backoff, rather than an unbounded busy loop.

### Final validation

- The regular `check` harness passed in
  `.context/search-harness/20260907T091258.286326Z-check/`; a later parallel `full`
  also passed in `20260907T093125.569030Z-full/` before the performance refinements.
- Final `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all eight
  steps in `.context/search-harness/20260907T100254.254219Z-full/`: 1,389 core,
  80 server and 5 tool unit tests, broker tests, native-without-sync and portable
  compilation, docs, and all three real-server broker E2E tests. Individual
  concurrency tests retain their own worker/runtime concurrency.
- Two parallel final attempts hit the broker harness's 10-second discovery wait;
  the later failure log contains `Address already in use` from its bind/drop port
  probe. Isolated retry passed. Serial test scheduling avoided that harness race;
  production settings and the test timeouts were not changed. Failure evidence:
  `20260907T092846.698857Z-full/03-test.log` and
  `20260907T095930.504752Z-full/03-test.log`.
- Final WASM build and all 13 tests passed, including persisted visibility and
  corruption handling (`.context/compaction-review-wasm-{build,test}.log`).
  Python and TypeScript client unit tests each passed 12 tests, including flag
  serialization and deletion statistics; generated bindings were refreshed using
  the repository scripts (`.context/compaction-{python,ts}-*.log`).
- Final CLI smoke verified default merge retains 8,192 physical/4,096 live rows,
  then explicit singleton compaction leaves 4,096 physical/live rows and no
  tombstones. All 4,096 indexed-only text matches retain their exact primary keys,
  and no retired segment files remain (`.context/compaction-final-smoke.log`).
  Documentation/link checks and `git diff --check` passed.

## Second deletion review: identity, cancellation, and overlapping maintenance

This pass traced native mutation admission through manager publication, PK cache
refresh, ordinary merge, compaction, and retirement. It also checked the shared
fast-column decoder used by native async and WASM builds. Three additional bugs
were reproduced before fixing them:

- A document with two primary-key values reserved its first value but persisted
  its last value in the single-value fast column. Inserts and updates now share
  one validator that requires exactly one nonempty text key, capped at 65,536
  bytes. Validation precedes reservations and staged deletion; ordinary deduped
  inserts cannot admit a key too large for the deletion API. The regression
  verifies rejected input leaves no pending mutation, preserves the original
  row, and does not reserve either invalid insertion key.
- Cancellation could stop the ANN compactor's survivor iterator while the helper
  still returned success with a truncated run. The outer compactor already
  rejected publication after cancellation. The helper now checks cancellation
  before finishing its footer as well. The regression cancels during label
  writes, while complete-byte comparisons still cover all five ANN encodings.
- Optimizer retry records survived replacement of their source segments.
  Failure admission now checks current metadata under the publication lock;
  both ordinary and vector-generation replacement remove retired records under
  that lock. A late failure cannot recreate an obsolete entry. Cleanup hashes
  only retired IDs, rather than scanning the entire retry table on each merge.
  The regression reproduces a failed compaction followed by ordinary merge and
  verifies that a subsequent failure for the old ID leaves no retry record.

Failing-before evidence is retained in `.context/review2-pk-before.log`,
`.context/review2-ann-cancel-before.log`, and `.context/review2-retry-before.log`.

Additional correctness coverage includes:

- An ordinary merge paused during copying while another commit updates a row
  and deletes additional keys. The merge must carry the latest masks, retain
  all physical rows, preserve a still-pending insertion reservation, and keep
  pre-deletion and pre-merge readers valid. The fixture uses 65- and 67-row
  sources so source boundaries cross bitmap words. Subsequent explicit
  compaction preserves exactly the expected live keys and replacement value.
- An injected PK visibility-load failure after durable update/delete
  publication, followed by abort and a recovery commit. Reservations remain
  conservative, the replacement survives, and its published deletion is not
  replayed. Deleted keys become reusable after successful refresh.
- A deterministic reference-map test spanning 24 batches of mixed inserts,
  updates, deletes, aborts, ordinary merges, both compaction entry points, old
  snapshots, and reopen. It checks exact keys and versions through indexed-only
  text and numeric fast fields, including duplicate admission after Bloom reopen.
- All 128 bitmap offsets against all source lengths from 0 through 130,
  checking every output row, prior tombstones, neighboring live rows and padding.
- A 33-chunk fast column with missing values and a one-row final chunk. A 4 MiB
  budget forces cache overflow, while 16 MiB caches the entire encoded column;
  complete `.fast` bytes and every decoded value/presence bit match.
- Full deletion-file bytes compared against a scalar text-key reference scan
  over stacked dictionaries with different local ordinal orderings.

Deletion now resolves target dictionary ordinals once per segment, then uses
the existing batch column decoder to test integer membership. The target set is
bounded by the admitted key count; no corpus-sized reverse index is introduced.
Both dictionary resolution and scanning use the existing background CPU pool,
and the decoder propagates cancellation without processing the rest of a column.
This removes per-row text decoding, dictionary lookup and string hashing.

The publication lock still spans deletion preparation and persistence. Existing
searcher snapshots remain usable, but acquiring a new manager snapshot or
publishing maintenance can wait behind that commit. This pass reduces that work;
it does not claim a bounded production p99 or introduce optimistic rebase/retry
semantics. Large cold-storage, ANN-heavy and sustained-churn measurements remain
outstanding, as do x86 measurements. Defaults remain unchanged.

### Deletion commit measurements

The existing `segment_merge` benchmark now also measures
`row_deletion/commit_64_keys`: each iteration copies identical immutable RAM
index files, opens a fresh writer, initializes PK state, and stages 64 evenly
spaced keys. Only `writer.commit()` is timed, including publication and PK
visibility refresh. Setup, live/physical-count verification, and worker shutdown
remain outside timing. The saved binaries bracket the ordinal-scan change and
use the same benchmark source, Rust 1.98.1 release/default native+sync flags,
Apple M4 / 32 GiB Mac, and unset `RUSTFLAGS`/`CARGO_ENCODED_RUSTFLAGS`.
No task-owned build or test ran during measurement.

Initial 10-sample runs with 0.5-second warmup/1-second requested measurement
were noisy: 4K rows measured 389.74 → 400.86 µs, then 498.94 → 1,465.7 µs;
the latter optimized interval spanned 629–2,686 µs. The 65K case improved in
both pairs (12.625 → 4.411 ms and 15.695 → 5.549 ms). Longer runs used 1-second
warmup and 3-second requested measurement, retaining 10 samples and alternating
binary order between sizes:

| Physical rows / deleted keys | String scan | Ordinal scan | Time reduction | Peak process RSS, before → after |
| ---------------------------- | ----------: | -----------: | -------------: | -------------------------------: |
| 4,096 / 64                   |   339.08 µs |    278.97 µs |          17.7% |              123.25 → 126.33 MiB |
| 65,536 / 64                  |   10.835 ms |    3.0113 ms |          72.2% |              123.70 → 130.47 MiB |

Long-run timing intervals were 337.98–340.57 / 277.19–280.23 µs and
10.799–10.890 / 2.9817–3.0421 ms respectively. RSS includes all benchmark
fixtures, setup, worker pools, and allocator retention, so these figures do not
isolate the deletion target set. The integer set adds bounded temporary storage;
the optimization is primarily a CPU improvement. Short-run RSS changed in the
opposite direction (130.81 → 126.73 and 141.78 → 135.39 MiB), reinforcing the
need to avoid inferring an isolated allocation delta from process peaks.

Raw logs are `.context/review2-deletion-{before,after}.log`, their `-repeat`
variants, and `review2-deletion-{small,large}-{before,after}.log`.
`.context/review2-deletion-evidence.json` records fixtures, environment, binary
hashes and timing/RSS values. Criterion's automatic cross-run comparison lines
refer to its last result, not necessarily the intended pair; the table uses
the named binaries' recorded point estimates. These figures do not establish
production tail latency or cold-storage throughput.

### Validation of this pass

- `python3 scripts/check_search.py check` passed in
  `.context/search-harness/20260907T112132.747688Z-check/`.
- Final `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all
  eight stages in `20260907T113345.954895Z-full/`, including 1,397 core tests,
  80 server tests, broker/tool tests, native async and portable compilation,
  documentation, and three real-server E2E tests. Serial test scheduling avoids
  the previously observed broker port-probe race; concurrency regressions still
  exercise their own concurrent tasks and multithread runtimes.
- The new sequence test initially used a stable segment ID as an array index;
  this test-fixture error was corrected to use the searcher's segment map before
  the successful final run. The earlier full failure is recorded in
  `20260907T112643.811977Z-full/`.
- Native without sync passed all 18 selected deletion/visibility tests,
  including the mixed sequence and scalar-versus-batch mask-byte comparison
  (`.context/review2-native-async-final.log`). The shared-code WASM rebuild and
  all 13 JavaScript tests passed (`review2-wasm-{build,tests}.log`).
- Documentation/link checks passed (76 files, 281 links, 20 benchmark targets),
  and `git diff --check` passed. Client/protocol files were unchanged in this
  pass; their earlier validation is recorded above.

### Mutation surfaces and portable writer review (September 7, 2026)

Delete/upsert now reach every writable surface: native core/CLI, IndexService,
broker, Python/TypeScript clients, and WASM LocalIndex. The additive RPCs preserve
explicit commit and whole-document/chunk semantics. The server holds the existing
exclusive writer guard during staged mutations; four shared admission permits
bound conversion and staging. Started blocking workers own their permit/guard
through cancellation. Envelope limits are shared by the server and broker from
`hermes-proto/mutations.rs`: 100,000 deletion keys / 8 MiB key bytes and 1,000
replacement documents / 32 MiB encoded bytes. Broker partition error mapping
validates total accounting, unique error positions, and bounds; it never invents
successful operations from an incomplete backend response. Cross-shard publication
remains non-atomic and mutations are not automatically retried.

Portable writes reuse the primary-key reservation/Bloom implementation, global
ordinal batch scanner, and mask encoder. Portable reopen builds its Bloom from
fast dictionaries; it does not yet persist a Bloom cache. Memory includes the
existing fast readers, Bloom bits (10 bits/key plus 100,000-key headroom), pending
key reservations, and dirty-segment live-key bitmaps. RAM data bytes remain shared
with the directory. Mask matching retains the bounded key-ordinal set and one
physical-row bitset. There is no corpus-sized reverse key map.

Review found quadratic metadata scans when refreshing many segment visibilities
or replacing many cached PK readers. Refresh now iterates visibility identities
once, looks up membership in metadata maps, and replaces readers with set
membership. Portable output protection/retirement also computes an ID set once
per operation. These sets are temporary and proportional to metadata, not rows.
Prepared live-key bitmaps are not decoded twice during portable initialization.

A failed/cancelled portable builder poisons the pending transaction until abort,
preventing an upsert from committing only its deletion. Metadata-save cancellation
is reconciled against the durable generation before cleanup or replay. All mask
and segment output IDs are claimed before writes. Portable open reclaims known
unreferenced segment artifacts; storage synchronization tracks attempted file
writes so retries can remove orphan files after successful metadata publication.
LocalIndex storage requires atomic per-file replacement and one writable instance
per namespace. RemoteIndex/IpfsIndex remain readers. Replacement batches preserve
the existing JS/serde conversion semantics and count JSON bytes without a second
serialized payload buffer.

New coverage includes RPC pre-admission size checks, stable partial-error positions,
concurrent same-key replacement, cancellation while awaiting the writer, two-server
partition routing and indexed-only chunks, malformed backend accounting, client
forwarding and single-item failures, portable build failure/cancellation, both
sides of metadata rename, storage failure before/after metadata replacement,
abort, reopen, and Bloom key reuse. Validation logs and measured evidence follow below.

The TypeScript transport test also reproduced loss of batch error details at
its default 4 MiB receive cap: 100,000 rejected deletions produced a 7,883,486-byte
response. TypeScript now uses the same bounded 50 MiB send/receive caps as Python.
The real gRPC regression receives every error and preserves index 99,999.

Validation for the mutation-surface extension:

- `python3 scripts/check_search.py check` passed all four stages in
  `.context/search-harness/20260907T121315.750336Z-check/`.
- `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all eight
  stages in `.context/search-harness/20260907T121523.104566Z-full/`: 1,397 core
  tests, 82 server tests, 57 broker unit tests, 13 broker integration tests,
  tool tests and four real-server broker tests. The expanded single/partitioned
  mutation E2E suite passed again in `.context/mutations-broker-e2e-final.log`.
- After the metadata-scan optimization, 52 deletion/PK tests passed in each of
  native sync and native async builds (`mutations-native-final-tests.log` and
  `mutations-native-async-tests.log`). These include complete deletion-mask byte
  comparisons, old snapshots, merge/compaction, and cancellation regressions.
- The portable writer's two fault-injection integration tests passed with
  `cargo test -p hermes-core --no-default-features --features wasm --test portable_mutations`.
  They exercise failed/cancelled builds and failed/cancelled metadata rename.
- WASM release build and all 19 Vitest tests passed. Python's 13 unit tests and
  TypeScript's 14 tests passed, including the real gRPC maximum-deletion-batch
  transport regression. Bindings were regenerated through repository scripts.
- Native all-target Clippy and portable `--features wasm --lib` Clippy passed
  with `-D warnings`. Portable Clippy also exposed two existing conditional/
  nested-control-flow warnings; those small no-op control-flow cases were fixed.
  Documentation contracts, formatting and `git diff --check` passed.

The shared-core refactor was measured against the saved pre-extension ordinal-scan
binary using unchanged `row_deletion/commit_64_keys` fixtures, the same release
compiler/flags/machine, 10 samples, 1-second warmup and 3-second requested
measurement. No task-owned builds/tests ran during these measurements. Small and
large fixtures alternated binary order; peak RSS includes fixture setup and worker
pools. Host process activity and binary SHA-256 values are captured in
`.context/mutations-perf-evidence.json`.

| Physical rows / deleted keys |          Before extension |           After extension | Peak process RSS, before → after |
| ---------------------------- | ------------------------: | ------------------------: | -------------------------------: |
| 4,096 / 64                   | 593.14 µs (572.49–618.69) | 635.18 µs (582.17–670.62) |              125.92 → 114.69 MiB |
| 65,536 / 64                  | 6.2745 ms (5.7011–6.8714) | 5.7149 ms (5.4895–6.2066) |              121.84 → 123.56 MiB |

The timing intervals overlap in both cases. These runs do not establish a
speedup or a regression; earlier measurements in this document came from a
different period of host load and should not be used as the before value for
this change. Memory peaks also include allocator retention. The removal of
quadratic segment-metadata scans is an algorithmic improvement; this single-
segment fixture does not measure that scaling benefit. No scheduling or
compaction defaults were changed based on these measurements.

### Canonical upsert API naming

The unreleased replacement API is named `upsert` across native/portable writers,
CLI, gRPC server/broker, Python, TypeScript, and WASM LocalIndex. RPC bindings are
regenerated from `UpsertDocuments(UpsertDocumentsRequest)`. Error messages and
usage examples use the same terminology. This is a naming change: insertion of
missing keys, complete replacement of existing documents/chunks, commit semantics,
limits and encoded storage are unchanged. No performance algorithm or defaults
changed; the preceding measured evidence still applies.

Validation after the rename:

- `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all eight
  stages, including the four `check` stages and four real-server broker tests.
  Evidence: `.context/search-harness/20260907T125925.873829Z-full/`.
- Regenerated Python and TypeScript bindings; all 13 Python and 14 TypeScript
  tests passed, including the real gRPC transport regression.
- Both portable writer fault-injection tests passed. The WASM release build
  and all 19 Vitest tests passed with the generated `upsertDocument` and
  `upsertDocuments` exports.
- A CLI smoke test invoked `upsert --help`, inserted a missing key, replaced
  that key, and verified one live document with only its replacement searchable.
  Logs: `.context/upsert-cli-smoke.log`, `.context/upsert-portable-tests.log`,
  `.context/upsert-wasm-tests.log`, `.context/upsert-python-tests.log`, and
  `.context/upsert-ts-tests.log`.
- Source/binding scans found no remaining old replacement API identifiers;
  formatting and `git diff --check` passed. No new correctness or performance
  findings remain from this naming pass.

## Hybrid fast-only text filters — 2026-09-07

The reported `type = journal-article` failure is confirmed on 1.8.132. A read-only
probe of a document shard returned three hits for a standalone type filter,
three for unfiltered fusion, and zero for the same fusion with the type filter;
none of the responses was truncated. The production field is
`field type: text<raw_ci> [fast]`: it has a fast column and no inverted postings.
The probe used the public word `quantum` against `short_document`, top three,
and a 1.5-second server budget. These three live samples establish the mismatch,
not production latency percentiles. Local evidence is retained in
`.context/type-filter-production-{metadata,evidence}.json`.

Server fusion converts common filters once and wraps every nomination branch
in core `FilteredQuery`, including RRF, formula ranking, candidate export and
feature collection. That wrapper prefers a complete document bitmap. Previously,
`TermQuery::as_doc_bitset` treated absent inverted postings as an empty bitmap,
while the ordinary term scorer matched the fast column. This emptied every
branch before scoring, independent of embedding availability or the hyphen in
the type value. The new behavior-named regression failed on that exact mismatch
before the fix (`.context/type-filter-red.log`).

Core now materializes fast-only equality on native/sync and portable execution,
preserving global text ordinals, missing/empty values, and existing first-value
semantics for multi-value fast columns. Indexed terms keep posting-based
membership. Single-value materialization reuses the existing batch decoder,
adding caller-controlled early exit and 2 KiB of stack scratch; it does not
allocate a document-sized candidate heap or change persisted bytes. The existing
16 MiB bitmap and 64-filter limits still apply. Tests cover a 200,001-document
segment above the generic scorer-fallback cap, positive/negative Boolean filters,
date/type combinations, unchanged scores/ordinals across fusion modes, two-shard
broker requests and WASM branch filters.

Deadline-aware scans discard incomplete bitmaps. The ordinary fast-text scorer
also checks the deadline while skipping nonmatching values. Bitmap enumeration
captures each observed document ID once: a second `doc()` call could otherwise
turn into `TERMINATED` between the check and the bitmap write. A deterministic
regression covers that cancellation boundary, and batch-reader tests cover early
exit inside/across batches and remapped text dictionaries across merged blocks.

The first large-fixture attempt exposed repeated blockwise-linear header decoding
when materialization reused scalar fast-field reads. It was stopped after a CPU
sample identified that path, then single-value materialization was changed to
batch reads. The related general limitation remains: ordinary fast-only term
scans and multi-value first-value scans still use the existing scalar decoder;
their codec-dependent cost is not a linear-time guarantee. This change makes no
new performance claim for those paths and changes no compression/default policy.

Warm local RPC samples used the same persisted 10,000-document fixture, Apple M4,
Rust 1.98.1, default debug server build flags and 20 warmups plus 100 sequential
requests per mode. The first 9,950 documents are books; the last 50 are journal
articles. Two text branches request top three. The saved pre-fix server was built
at `7076b38d`; the intervening `7a4b380c` change only bumps release versions.
The fixture metadata SHA-256 stayed
`b3f3f04e9abf73e4f78cbf646f0d72eb37de19a2230e314fb69b26921a395c43`.

| Request                           |                  Before p50 / p95 | After p50 / p95 | Correctness                     |
| --------------------------------- | --------------------------------: | --------------: | ------------------------------- |
| Standalone fast-only type filter  |                    6.69 / 9.00 ms |  2.90 / 3.66 ms | Same IDs and score bits         |
| Unfiltered hybrid                 |                    6.01 / 6.99 ms |  2.78 / 3.12 ms | Same IDs and score bits         |
| Hybrid with indexed-type control  |                    5.18 / 6.38 ms |  2.55 / 2.93 ms | Same IDs and score bits         |
| Hybrid with fast-only type filter | 1.48 / 1.67 ms, incorrectly empty |  3.08 / 3.72 ms | Matches indexed control exactly |

Maximum sampled server RSS across the four modes was 40,912 KiB before and
37,520 KiB after; this includes mapped pages and heap, not a separate heap profile.
No builds from this workspace ran during sampling, but the shared host was not
controlled. Unchanged controls also became faster, so these before/after times
are not evidence of a general speedup. The corrected fast-only filter cost about
0.54 ms more than the indexed control in the after run. A fast-only field still
requires reading its column; indexing that field and rebuilding existing data
would remove that scan. This fix does not add a request cache: branches retain
the existing filter ownership and bounded memory lifetime. Script and raw samples:
`.context/measure_type_filter.py`, `.context/type-filter-{before,after}-samples.jsonl`.

All eight full-harness steps passed with `RUST_TEST_THREADS=1` and the isolated
native build directory: `.context/search-harness/20260907T135854.914377Z-full/`.
This includes 1,373 core unit tests, server/broker/tool checks, native-without-sync
and portable compilation, documentation and all three real-server broker tests.
The one-thread setting avoids the previously recorded broker discovery flake;
no production timeout or concurrency setting changed. Thirteen targeted common
filter tests also passed with native async execution, and the WASM release build
and all 13 runtime tests passed. Logs:
`.context/type-filter-{full-release,native-async,wasm-build,wasm-tests}.log`.

The earlier full run caught an invalid new RRF test request: `candidate_depth`
is only accepted for L1/exports. Correcting that test fixture in server and broker
tests made the regression and final full run pass. Its initial failure remains
in `.context/search-harness/20260907T133822.951694Z-full/`. An exploratory run
was intentionally stopped while replacing the scalar scan; the harness logged
an interrupt-cleanup `Operation not permitted` error in
`.context/search-harness/20260907T133111.044272Z-full/`. Neither is an unresolved
failure of the final checks. The earlier native-async phrase-ordinal discrepancy
and parallel broker discovery flake remain separate recorded findings.

## 2026-09-09: measuring retained segment candidates for L1

This is a design experiment, not a serving change. Retaining already-produced
segment candidates improves formula top-K recall on mixed segments, but it can
add substantial L1 work without improving recall on clustered segments. The
measurements do not support enabling it universally.

### Method and invariant

The isolated checkout at `.context/segment-l1-measure-worktree` is based on
`6989e797` (1.8.134). Test-only hooks observe completed segment result lists in
the existing searcher, **after** its shared threshold is raised and **before**
its shard merge discards results. They do not alter segment depth, shared vertical
thresholds, BMP's global LSP selection, search concurrency or scoring kernels.
L1 runs after nomination; its predictions never feed retrieval thresholds.
The normal nomination lists remain included, with addresses, scores and passage
rows checked against the captured copies on every extra-candidate request.

Each branch keeps a bounded heap ordered by its existing raw score and canonical
address tie-break. With normal depth `d` and extra allowance `e`, this heap keeps
at most `d + e` candidates from completed segment results. It does not capture
all visited documents or guarantee an independent per-segment top-d. A candidate
pruned inside segment execution remains unavailable. This is one specific extra
selection policy; the experiment does not establish its optimality.

The five compared policies use three nomination branches and final top-10:

| Policy     | Retrieval depth per branch | Maximum retained slots across branches |
| ---------- | -------------------------: | -------------------------------------: |
| Baseline   |                         20 |                                     60 |
| Extras 20  |                         20 |                                    120 |
| Extras 80  |                         20 |                                    300 |
| Deeper 40  |                         40 |                                    120 |
| Deeper 100 |                        100 |                                    300 |

Retained slots include duplicates across branches, so document unions are smaller.
The deeper controls have the same maximum retained allowance, not necessarily the
same actual union or read cost. They use the core interface; an RPC caller must
also satisfy the existing result-window/candidate oversubscription limits.

All policies use the existing core candidate scorer, preserve organic scores,
backfill missing cells, and apply the formula
`bm25 + 2 * ln(1 + sparse) + 8 * dense`. The features are four-term BM25,
12-dimension BMP sparse retrieval and 128-dimensional F32 flat dense retrieval.
Document features use MAX; the passage fixture aligns three body/sparse/dense
ordinals per document and applies MAX after passage L1. Some documents have no
sparse or dense field. The formula and generator were fixed before sampling.
BMP uses its normal block/heap pruning with gamma 0; a separate capped case uses
one global gamma 2. Each case keeps that setting fixed across all five policies.

The fixed seeded corpus has 32 topics and 16,384 documents, laid out as one,
four or 16 mixed segments, 16 topic-clustered segments, or 16 uneven segments
(one holds 80% of the corpus). Additional cases cover capped BMP and three
passages per document. A 65,536-document mixed fixture checks scale. A one-search-
thread repeat uses the **same persisted files** as the four-thread mixed fixture.
Fixture file hashes are retained with the raw results.

Exhaustive core feature scoring over every fixture document (every stored passage
for the passage case) defines the reference top-10. **Recall below is agreement
with that formula reference, not human relevance or teacher-labelled recall.**
There were no representative production requests or relevance labels available
in the workspace. The generator is synthetic; the binary IVF fields used by the
live installation are not represented by this F32 flat dense fixture.

### Results

Every cell below is **formula recall@10; median core latency**. Latency includes
nomination, pool assembly and L1, with compilation, fixture creation, expression
compilation, statistics preparation, reference scoring, assertions, JSON and RPC
outside the timed region. Each policy has 32 queries repeated four times, with
rotating policy order. All sampling uses the same release binary on Apple M4
arm64, rustc 1.98.1, two Tokio workers and four search threads unless noted.
All payloads fit in memory; these are warm, sequential-request measurements.

| Fixture                                           |       Baseline |      Extras 20 |       Extras 80 |      Deeper 40 |      Deeper 100 |
| ------------------------------------------------- | -------------: | -------------: | --------------: | -------------: | --------------: |
| 16k documents, 1 segment                          | 75.0%; 0.28 ms | 75.0%; 0.28 ms |  75.0%; 0.28 ms | 94.7%; 0.34 ms | 100.0%; 0.47 ms |
| 16k documents, 4 mixed segments                   | 75.0%; 0.28 ms | 94.7%; 0.32 ms |  99.7%; 0.40 ms | 94.7%; 0.35 ms | 100.0%; 0.50 ms |
| 16k documents, 16 mixed segments                  | 75.0%; 0.50 ms | 94.7%; 0.59 ms | 100.0%; 0.74 ms | 94.7%; 0.64 ms | 100.0%; 0.86 ms |
| 16k documents, 16 clustered segments              | 75.0%; 0.24 ms | 75.0%; 0.53 ms |  75.0%; 0.73 ms | 94.7%; 0.31 ms | 100.0%; 0.47 ms |
| 16k documents, 16 uneven segments                 | 75.0%; 0.38 ms | 85.1%; 0.50 ms |  85.3%; 0.58 ms | 94.7%; 0.49 ms | 100.0%; 0.70 ms |
| 16k documents, 16 mixed segments, gamma 2         | 71.6%; 0.45 ms | 91.9%; 0.56 ms | 100.0%; 0.70 ms | 91.9%; 0.59 ms | 100.0%; 0.83 ms |
| 16k documents, 3 passages, 16 segments            | 56.2%; 0.86 ms | 86.2%; 1.00 ms | 100.0%; 1.30 ms | 86.2%; 1.06 ms |  99.7%; 1.61 ms |
| 65k documents, 16 mixed segments                  | 48.4%; 0.99 ms | 71.2%; 1.07 ms |  98.1%; 1.32 ms | 71.2%; 1.19 ms |  98.1%; 1.55 ms |
| 16k documents, 16 mixed segments, 1 search thread | 75.0%; 0.75 ms | 94.7%; 0.83 ms | 100.0%; 0.95 ms | 94.7%; 0.96 ms | 100.0%; 1.31 ms |

For 65k documents, Extras 80 retained about 286 documents versus 60 in the
baseline and supplied 5.16 previously unnominated documents to the final top-10
on average. Its p95 was 1.77 ms versus 1.29 ms baseline and 1.90 ms Deeper 100.
On 16k mixed segments, Extras 20 supplied 2.16 new top-10 documents per request,
and Extras 80 supplied 2.50; 30 of the 32 queries benefited from extra winners.
On clustered segments both supplied **zero** new winning documents despite
retaining about 116 and 281 documents. Their respective median L1 times grew
from 0.06 ms baseline to 0.34 and 0.52 ms. Captured candidates from other segments
could not recover the stronger candidates discarded inside the topic's segment.

The uneven case gained approximately ten recall points from Extras 20, but
Deeper 40 gained approximately twenty points at a similar median cost. Raising
the extra allowance to 80 barely improved its recall. More retained rows alone
are therefore a poor reason to spend additional backfill budget. Segment merging
and corpus layout can materially change the benefit.

### Pruning, work and memory

Normal nomination document identities and order were unchanged in the paired
extra/baseline measurements. Native text scores sometimes varied by one float
ULP across repeated executions; the captured row was always byte-identical to
its originating normal row within that execution. Scoring schedules can vary
under the shared threshold without changing its policy.

BMP blocks scored per request confirm that extras did not obtain their gains by
requesting deeper BMP work: approximately 221 blocks for both baseline and
Extras 80 on the 16k mixed fixture, versus 490 for Deeper 100; approximately 377
versus 377 versus 1,132 on the 65k fixture. Small run-to-run block-count variation
is retained in the samples; copying captured results can affect scheduling even
though thresholds are published before capture. There is no claim that the
instruction trace or elapsed nomination time is identical.

Separate untimed allocator passes measured requested allocation bytes and the
maximum increase in live heap from request start. They exclude assertion/report
allocations and include temporary search/scoring buffers. These are not absolute
heap residency, mmap residency or process RSS; thread overlap affects the peak.
For the 16k mixed fixture:

| Policy     | Extra capture buffers | Median allocated bytes per request | Largest observed heap increase | Backfilled component values | Charged exact-vector / BMP bytes |
| ---------- | --------------------: | ---------------------------------: | -----------------------------: | --------------------------: | -------------------------------: |
| Baseline   |                 0 KiB |                          1.324 MiB |                      177.6 KiB |                         223 |                   18.3 / 8.4 KiB |
| Extras 20  |               8.8 KiB |                          1.413 MiB |                      193.3 KiB |                         424 |                  34.6 / 16.0 KiB |
| Extras 80  |              21.9 KiB |                          1.570 MiB |                      198.1 KiB |                         869 |                  70.7 / 32.6 KiB |
| Deeper 100 |                 0 KiB |                          2.190 MiB |                      205.6 KiB |                         869 |                  70.7 / 32.6 KiB |

Capture-buffer figures include the bounded heap's normal candidates as well as
its extras. Component counts include each query component, not just one value
per feature. Charged reads are the scorer's vector/BMP admission accounting;
text mmap probes and OS page faults are not represented by those byte totals.
The complete initial process peaked at 281.6 MiB RSS including fixture building
and exhaustive references; it cannot be attributed to any one policy. No builds
from this workspace ran during timed sampling. Other activity on the shared Mac
was uncontrolled, so small latency differences should not be generalized.

### Evidence and limits

There are 5,760 timed requests and 360 separate memory requests across the nine
fixture/thread cases. Raw rows include normal nomination signatures, pool size,
new winners, formula/pool recall, phase timings, BMP work and feature-read
accounting. Artifacts:

- `.context/segment-l1-measure-results/`: the seven 16k cases, environment,
  fixture hashes, raw JSONL, and CSV/Markdown/JSON summaries.
- `.context/segment-l1-measure-large/`: the 65k mixed case.
- `.context/segment-l1-measure-single-thread/`: one-thread repeat over the same
  16k mixed index files.
- `.context/analyze_segment_l1.py`: summary calculations.
- `.context/segment-l1-measurement.patch`: test-only instrumentation and harness
  for a detached checkout of `6989e797`; no public API is added.
- `.context/segment-l1-measure-{build,run,large,single-thread}.log`: commands'
  build/execution logs. Memory instrumentation is disabled during timed passes.

The ignored unit harness is invoked as follows, after applying the experiment
patch to its detached checkout:

```sh
CARGO_TARGET_DIR="$PWD/.context/segment-l1-measure-build" CARGO_BUILD_JOBS=4 \
  cargo test --locked --release \
  --manifest-path .context/segment-l1-measure-worktree/Cargo.toml \
  -p hermes-core --lib segment_nomination_measurement::measure --no-run
SEGMENT_MEASURE_OUTPUT="$PWD/.context/segment-l1-measure-results" \
  SEGMENT_MEASURE_DOCS=16384 SEGMENT_MEASURE_QUERIES=32 \
  SEGMENT_MEASURE_REPEATS=4 SEGMENT_MEASURE_THREADS=4 \
  <test-binary> --exact segment_nomination_measurement::measure \
  --ignored --nocapture --test-threads=1
python3 .context/analyze_segment_l1.py
```

`SEGMENT_MEASURE_FIXTURE=doc_s16` selects the scale/thread repeats; their output
roots must differ. Use `SEGMENT_MEASURE_DOCS=65536` for the scale repeat and
`SEGMENT_MEASURE_THREADS=1` with the shared persisted fixture for the thread
repeat. Correctness smoke runs use 512 documents and two queries; their timings
are not included in the performance summaries.

Read-only live metadata collection found one existing shard with 40 segments,
26,291,811 physical documents, segment sizes from 557 to 4,187,252 and median
4,724. The smallest 24 segments contain 67,071 documents. This motivated the
uneven fixture; it is not a measurement of live recall or latency. No production
query workload, process, deployment, index or schema was changed.

Production relevance, x86 performance, binary IVF, cold payloads, concurrent
requests/ingestion, RRF-dependent formulas and phrase-feature costs remain
unmeasured. In particular, formulas containing RRF still require ranks at the
correct global scope before final selection. The experiment supports a bounded
opt-in trial on representative queries; it does not justify a default change or
predict a production speedup.

The experiment's 512-document smoke cases passed with both native sync and native
without sync; their logs are `.context/segment-l1-measure-smoke.log` and
`.context/segment-l1-measure-async-smoke.log`. The repository's required
`python3 scripts/check_search.py check` passed with `RUST_TEST_THREADS=1`, including
formatting, focused Clippy, core/server/broker/tool tests and native-async
compilation. Evidence is in
`.context/search-harness/20260909T052512.176694Z-check/`. Full RPC/WASM checks were
not rerun: the main checkout changes only this report, and the instrumentation
exists only in the isolated experiment checkout under `cfg(test)`.

`python3 .context/run_segment_l1.py --output <new-directory>` automates creating the
detached checkout, applying the patch, building, measuring and summarizing. The
source, raw measurements, summaries and instructions are also packaged in
`.context/segment-l1-measurement.tar.gz`; fixture index files are regenerated,
not included in the archive.

## 2026-09-09: feature reads and scoring review

This subsection records the initial design and measurement pass, before the
implementation follow-up below.
The invariant is identical candidates, organic scores, raw missing/zero values,
passage ordinals, formula outputs and request-wide work/read budgets. The cost
model separates segment/feature setup and scheduling, selected payload reads,
scoring kernels, feature assembly and formula evaluation.

### Measured opportunity: amortize CPU scheduling

The native server awaits `score_candidates_with_retrieved_and_rrf` directly in
`hermes-server/src/search_service.rs`. Core groups candidates by segment and
processes those groups sequentially in
`hermes-core/src/query/candidate_scoring/execution.rs`. Each BMP component and
each dense/binary scoring batch independently enters `install_search_cpu`.
Consequently, a small pool spread across 16 segments can incur approximately
30 separate synchronous CPU-pool handoffs. The text probes, feature assembly
and formula loop run on the calling thread. This differs from L0's coarser
native search dispatch.

The experiment freezes each candidate pool and its organic branch lists, then
compares the existing L1 call with the **same call** entered once through the
shared search CPU pool. Nested calls then stay on that worker. Both paths still
process segments sequentially; no parallel segment scorer, scoring algorithm,
candidate expansion or pruning change is involved. The experiment polls the
mmap future with `now_or_never()` and asserts that every read is immediately
ready. It does not block arbitrary async I/O on a Rayon worker. A serving design
would need bounded async admission/completion around core CPU work, preserving
lazy-directory, current-thread runtime, cancellation and WASM behavior.

Median L1 latency, including plan validation, feature backfill, prediction and
sorting, but excluding retrieval, expression parsing, statistics and RPC:

| Fixture / branch depth              | Mean pool | Current call | One worker entry | Median reduction |
| ----------------------------------- | --------: | -----------: | ---------------: | ---------------: |
| 1 segment / 20                      |      57.6 |      69.2 us |          64.4 us |               7% |
| 16 mixed segments / 20              |      57.6 |     320.9 us |         123.7 us |              61% |
| 16 mixed segments / 100             |     246.8 |     520.8 us |         263.8 us |              49% |
| 16 uneven segments / 20             |      57.6 |     180.4 us |          91.6 us |              49% |
| 16 clustered segments / 20          |      57.6 |      60.8 us |          56.9 us |               7% |
| 16 segments, passage features / 20  |      59.2 |     402.3 us |         179.9 us |              55% |
| 16 segments, passage features / 100 |     281.1 |     768.8 us |         502.5 us |              35% |

For the mixed 16-segment case, p95 changed from 478 to 154 us at depth 20,
and from 893 to 307 us at depth 100. The one-segment/clustered cases supply most
nominees from one segment, so consolidating dispatch has little to amortize;
their small median changes are inconclusive and their p95 did not improve.

A separate repeat with **one** search-pool thread over the same mixed index
changed medians from 242 to 115 us and from 403 to 243 us respectively. The gain
therefore does not require scoring segments in parallel. Worker locality and
execution scheduling both change; the entire latency difference should not be
described as a measurement of queue wait alone.

The native-without-sync control also passed the same byte comparisons. In that
build `install_search_cpu` is already inline, so both invocation styles should
perform the same work. Measured medians were 105.4/105.7 us at depth 20 and
233.8/235.3 us at depth 100, showing no useful difference. This is a control
within that build, not a recommendation to disable native search parallelism.

Separate instrumented passes explain the difference. For the small mixed pool,
BMP/vector dispatch spans totalled about 218 us per request, while their nested
kernel spans totalled about 21 us. With one worker entry those figures were
about 11.6 and 10.5 us. These spans overlap, contain instrumentation overhead,
and come from eight queries; they are attribution evidence, not an additive
latency breakdown or replacements for the unprofiled timing samples.

Every comparison checks the complete serialized results and feature rows,
including ordering, scores, missing values and passage ordinals, byte for byte.
Backfilled component counts and charged bytes also match: approximately
223 components, 18.25 KiB of exact vectors and 8.40 KiB of BMP payload at depth
20; 869 components, 70.72 KiB and 32.64 KiB at depth 100. No candidates are
discarded to obtain the improvement.

Memory is essentially unchanged by scheduling. The mixed depth-100 call
allocated a median 373.9 KiB through 3,407 allocations in either mode, with a
largest observed live-heap increase of 67.6 KiB. This shows a separate opportunity
to reduce temporary allocations. These numbers include only L1 and must not be
compared directly with the previous section's whole-search allocation totals.

### Further findings, in suggested implementation order

1. **Reduce text cursor setup and reuse bounded scratch.**
   `term::score_term_candidates` creates a posting iterator for every component
   in every occupied segment. `BlockPostingIterator::owned` allocates two
   128-element `u32` buffers and decodes block zero before the first target seek,
   even when the first candidate belongs to a later block. It also maintains
   position-frequency prefixes although ordinary BM25 probes need no positions.
   In the one-worker profile, text backfill took about 58 us for the small mixed
   pool and 71 us for the large one, versus approximately 10/21 us for BMP
   and less than 1 us for the dense arithmetic. This makes text setup/probing a
   more promising next target than changing the formula evaluator. A targeted
   first seek and reusable decode buffers belong in the existing posting reader;
   retain positional cursor semantics for phrase consumers. The fixture does
   not isolate how much time any one of these changes would save.

2. **Reduce feature assembly allocations.**
   `execution.rs` builds nested document feature vectors, tree maps for passage
   rows, and another document-to-location tree for every document-scope feature.
   `DocumentExpression::score` then allocates a `(ordinal, score)` vector for each
   component/document reduction. Chunk-field sets are rebuilt per segment and
   again per output document; model-bearing results clone old positions before
   replacing them. Reuse admitted scratch, hoist the field set, and evaluate
   contiguous row/location spans through the existing combiner. Preserve strict
   ordinal reduction order, missing versus zero, negative boosts, and the
   difference between `MAX(a) + MAX(b)` and `MAX(a + b)`. Flat internal storage
   should still produce the same owned export rows at the boundary.

3. **Prepare each scoring query once.**
   L0 already shares `PreparedBmpQuery` across segment scorers. L1's
   `score_bmp_candidates` rebuilds its quantized/sorted vectors, candidate mask
   and phase-one metadata for every segment/component, even though point scoring
   uses no LSP or block-pruning plan. `score_vector_candidates` similarly
   recomputes the query norm and F16 representation on every segment call;
   the F16 copy is unused by F32/UInt8 kernels. Reuse immutable preparation in
   the existing core plan, keeping segment-specific BMP scale and field/dimension
   validation at execution. In the one-worker mixed depth-100 profile, BMP
   preparation totalled about 4.5 us; vector preparation **including its output
   and raw-buffer allocations** totalled about 6.4 us. This is a smaller follow-up
   than scheduling or text work on this fixture, not a measured speedup yet.

4. **Share payload reads when several branches use the same field.**
   Each feature/component currently resolves locations and reads its selected
   vectors independently. Two dense queries over the same body field can read
   and copy the same flat rows twice; multiple sparse queries can validate and
   traverse the same forward vector repeatedly. A bounded field/segment batch
   could resolve the union of missing logical cells, read a row once, and apply
   the existing kernels for each required query before scattering scores back
   to branch slots. Preserve organic values even when another branch needs the
   same row, and charge both distinct payload bytes and component work. This
   needs a repeated-field/multi-query fixture: the measured three-field workload
   does not quantify its benefit.

5. **Treat lazy text reads and MaxScore presence discovery separately.**
   `reserve_candidate_text_reads` looks up term metadata before lazy reads;
   `get_postings`/`get_positions` then look it up again and request the complete
   term range. Mmap returns byte views, but a lazy backend can materialize those
   ranges for very few candidates. Reader-owned prepared term handles, selected
   block reads, and bounded I/O batching deserve a cold/remote benchmark.
   Separately, `maxscore_candidate_locations` invokes
   `SparseIndex::probe_candidates(..., None, ...)`, which probes **all retained
   dimensions** to discover field presence and ordinals; scoring then probes the
   query dimensions. Reuse discovery across compatible branches before
   considering a format change. Looking only at query dimensions would turn
   present-but-zero values into missing values and change formulas with missing
   defaults. The current production metadata uses BMP, and the local benchmark
   also uses BMP, so this is a conditional finding rather than its measured cost.

The dense path already sorts physical targets and coalesces adjacent flat-vector
reads. BMP already has an evictable forward representation and validates only
selected payloads. Those are useful existing mechanisms to build on. A
short-circuit in the BMP arithmetic loop after the query dimensions are exhausted
could avoid some work, but `BmpForward::vector` still validates the complete
selected vector first. Do not remove that validation or pin every vector to
make warm measurements look better. Gapped-read coalescing and prefetch should
be evaluated against bytes touched and cold page faults as well as elapsed time.

### Formula cost and correctness constraints

The compiled formula `bm25 + 2 * ln(1 + sparse) + 8 * dense` costs about
1.7 us per small document pool and 7.3 us per large document pool in a replay of
the existing model scorer, with **zero allocations** in those document-only
passes. Passage merge/reduction raises those figures to 5.9 and 28.6 us and
allocates scratch in `model.rs`. Replacing the expression package is therefore
low priority for these requests. Longer formulas, RRF contribution processing,
and documents with many passages need their own measurements.

L1 still backfills branches that are absent from the formula. An inference-only
request could potentially use a formula-dependency mask, but a mask alone would
break raw exports and broker RRF: the broker deliberately asks shards to run a
constant formula while exporting every raw branch. Passage nomination must also
remain intact. Treat pruning unused feature reads as an explicit execution-plan
optimization with export requirements, not a change to formula or missing-value
semantics.

Scoring more segment candidates should follow this cleanup, then be remeasured.
These results demonstrate savings without changing candidate recall; they do
not establish that a larger pool is free. Additional missing features still
require payload access, and the extra pool must remain bounded. Per-segment
parallelism is another proposal, not part of this experiment: it must share the
request's work/read admission and CPU capacity, account for simultaneous scratch,
and be tested under concurrent queries before a throughput claim.

### Reproduction and validation

Evidence is under `.context/feature-scoring-review/`, based on `6989e797`:

- `measurement.patch`: the complete test-only patch for a detached checkout.
  `original-segment-measurement.patch` retains the preceding experiment.
- `results/`: five fixture layouts, raw JSONL and summaries.
- `single-thread/`: the same mixed fixture with one search-pool thread.
- `async-control/`: native-without-sync byte-equivalence and scheduling control.
- `analyze.py`: summary calculations and paired work/byte-count audit.
- `build.log`, `run.log`, `single-thread.log`, `focused-tests.log`.

The experiment reuses the preceding section's 16,384-document mmap indexes,
compiler, release profile and machine (Rust 1.98.1, Apple M4/arm64). It uses
32 fixed-seed queries, depths 20/100, 12 repetitions and rotating variant order,
with allocation and phase instrumentation disabled during latency measurements.
There are **9,216 timed L1 calls**, 192 separate phase samples and 192 separate
allocation samples across the four-thread and one-thread runs. Formula replay
is separate: 384 timing estimates of 64 full-pool evaluations and 384 allocation
passes. It includes passage/context reduction and score checks, but excludes
copying the input rows. The native-without-sync control adds 512 timed L1 calls,
32 phase samples, 32 allocation samples, and 64 formula timing/allocation pairs;
its timings are kept separate. No builds from this workspace ran during timed sampling;
other activity on the shared Mac was uncontrolled.

After applying the patch and building the same optimized core unit-test binary:

```sh
FEATURE_MEASURE_DATA="$PWD/.context/segment-l1-measure-results" \
  FEATURE_MEASURE_OUTPUT="$PWD/.context/feature-scoring-review/results" \
  FEATURE_MEASURE_REPEATS=12 FEATURE_MEASURE_THREADS=4 \
  <test-binary> --exact \
  segment_nomination_measurement::feature_reads::measure_features \
  --ignored --nocapture --test-threads=1
python3 .context/feature-scoring-review/analyze.py \
  .context/feature-scoring-review/results
```

Set `FEATURE_MEASURE_THREADS=1`, `FEATURE_MEASURE_FIXTURE=doc_s16`, and a new output
directory for the control. The preceding section's runner regenerates the
fixture files when needed. Optimized focused candidate-scoring tests passed
in both native builds (13 passed, one existing manual benchmark ignored in each).
The main checkout only
changes this report; the earlier `check` result at the same source revision
remains applicable. RPC, WASM, x86, cold payloads, production binary vectors and
concurrent-query throughput are not validated by these measurements. No serving
change, publication or production mutation was made for this review.

The source, samples, analyzer and standalone reproduction instructions are
packaged in `.context/feature-scoring-review.tar.gz`. Fixture indexes and compiled
binaries are excluded; the included patch regenerates the fixtures. The report
passed Prettier, `git diff --check`, and the repository's documentation/ownership
contract check for this review.

## 2026-09-09: implemented feature-read and scoring improvements

The follow-up implements the measured scheduling improvement and reduces
repeated preparation and temporary allocations. It does **not** enable the
earlier retained-segment-candidate proposal. Candidate nomination, pruning,
scores, raw exports, formulas, storage/wire formats and request limits retain
their existing semantics.

### Implementation and ownership

- `Searcher::run_search_cpu` polls the existing borrowed scoring future on the
  shared search pool when called from a multithread Tokio runtime. A poll that
  encounters pending I/O returns to the original task, releasing the worker.
  The worker enters the caller's Tokio handle for async directory operations.
  There is no spawned/detached scoring task or second scorer: a request drop
  drops its future after any active scoped poll returns. Native current-thread
  and non-Tokio callers, native without sync, and WASM keep their existing path.
- The existing posting reader accepts reusable decode buffers and starts
  selected probes at their first requested block. Ordinary posting iteration
  and its position-prefix semantics are retained. Request-owned text scratch
  is reused across components and segments.
- BMP query quantization is cached per immutable component, with bounded
  replacement when segment dimensions differ. Segment dequantization remains
  local. Dense query norms are cached per component, and F16 query copies are
  built only for F16 segments. These caches never outlive the scoring request;
  they contain query preparation, not corpus payloads.
- Document feature reduction groups sorted location spans in one reusable
  vector, replacing per-document trees/vectors. Small reductions use the same
  combiner with inline storage. Passage/context inference uses a fixed feature
  array and inline score scratch, validates duplicate ordinals before inference,
  and preserves ordinal order for strict float reductions. Chunk-field sets are
  prepared once, and formula results construct their new positions directly.

The request still computes all required missing features, including unused
formula branches needed for raw export and broker RRF. Overlapping payload-read
sharing, lazy text range planning, MaxScore presence-discovery changes and
per-segment parallelism remain separate proposals. No new cache or candidate
policy is enabled for these purposes.

### Before/after measurements

Both binaries use the same compiler, release flags, host and persisted fixtures
from the review: Rust 1.98.1, Apple M4/arm64, warm mmap, 16,384 documents, 32
fixed-seed queries, three branches and the same nonlinear formula. The before
binary is based on `6989e797`; the after binary contains this implementation.

The before run saves the complete organic branch lists, candidate pools and
serialized expected results/features. The after run replays those saved inputs
and compares **every output byte**, rather than allowing retrieval to select a
different candidate pool. All comparisons passed, and the paired component,
vector-byte and BMP-payload charges are identical. The candidate counts below
are means; latency cells are p50/p95 in microseconds and cover L1 only.

| Fixture / branch depth                     | Pool docs | Before p50/p95 | After p50/p95 |
| ------------------------------------------ | --------: | -------------: | ------------: |
| 1 segment / 20                             |      57.6 |       72 / 122 |       51 / 61 |
| 1 segment / 100                            |     246.8 |      205 / 267 |     155 / 192 |
| 16 mixed segments / 20                     |      57.6 |      316 / 583 |      94 / 112 |
| 16 mixed segments / 100                    |     246.8 |      518 / 989 |     200 / 236 |
| 16 clustered segments / 20                 |      57.6 |        57 / 73 |       42 / 50 |
| 16 clustered segments / 100                |     246.7 |      174 / 211 |     130 / 168 |
| 16 uneven segments / 20                    |      57.6 |      165 / 239 |       68 / 81 |
| 16 uneven segments / 100                   |     246.8 |      404 / 523 |     183 / 205 |
| 16 segments, passage features / 20         |      59.2 |      392 / 675 |     147 / 182 |
| 16 segments, passage features / 100        |     281.1 |     754 / 1105 |     427 / 507 |
| 16 mixed segments, one search thread / 20  |      57.6 |      241 / 321 |      93 / 113 |
| 16 mixed segments, one search thread / 100 |     246.8 |      400 / 497 |     196 / 227 |

Thus the mixed four-thread case improves median L1 latency by approximately
70% at depth 20 and 61% at depth 100. Comparing with the before binary's
experimental one-worker mode isolates the remaining opportunity approximately:
121/256 us before versus 94/200 us after, another approximately 22% reduction
from preparation/allocation changes. This is not a separate kernel benchmark.

Separate allocator passes show lower allocation traffic, with a small increase
in the live heap from retaining reusable scratch until request completion:

| Case                        | Allocated KiB before/after | Allocations before/after | Largest heap increase KiB before/after |
| --------------------------- | -------------------------: | -----------------------: | -------------------------------------: |
| Mixed segments, depth 20    |                  199 / 104 |               1664 / 961 |                            17.6 / 19.8 |
| Mixed segments, depth 100   |                  374 / 234 |              3407 / 1382 |                            67.6 / 70.0 |
| Passage features, depth 20  |                  216 / 133 |              2123 / 1430 |                            32.8 / 35.5 |
| Passage features, depth 100 |                  548 / 414 |              5880 / 3600 |                          154.0 / 155.9 |

Across the fixtures, the largest observed per-request live-heap increase rises
by approximately 1.9–4.3 KiB. These are requested allocation bytes and heap
changes measured around L1; they do not describe absolute heap, mmap residency
or process RSS. Query-preparation residency scales with the admitted active
components and their vector dimensions, so the table does not establish a
universal scratch increment for arbitrarily large formulas/queries.

The small passage-formula replay drops from 5.9 to 4.6 us, and the large replay
from 28.9 to 22.4 us; both now allocate no scratch for these at-most-three-passage
fixtures. Larger reductions spill to bounded heap storage; 17/65-passage tests
cover that boundary. The formula library and arithmetic are unchanged.

There are 18,432 timed calls across the two binaries and six fixture/thread
cases, with phase and allocation accounting disabled during timing. Half use
the normal caller and half retain the earlier worker-entry control. Separate
passes contain 384 phase samples and 384 allocation samples, plus formula
replays. Compilation and full validation finished before the timed after run;
no builds from this workspace overlapped either timed comparison. Activity on
the shared Mac was otherwise uncontrolled. This is warm single-request evidence,
not a production throughput, x86, cold I/O or relevance benchmark. The source
changes no architecture-sensitive scoring or retrieval defaults.

### Validation and evidence

`python3 scripts/check_search.py full` passed all eight steps, including the
entire `check` sequence, native-without-sync and portable builds, API docs and
the broker's real-server tests. The core suite passed 1,407 tests with 18 existing
manual benchmarks ignored. Evidence:
`.context/search-harness/20260909T064235.450347Z-full/`.

The new regressions cover CPU-pool resumption after pending I/O, cancellation
and panic/error propagation, current-thread runtimes, posting seek/position
equivalence with recycled buffers, spill-sized passage reductions, and distinct
queries across segments using BMP, F32, F16, UInt8 and binary vectors. Existing
organic-score, missing-value, phrase, reorder and admission tests also passed.
The WASM release build, clean npm install and all 20 WASM tests passed.
The native-without-sync candidate-scoring run passed 15 tests, with one existing
manual benchmark ignored.

Evidence under `.context/feature-scoring-implementation/` includes:

- `before/`, `after/`, `before-single-thread/`, `after-single-thread/`: raw
  JSONL and summaries; `frozen/*.json` contains the persisted before inputs and
  expected output bytes.
- `before-measurement.patch`, `after-measurement.patch`: complete experiment
  source for detached checkouts of the base revision, including the production
  change in the latter. The fixture builder belongs to the before experiment;
  the after experiment reuses those index files and frozen pools.
- `instrument_after.py`, `compare.py`, `analyze.py`, compiler/runtime logs,
  validation logs and fixture/environment hashes.

The reproduction source and measurements are bundled as
`.context/feature-scoring-implementation.tar.gz`; compiled binaries and index
files are excluded. Recreating an index assigns new segment IDs, so regenerate
its frozen pools with the before binary before running the after comparison.
