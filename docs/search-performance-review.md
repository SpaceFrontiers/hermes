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
  Core and broker share the same formula. The contract is `linear_v2` /
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
