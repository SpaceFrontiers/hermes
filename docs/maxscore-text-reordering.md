# MaxScore text reordering

Implemented field-local text RGB with standalone and merge-time planning,
map-preserving copy merge, and compaction. Query execution preserves stable
document IDs and canonical scores while traversing physical posting order.

## Required behavior

MaxScore text must follow BMP's field-local locality model. An opted-in field
owns a physical scoring-unit order and resolves hits back to stable document
IDs and value ordinals. Store, fast fields, unrelated text and vector fields
retain their existing IDs and bytes. `reorder` selects fields; the existing
`reorder_on_merge` policy, BP budgets, concurrency gate, cancellation and output
claims govern merge-time work. Ordinary merges continue copying compatible
encoded blocks. Reordering is an explicit exception requiring re-encoding.

Plain text remains one scoring unit per document, including aggregated values;
chunked text remains one unit per value. Reordering must preserve IDF, average
lengths, raw score bits, position/ordinal semantics, exact membership/counts,
query-global statistics and tie ordering. In particular, plain fields must not
inherit the chunked field's 90th-percentile BM25 length floor.

## Ownership and representation

The scoring implementation keeps public construction, dispatch, cursor state,
and shared scratch in `query/scoring.rs`. Private `scoring/conjunction.rs` and
`scoring/windows.rs` modules own intersection and window traversal respectively;
both implement the existing `MaxScoreExecutor` and use the same cursor, BM25
batch scorer, collector, and scratch. Ranked conjunction entry and the proven
union tail share one pruning dispatch. Exact counted conjunctions always choose
exhaustive traversal. Unit regressions live in `scoring/tests.rs`.

This source decomposition preserves score operation order, strict pruning
comparisons, logical-ID tie ordering, cancellation checks, public paths, and
persisted bytes. It adds no allocation, cache, dynamic dispatch, or traversal:
scratch remains bounded by `MAX_QUERY_TERMS` and the existing window/block sizes.

Extend `segment/chunk_map.rs` and `segment/text_reorder.rs`; do not add another
permutation writer or query executor. A plain-text map records one physical
slot per document with ordinal zero and unfloored document lengths. Persist a
distinct, version-gated section kind so older readers reject data whose physical
IDs they would otherwise mistake for document IDs. Existing unpermuted segments
remain readable, including when merged with mapped sources.

The segment reader distinguishes physical mapping from chunked scoring semantics.
Query planning uses the map for physical traversal, predicates and target probes;
corpus statistics continue using the declared scoring unit. Existing mapping,
complete-membership, position and result-folding owners handle translated hits.
Plain token positions must survive folding, rather than becoming chunk ordinals.

Merge planning computes one permutation per opted-in field over the surviving
source units before text output is written. The existing posting/position merge
writer consumes this plan, and the map writer emits the matching mapping. It
must not write a first full generation just to reopen and reorder it. No new
publication protocol or ANN rebuild is introduced. Permutation, CSR, term and
position scratch must be charged to the existing budget or spilled in bounded
runs; cancellation must discard unpublished output through its current owner.

## Required validation and measurement

Behavior tests must cover plain/chunked/multivalue/missing fields; term, OR, AND,
exclusion and phrase queries; exact IDs/raw scores/counts/positions before and
after reorder; independent field permutations; old/new source merges; deletion
and compaction; budgets, cancellation and reopen. Unchanged representations are
compared byte-for-byte. Native, async and portable readers must agree. Lifecycle
changes require the full harness. The WASM build and tests are part of the September 17 merge review.

Compare unpermuted and RGB fixtures with the same corpus/compiler/flags and
record build/reorder time, scratch/RSS, bytes, query latency and correctness on
ARM and x86. Measure execution changes separately on frozen index bytes. Do not
claim parity with Tantivy from a selective workload or a changed fixture alone.

### Collection-boundary implementation

The repair explicitly admits compatible term/phrase/Boolean/boost trees into
a field-local physical address space at collection. Query implementations opt in;
unknown queries, cross-field trees, chunked units and fast-column fallbacks retain
logical execution. Child scorers remain sorted within one address space. Existing
Boolean composition and bounded posting batches perform the work; no second
Boolean executor is introduced. Complete streams avoid mapped replay altogether.

Only collection translates physical hits to stable IDs, before heap comparisons
and position callbacks. Deletion predicates resolve candidates through the same
map. Same-field complete/count execution skips frequency scoring naturally when
the collector does not request scores. Ranked term/OR paths already benefiting
from RGB keep their existing executor. Nested physical composition requests
complete child streams to prevent any child heap from truncating candidates in
the wrong address space. The cost is bounded query scratch plus one map lookup
per collected candidate; no corpus-sized per-query permutation is built.

Ranked compatible conjunctions additionally use the existing typed block
executor. It compares stable IDs inside its heap, then translates only its
retained hits back to the enclosing physical stream; this adds O(k) inverse
lookups. Standalone ranked collection consumes these retained results directly,
without another heap. Deleted readers retain complete physical traversal
so filtering cannot discard a truncated winner set. Complete nested clauses
never receive a ranked cutoff. Plain document maps also retain dictionary-based
term cardinality and same-field two-term union counts: their one-to-one mapping
does not change document frequency. Chunked units and independent field maps do
not inherit that optimization.

Ranked handoff uses the existing bounded precomputed-result protocol. Before a
mapped result is truncated, collection takes all retained hits, restores stable
IDs, and applies stable tie order. It must not rebuild a second heap. The map
reader already validates a dense one-to-one document permutation; its logical
slot column therefore provides constant-time inverse lookup, without binary
searching mapped document IDs for each retained hit.

### Required union tails and mapped batch admission

The matched Lucene run isolates the largest family gap in unions. The selected
reader prepares one conservative global bound per mapped exact union: the largest
possible sum omitting one term. Once a full top-k heap strictly exceeds that
bound, every remaining competitive result requires all terms. The reader advances
every cursor past the fully processed window and reuses the existing typed
intersection with the same heap, map and canonical score order. The monotonic
threshold keeps that proof valid for the rest of the stream. Non-finite/negative
term bounds and approximate heap factors disable this transition. The existing
pruning margin covers floating-point reduction error. There is no new scorer,
per-window classification or corpus-sized state. Exhaustion or cancellation at
the handoff returns already-collected hits; crossing the processed boundary
prevents duplicates.

Mapped term/conjunction batches now use the existing eight-score admission loop.
A generic document-ID resolver runs only after its score screen, before the
canonical total-order heap comparison. Underfilled/seeded heaps, equal scores,
non-finite values and stable-ID ties retain the same admission semantics. There
is no allocation or second collector. Exact counts and persisted bytes are
unchanged. Paired probes justify selection; the final combined measurement is
recorded in the repair report.

### Preserve compact formats during explicit reordering

The standalone rewrite now preserves each term's compact posting headers and
compact position directory independently through the existing serializers.
Previously it reconstructed required payloads using legacy output layouts,
losing the compact representation. Inline eligibility remains unchanged. Unplanned fields still use the existing encoded-copy merge path.
No new version, codec, norm policy or publication protocol is introduced.

The regression reproduces layout loss before the repair and compares ranked
score bits, counts, positions and untouched field payload bytes afterward.
Legacy input remains legacy. A newly rebuilt compact RGB fixture is a separate
layout experiment: reader-only measurements continue to use the original frozen
index. Its physical permutation and exact references must match the existing RGB
fixture before attributing timing or residency differences to encoding.

### Two-term contribution elision

Mapped windows omit duplicate per-term scores and presence bits for two
finite, nonnegative contributions. IEEE addition of those two values is
commutative; the existing accumulator already produces the canonical sum,
including the initial positive zero. Longer queries, unsupported scoring
parameters and non-finite bounds retain ordered reduction. It changes neither
pruning nor heap admission. This is selected after exact score-bit references
pass on ARM and x86: x86 official top-10 improves 421.535 → 414.659 µs and
top-1000 improves 833.542 → 819.923 µs; ARM ranked results also improve modestly.
The 5.7% top-10 deficit against same-run Lucene remains.

### Local essential-term window boundaries

[Lucene 10.4 MaxScoreBulkScorer](https://github.com/apache/lucene/blob/releases/lucene/10.4.0/lucene/core/src/java/org/apache/lucene/search/MaxScoreBulkScorer.java)
computes the next outer window from the previous local essential set.
Hermes currently uses the global partition for every boundary, even when a
frequent clause was non-essential in the preceding local window. An isolated
mapped-union experiment retains the previous local essential set for choosing
only the next end boundary. The start still includes every globally essential
cursor, and fresh bounds cover every term over the entire resulting interval;
therefore no previous-window proof is reused to exclude a candidate. The existing
4096-ID cap, score reduction, pruning and heap semantics remain unchanged. This
may amortize bound work without adaptive counters or another executor.

### Coarse ratio-bound pruning on mapped fields

Window execution currently attempts whole-group skips only when a cursor has
impact metadata, even though ratio-bounded RGB postings also store conservative
group bounds. An isolated experiment admits mapped ratio-bounded lists to the
existing group-skip proof. It changes only admission to that proof: group spans,
canonical bound summation, future cursors, deadlines and metadata-only skips
remain owned by the same executor. The cost is one coarse check per encountered
group; the possible benefit is avoiding repeated fine-window bounds in locally
weak RGB clusters. Exact-result tests and both architectures must determine
whether the extra checks pay for themselves.

## Merge review: bounded planning and norm preservation (September 17)

Standalone text reorder must retain untouched norm encodings, including byte
norms, through the owning chunk-map writer. It may not expand an unrelated norm
column merely because another field is reordered. New-column, copied-column and
merge output share the existing CHNK format; no additional file format is needed.

The text BP planner must charge retained field plans, vocabulary discovery,
posting input, pair/CSR construction and graph scratch before allocating them.
It reuses BMP's frequency selection and candidate fitting policy. Fixed scratch
that cannot fit is an explicit budget error. Graph construction rejects invalid
physical IDs and observes cancellation during dictionary and posting traversal.
A term rewrite must similarly admit its retained postings/positions and encoder
scratch before decoding. These checks bound lifecycle work; they do not alter
query scores or introduce a new reorder algorithm.

Explicit reorder repacks position ranges through the existing stream encoder so
permuted document boundaries do not force partially filled output blocks. The
range reader retains only bounded block scratch. Ordinary compaction keeps its
encoded-block copy policy; reorder is the explicit rebuild exception. Regression
coverage compares the repacked bytes with a fresh encoding of the same deltas.

## Integrated merge-time implementation (September 17)

The merge text stage builds BP directly over concatenated source physical
units before opening its output. A shared bounded k-way dictionary traversal
serves both posting merging and BP vocabulary/edge passes. Standalone and
merge-time text use the same planner, permutation validation, and term rewrite.
Source-local cursors retain their source index until position ranges are written.
Only opted-in fields rebuild; all others keep the encoded-copy path.

The CHNK merge writer accepts validated order/inverse slices. It streams
permuted source columns and translates each source logical slot through the inverse,
without allocating another complete map. Legacy plain inputs use the existing
budgeted identity migration; missing norms with nonzero tokens remain an error.
Permutations, source directories, source block buffers, and retained field plans
share the configured BP scratch limit. Map writing follows term rewriting to
avoid overlapping two budget-sized scratch allocations. Text plans and the
shared reorder permit are released before BMP work. Text and BMP convergence
jointly determine publication metadata; truncated BP remains observable.

Tests must compare against copy-merge plus standalone reorder, including raw
payload bytes, scores/positions, missing and multi-valued fields, mixed source
layouts, multiple fields, cancellation and failed output cleanup. No second
segment generation, new format, or new publication path is permitted.

## Candidate probing and canonical window accumulation (September 17)

The text-window executor preserves frozen RGB bytes and canonical f32 document
scores. Native sync and async search share this owner; WASM uses the portable
implementation. None of these reader changes introduce a format, configuration
default, second scorer, or retained cache. Measured selection and rejected
experiments are recorded in [the performance review](search-performance-review.md).

Optional candidate probes retain every input candidate. The shared membership
probe specializes that distinction: required membership compacts matched IDs
and scores, while optional membership changes only matching scores. A lower-bound
search is unnecessary when the current posting already reaches the candidate.
Posting slots use bytes, with a compile-time assertion on block capacity. Both
instantiations preserve block-level and 64-candidate deadline checks and use the
same deferred TF decoder and batch scorer.

When every term is essential and the prepared scorer proves finite nonnegative
contributions, visiting terms in canonical query order makes the dense window
accumulator exact. Absent positive-zero additions cannot change that sum. These
windows omit duplicate contribution-plane writes and the final fold. Windows
that defer any term retain the existing per-term storage and canonical fold;
the existing two-term commutative case also retains its admission.

### Bounded window setup and cost-aware partitioning

When the current posting block covers the window, the bound lookup returns its
cached bound without searching the skip directory. The existing whole-group
substitution and zero clamp remain intact. Other windows use the same
multi-block/group traversal and bound arithmetic.

Mapped OR queries with several globally essential terms amortize small windows.
The executor tracks evaluated windows and candidates before optional probing.
An average below 32 candidates per term doubles the minimum span, capped by the
existing 4,096-ID scratch; adequately populated windows reset it to one. Required
queries and unmapped execution retain their original spans. Every enlarged
window still computes conservative bounds over its full range. The motivating
algorithm is Lucene's `computeOuterWindowMax` in
[MaxScoreBulkScorer](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/MaxScoreBulkScorer.java).

For finite nonnegative mapped OR scorers, local bounds are ordered per posting-
list entry. Reciprocal list costs are prepared once per query in a fixed
`MAX_QUERY_TERMS` array (eight bytes per term slot on the stack). The existing
prefix-bound proof and canonical scorer retain responsibility for correctness;
there is no second executor, unbounded scratch, or retained cost cache.

### Ranked mapped-pair pruning

Two-term semantic AND top-k and proven OR-to-conjunction tails reuse the same
compile-time pruning mode in the existing typed intersection. Admission requires
a mapped field, exact heap factor, finite prepared bounds and ratio metadata on
both lists. All other conjunctions and **every exact-count traversal** retain
the exhaustive instantiation.

Until the earlier current block end, every possible match is bounded by the sum
of those two block bounds. A sum strictly below the existing conservative heap
threshold advances the block ending first; equal ends advance both. This always
makes progress and can avoid decoding TFs and later payloads. Equal-score bounds
remain eligible, preserving stable logical-ID ties. The scorer, canonical sum,
collector, cursor protocol and deadline checks remain shared. The count returned
by a pruned traversal is a work statistic, never an exact result count.

Regression tests compare public sync/async ranked results and exact-count
results, preserve ties, and observe expired budgets. A strong-prefix fixture
scores 256 of 8,192 matches; tied scores still require all 8,192, and counted
execution remains exhaustive in either case. Final validation and independent
paired timing are recorded in the performance review.
