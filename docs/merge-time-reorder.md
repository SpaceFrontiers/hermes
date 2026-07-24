# Merge-Time BP Reordering

Status: design, implemented; actualized 2026-07-24.

## Problem

Recursive Graph Bisection (BP) reordering was originally a standalone,
whole-segment operation: merges block-copied BMP data in concatenated source
order, and a separate pass (`hermes-tool reorder` / the server's background
optimizer) later rewrote the entire segment — copying every unchanged file and
rebuilding the sparse file. For a freshly merged segment the index was
rewritten **twice**: once by the merge, once by the reorder.

That makes ordering quality arrive in one large, deferred, IO-heavy step
instead of amortizing over the index's natural rewrite cycle.

## Design

Run BP _inside_ the merge, where the output sparse file is being written
anyway:

- Index-level SDL option, persisted in the schema at creation:

  ```sdl
  index articles {
      reorder_on_merge: true
      field splade: sparse_vector<...> [indexed, reorder]
  }
  ```

  Absent = disabled — merges block-copy exactly as before (the default
  preserves current behaviour). Programmatic:
  `SchemaBuilder::set_reorder_on_merge(true)`. Setting the option with no
  `reorder`-attributed field warns loudly at parse time (it would do
  nothing).

- Per-field gate: only sparse fields carrying the `reorder` schema attribute
  (`field splade: sparse_vector<...> [indexed, reorder]`,
  `SchemaBuilder::set_reorder`) are BP-reordered — by merges AND by the
  standalone reorder paths (`hermes-tool reorder`, `IndexWriter::reorder`,
  the server's background optimizer). Fields without the attribute have
  their blob copied byte-identically (insertion order preserved — right for
  corpora whose insertion order already clusters well, e.g. date-sorted).
  Skips are logged loudly. Note this tightened an old behavior: standalone
  reorder used to BP every BMP field regardless of the attribute.
- When enabled, `merge_sparse_vectors` replaces the byte-level block-copy of
  BMP fields with: build a multi-source forward index from the source
  `BmpIndex`es → run BP (`graph_bisection`, same parameters as standalone
  reorder) → write the merged blob in permuted order. Doc-id maps are patched
  with per-source offsets during the write.
- The merged segment is marked `reordered: true` in metadata, so the
  background optimizer skips it — eliminating the second whole-segment
  rewrite.
- Non-BMP fields and all other segment files merge exactly as before.

Amortization property: the tiered merge cascade rewrites small segments often
and large segments rarely, so BP cost is paid in proportion to (and at the
same moment as) IO the merge already spends. BP warm-starts from input order,
but concatenating individually ordered sources does not guarantee that their
global record-level partition tree is already aligned. `BpGranularity::Auto`
uses the cheap block-level path when source block coherence justifies it;
otherwise record-level BP can still dominate merge wall time.

## Backlog and explicit force-merge behavior

Merge-time BP is optional optimization, not a prerequisite for a correct
replacement. Under the large-scale tiered policy, a topology is “severe” when
eligible live segments exceed twice the computed tier budget. Automatic
merges then block-copy BMP payloads and mark outputs unreordered. Wide (up to
24-input) compaction drains the segment backlog first, and the optimizer later
orders the much smaller output set. This prevents every automatic merge slot
from serializing behind long BP passes during instant indexing.

Explicit force merge reserves foreground merge capacity and, when
`reorder_on_merge` is enabled, foreground BP capacity. It packs sources under
`max_segment_docs`, reduces groups through a balanced 64-way hierarchy when
needed, and runs BP only on a group's final reduction.
Intermediate reductions use the fast block-copy path, so hundreds of tiny
segments do not repeatedly reorder the same documents. Reader and primary-key
snapshots refresh after every replacement to retire source files incrementally.

## Interior-padding correctness (bug fixed alongside)

Fresh BMP segments pad only the tail of the doc map (`u32::MAX` slots after
`num_real_docs`). Block-copy merges concatenate sources _including_ each
source's tail padding, producing merged segments whose real docs are **not**
contiguous in vid space. The standalone reorder and BP forward-index builder
assumed `vid < num_real_docs` ⇔ real doc — on such segments this silently
dropped real docs shifted past `num_real_docs` (their postings and doc-map
entries vanished from the reordered blob) and treated interior padding slots
as empty docs.

Fix: both the forward-index builder and the reorder blob writer now derive
realness from the doc map itself (`doc_map_ids[vid] != u32::MAX`) via
per-source real↔virtual mappings. This is pinned by
`test_bmp_reorder_after_merge_keeps_interior_padded_docs`. A side benefit:
any BP rewrite (merge-time or standalone) compacts interior padding, so the
output always has tail-only padding.

## Cost

- Extra merge CPU: forward-index build + BP. Bounded by the same
  `memory_budget` df-dropping as standalone reorder (24 GB default). BP runs
  in `spawn_blocking`; Rayon parallelism uses the process-wide background
  pool. The server's `--optimizer-concurrent-passes` gate covers this path too,
  so merge-time and standalone passes cannot multiply without bound.
- Saved: one full segment rewrite per merge (the optimizer pass), including
  its read traffic against the page cache.
- Merge wall-clock grows; for latency-sensitive ingest keep the flag off and
  rely on the background optimizer. Severe automatic-merge backlogs make this
  trade automatically until topology returns within budget.

## Future (not implemented)

- Debt-driven scheduling: prioritize standalone reorder by observed
  `blocks_scored/blocks_total` pruning ratio per segment.
- Routed placement: order small-segment docs by superblock term-mass
  signatures of the largest segment before merging.
