# BMP Grid Memory: Compression Research & Roadmap

Status: research (2026-07-10). Phase 1 (per-field `bmp_block_size` in SDL +
MADV_RANDOM on the grid) implemented; Phase 2 (sparse grid) **benchmarked
and rejected at current scale** — see measurements; Phase 3 (dim pruning)
assessed — not recommended.

## Memory anatomy of a BMP segment (V13/V14)

Everything is mmap-backed (page cache, not heap), but resident-set pressure
is real on memory-bound deployments. Per field, per segment:

| section        | bytes                          | 18M docs, 105879 dims, block 64 |
| -------------- | ------------------------------ | ------------------------------- |
| block grid (D) | `dims × ceil(num_blocks / 2)`  | **≈ 14.9 GB**                   |
| sb_grid (E)    | `dims × ceil(num_blocks / 64)` | ≈ 465 MB                        |
| doc maps (F+G) | `6 × num_virtual_docs`         | ≈ 108 MB                        |
| block data (B) | `2 × postings + headers`       | ≈ 6-8 GB (data, not overhead)   |

The **dense block grid dominates**: every one of `dims` vocabulary rows gets
a 4-bit cell in _every_ block, even though a typical SPLADE block touches
only a small fraction of the vocabulary. Measured coherence
`d = total_postings / total_terms ≈ 2-3` at block 64 implies the average
non-zero density of the grid is `total_terms / (dims × num_blocks)` — for
the production corpus roughly **3-4% non-zero**. The dense grid stores the
other ~96% as zero nibbles.

## Phase 1 (shipped): per-field `bmp_block_size` + grid MADV_RANDOM

Grid bytes scale as `1/block_size`. Moving a field from 64 → 256:

- grid: 14.9 GB → **3.7 GB** (4×), sb_grid: 465 MB → 116 MB (18M docs);
  at 50M docs: 41 GB → **10 GB**.
- pruning granularity coarsens: a block is now 256 docs, so a superblock
  covers 16k docs. The BMP paper (Mallia, Suel & Tonellotto, SIGIR 2024)
  favors b=8-32 for in-RAM MS MARCO-scale corpora; at 100k-dim vocabularies
  and 10M+ doc segments the grid does not fit that regime — coarser blocks
  are the correct trade (see also docs/budgeted-reorder.md research notes).
- Block size is **per field, uniform across all segments** (default 64) —
  set it in SDL at index creation; there is deliberately no auto-escalation
  and no mixed-block-size support (block-copy merge and blockwise reorder
  move blocks verbatim and validate uniformity loudly).
- The block grid is now `MADV_RANDOM`: queries at production scale read 32
  bytes per (query dim, surviving superblock) at effectively random offsets,
  and default kernel readahead amplified each probe into 128 KB of page
  cache (~4000×) — marching the entire data-sized grid into memory and
  OOMing cgroup-limited deployments. With MADV_RANDOM only the touched 4 KB
  pages fault in, and they stay cleanly reclaimable.

SDL:

```sdl
field embedding: sparse_vector<u32> [indexed<format: bmp, dims: 105879,
    max_weight: 5.0, bmp_block_size: 256>]
```

## Phase 2 (benchmarked, rejected at current scale): sparse (CSR) block grid

### Idea

Store, per dim, only the blocks where the dim actually occurs:
`row(t) = [(block_id, impact_u4), ...]` grouped into superblock runs. The
entry count is exactly `total_terms` — already a footer stat.

### Why everyone else keeps the grid dense

Block upper bounds stored sparsely are not exotic — that is exactly what
DAAT block-max engines (BMW/MaxScore) do, per posting list. BMP's specific
contribution (Mallia, Suel & Tonellotto, SIGIR 2024) was to **densify**
those bounds into a dims×blocks matrix so block-at-a-time scoring becomes a
contiguous SIMD sweep (`accumulate_u4_weighted`, NEON/SSE4.1). Sparsifying
the grid un-does the thing that makes BMP fast. At larger scales the field
does not sparsify the grid either — it abandons exhaustive grids for
approximate per-list summaries (Seismic: Bruch, Nardini, Rulli & Venturini,
SIGIR 2024; journal version arXiv:2509.24815 — assessed in
docs/seismic-research.md).

### Measured (microbenchmark, aarch64 NEON, warm cache)

`grid_bench` in `segment/reader/bmp.rs` — Zipf(1.0) over 100k dims, 120
nnz/doc, 2M docs, 16-dim queries, 30% surviving superblocks, real SIMD
kernel for dense vs binary-search + scalar scatter for CSR sb-runs
(correctness cross-checked entry-exact between layouts):

```
cargo test --release -p hermes-core --features native --lib -- \
  --ignored bench_grid_dense_vs_csr --nocapture
```

| config    | density | grid memory: dense → CSR | block-UB compute: dense → CSR | probes hitting |
| --------- | ------- | ------------------------ | ----------------------------- | -------------- |
| block 64  | 3.5%    | 1562 → 227 MB (**6.9×**) | 12 → 59 ns/probe (**5.1×**)   | 94%            |
| block 256 | 10.4%   | 391 → 137 MB (**2.9×**)  | 11 → 72 ns/probe (**6.3×**)   | 99%            |

Two hypotheses failed:

1. **"Sparse rows skip absent dims"** — real query dims are Zipf-head dims,
   present in nearly every superblock (94-99% probe hit rate). The skip
   advantage barely exists; every hit pays a binary search plus a scalar
   scatter instead of one 32-byte SIMD sweep.
2. **The memory win shrinks exactly where large fields are going**: at
   block 256 density triples, so CSR yields only 2.9× — for a 6.3×
   hot-loop regression.

Note also that the dense grid is mmap-backed: rows of dims that queries
never touch are never faulted in, so the _resident_ dense grid is already
concentrated on head-dim rows — the same rows CSR must keep hot.

**Verdict: not worth it at 100k-dim vocabularies with block 256.** Revisit
only if (a) vocab grows ≫100k (density falls, memory ratio improves), or
(b) profiling shows the deployment truly page-thrashes on grid rows — and
then benchmark against the cheaper alternative below first. x86 AVX2 not
yet measured (aarch64 only); re-run the bench before any decision.

### 2-bit grid impacts — MEASURED, IMPLEMENTED (`bmp_grid_bits: 2`)

Halve the dense grid (dims × blocks / 4) by quantizing block UBs to 2 bits
(ceil-quantized, recall-safe like `quantize_u8_to_u4_ceil`). Same layout,
same SIMD shape, no probe-pattern change, no format fork beyond a footer
flag.

Measured (`bench_aggressive_quantization`, 400k docs / 256 topics /
BP-reordered, exact k=10; grid lattices ceil-rounded on the stored file so
the REAL executor runs both ways; top-k asserted bit-identical — bounds
only loosen, exactness is structural):

| grid            | topical queries (blocks/q) | background-only queries (blocks/q) |
| --------------- | -------------------------- | ---------------------------------- |
| 4-bit (shipped) | 25.7                       | 6057                               |
| 3-bit (9 lvl)   | 25.8 (+0.4%)               | 6122 (+1.1%)                       |
| 2-bit (4 lvl)   | 25.8 (+0.4%)               | 6188 (+2.2%)                       |

Why so cheap: the exact u8 `sb_grid` prunes first and shields the block
grid — coarse block bounds only act inside surviving superblocks. **2-bit
grid ≈ free** (grid 10 GB → 5 GB at 50M docs / block 64; combine with
`bmp_block_size: 256` for 2.5 GB).

**Implemented**: SDL `indexed<bmp_grid_bits: 2>` (default 4; per field,
uniform across segments — merge/blockwise validate loudly). The V13 blob
footer's reserved field carries the cell width (0 = legacy 4-bit), so
existing segments stay readable. 2-bit currently uses an unrolled scalar
accumulate/mask kernel (4 cells/byte); SIMD u2 variants are a follow-up if
profiling asks for them — the block grid is probed only inside surviving
superblocks. Exactness is pinned by
`test_bmp_grid_bits_2_exact_parity_and_roundtrip` (identical top-k vs a
4-bit index over the same corpus, through block-copy merge and BP reorder).

### 4-bit posting weights — MEASURED, REJECTED

Snapping posting impacts to the u4 lattice (u8 multiples of 17; emulated at
build time so quantization applies to both scores and grid maxes) costs
real quality: **recall@10 vs the 8-bit index = 95.1%** on the same corpus,
matching the classical 3-5% loss for sub-8-bit impact quantization
(Anh & Moffat lineage). The saving is only ~25% of the postings section —
NOT 50%, because a BMP posting is 2 bytes `(local_slot: u8, impact: u8)`
and only the impact half shrinks: packed, 2 B → 1.5 B per posting. Bad
trade — weights stay 8-bit.

## Phase 3 (assessed, not recommended): dropping grid rows for rare dims

Term-centric static pruning (Carmel et al., SIGIR 2001 lineage) suggests
dropping upper-bound rows for rare dims and treating their UB
conservatively (global max weight for all blocks). But a conservative UB
inflates every block bound for queries containing rare dims — exactly the
discriminative dims BMP prunes best with — and with CSR rejected the
"row costs bf_t entries" argument is moot: under the dense grid rare-dim
rows are mmap pages that queries rarely fault in anyway.

## SSD-friendliness / paging behavior

The grid is mmap-backed; "SSD-friendly" means the resident set stays
bounded and misses are served as clean 4 KB NVMe refaults:

- **MADV_RANDOM on the grid** (shipped): kills readahead amplification —
  per query the grid working set is `query_dims × surviving_superblocks ×
4 KB` instead of ×128 KB. Pages are clean and reclaimable; under memory
  pressure the kernel evicts them and cold queries pay ~20-50 µs NVMe
  refaults per page instead of the pod OOMing.
- **sb_grid stays readahead-friendly and pinnable** (priority 4 in the pin
  budget): it is swept contiguously per query dim and is 64× smaller than
  the grid — keeping it resident preserves superblock pruning, which is
  what keeps the number of grid probes small in the first place.
- The block grid is deliberately never pinned (data-sized).

## Rollout

1. ✅ Phase 1 — rebuild large indexes with `bmp_block_size: 256` in SDL
   (4× on grid + 4× on sb_grid); MADV_RANDOM ships with the binary and
   needs no rebuild.
2. If still memory-bound after 256: benchmark 2-bit grid impacts (skip
   ratios on production data), and re-run `bench_grid_dense_vs_csr` on the
   production arch before reconsidering CSR.
