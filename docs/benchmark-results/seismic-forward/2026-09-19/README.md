# Forward codec evidence, 2026-09-19

Baseline: clean main commit `782227808a54c560b231d19e0a4d46662ecca334`.
Candidate: the production source hashes in [manifest.json](manifest.json).
Both runners use identical dependency lockfiles, Rust 1.98.1, release/thin LTO,
one codegen unit, and `-C target-cpu=native`. The VM has an x86 Xeon 2.8 GHz,
eight vCPUs and approximately 62 GiB RAM. Search uses four workers and copy
pinning with a 64 MiB per-segment budget. See [environment.json](environment.json)
and the [reproduction instructions](../README.md).

The 1M fixture has 126,320,094 coordinates and Float32 weights. Warm top-100
search uses 200 queries, three passes with the first excluded, and three trials
per format in balanced order. Each run discards clean cached ranges and
sequentially prewarms its files. Peak RSS includes mapped pages and heap.

| Layout                   | Warm mean ms | Warm p95 ms | Peak RSS GiB | Exhaustive mean ms | Top-10 mean ms |
| ------------------------ | -----------: | ----------: | -----------: | -----------------: | -------------: |
| Original main, U32       |        29.85 |       48.23 |         2.00 |             275.02 |          16.34 |
| Candidate, U32           |        27.99 |       45.43 |         2.00 |             259.31 |          15.63 |
| Candidate, forced U24    |        31.50 |       51.89 |         1.89 |             352.05 |          17.30 |
| Candidate, adaptive gaps |        32.79 |       53.68 |         1.70 |             378.16 |          17.62 |

Exhaustive and top-10 probes contain five and 50 queries respectively, with
three passes each and the first excluded. Values above average trial statistics;
p95 values are averages of per-trial p95s, not pooled percentiles. All warm
trials report zero physical reads after prewarming. Full distributions, p99,
open time and fault counters are in [1m-results.json](1m-results.json); grouped
statistics are in [1m-summary.json](1m-summary.json).

## Memory limit

The pressure probe uses a fresh 1 GiB cgroup, swap disabled, cold input files,
and 20 queries repeated three times. Latency excludes the first pass; I/O
counters include all three passes. Two independent trials per format run in
forward and reverse order. Every trial returns exactly the baseline IDs/scores.

| Candidate layout | Mean ms |  p95 ms | Trial mean range ms | Physical reads GiB |
| ---------------- | ------: | ------: | ------------------: | -----------------: |
| U32              |  984.41 | 2026.23 |       980.11–988.71 |               2.44 |
| Forced U24       |   35.77 |   58.86 |         35.72–35.83 |               0.96 |
| Adaptive gaps    |   33.97 |   54.46 |         33.19–34.74 |               0.77 |

This is a cache-capacity threshold for this **20-query working set**, not a
universal 29x speedup. Raw rows repeatedly refault; adaptive rows fit under the
limit (about 850 MiB cgroup peak, no limit events), while U24 barely fits.
Neither this probe nor the 1M fixture determines RAM requirements at billions
of vectors. The broader warm 200-query workload touches more pages. Compression
saves bytes but adds CPU work, especially for exhaustive lookup-table scoring.

## Wide IDs

The wide fixture has 100k real vectors, with document/query IDs shifted by
65536 into a 100k vocabulary. It uses the existing sorted-query scorer. It
preserves the original sparsity and gaps and is **not** a genuine 100k-token
embedding-model benchmark.

| Layout                   | Warm mean ms | Peak RSS MiB | Exhaustive mean ms |
| ------------------------ | -----------: | -----------: | -----------------: |
| Original main, U32       |        10.33 |       302.58 |              78.87 |
| Candidate, U32           |        10.23 |       302.50 |              77.56 |
| Candidate, forced U24    |        10.46 |       290.32 |              80.27 |
| Candidate, adaptive gaps |        10.72 |       271.00 |              77.72 |

See [wide-results.json](wide-results.json) and [wide-summary.json](wide-summary.json).
All measured exact/approximate IDs and scores match the baseline, including
above-65535 IDs. This is equivalence to the existing candidate policy; relevance
recall and concurrent throughput were not measured. Native ARM and portable
codec correctness were tested, but no final ARM latency comparison is claimed.

## Validation and provenance

[validation.json](validation.json) records the full harness before the hot-loop
refinements and the final focused harness after them: 2013 native tests passed,
25 existing tests ignored; the full run also passed five real-server tests.
The final WASM build passes all 37 tests. The focused Seismic suite passes 111
tests; the native Linux codec suite passes four tests covering all controls,
tails, high bases, and width boundaries.

[archive.sha256](archive.sha256) identifies the retained full evidence archive.
Its checksum, measured binaries, identical runner lockfiles, and all recorded
production-source hashes were verified after download. Large artifacts stay in
`.context/forward-compression/forward-final-evidence.tar.gz`: raw hits/logs,
source archives, binaries, and rejected-prototype measurements. The latter are
not included in the final tables. The VM is stopped after export; its status is
confirmed as `TERMINATED` in [vm-status.json](vm-status.json).
