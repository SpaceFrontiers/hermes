# Ranked pruning evidence — September 24, 2026

The [report](../../ranked-pruning-followup.md) explains the changes, rejected
experiments, measurements and remaining limitations. The
[research assessment](../../score-guided-traversal.md) describes proposed future
structures separately from the measured implementation.

The numbered `results.zip.001`–`.003` parts retain final and exploratory cloud measurements, request/response
checks, exhaustive audits, CPU and memory samples, separate diagnostic counters,
ARM samples, source overlays, compiler/executable provenance, measurement
drivers and validation logs. It excludes executables and the corpus. The
manifest records the archive hash and each retained file's hash.
The parts keep individual files below the repository's large-file limit. The
analysis command concatenates them in memory automatically; concatenating them
in numbered order also produces an ordinary ZIP archive.

Recompute the summary and verify its result/error gates with:

```bash
python3 docs/benchmark-results/ranked-pruning-2026-09-24/analyze.py
```

Cloud before/after values use the median of six repetitions across two reversed
session orders. Session medians, min/max QPS, CPU microseconds per request,
logical CPU utilization and peak RSS/anonymous RSS remain in `summary.json`.
Luxir has three fresh repetitions. ARM values average two process medians and
retain individual sample arrays and exact ranking signatures.

These are warm, targeted measurements of 15 queries from three problem
families, not an all-826-query comparison. Cross-layout numbers are not a clean
RGB ablation because the historical indexes used different indexing-worker
counts. The shared-Mac ARM fixture also contains measured regressions and cannot
establish a universal no-regression result.
