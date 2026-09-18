# Score-bound follow-up evidence — 2026-09-13

See the [results and limitations](../../search-benchmark-ratio-results.md).
`raw-results.zip` preserves complete timing samples, exactness gates, validation,
RSS measurements, source/binary hashes and scripts. `manifest.json` gives the
SHA-256 and size of every member. Cloud and ARM runs remain separate; v1 and v2
are intermediate candidates, and v3 is the final production source.

Extract and recompute the final paired cloud comparison from the repository root:

```bash
python3 -m zipfile -e docs/benchmark-results/score-bounds-2026-09-13/raw-results.zip .context/score-bound-results
python3 scripts/search_benchmark/analyze.py .context/score-bound-results/cloud/bench-evidence/ratio-v3-official.json \
  --baseline hermes-before-ratios --candidate hermes-ratios-v3
```

Use `--baseline tantivy-0.26 --candidate hermes-ratios-v3` for the competitor
comparison. The official run has all 962 queries, 60-second warmups and ten
repetitions. The supplement has all 714 distinct terms, ten-second warmups and
five repetitions. Cache-capacity comparisons use the same final binary and
index: ten seconds/five repetitions for terms, 30 seconds/five repetitions for
the official set. They are separately labeled and change no defaults.

ARM runs use only the first 100,000 corpus documents. Engine `hermes-arm-plain`
is the initial ratio-capable binary on the legacy index; `hermes-arm-ratio` is
that binary on the ratio index; `hermes-arm-single` is v2 on the ratio index;
`hermes-arm-handoff` is v3 on the ratio index; `hermes-arm-handoff-plain` is v3
on the legacy index. Keep the separate index-build effects and repeated-run
COUNT drift visible when interpreting small differences.

The larger local `.context/hermes-ratio-benchmark-evidence.tar.gz` additionally
contains frozen source overlays, the base Git bundle, Linux/ARM binaries,
complete cloud evidence and the ARM corpus prefix. It has a checksum sidecar.
The older `.context/hermes-benchmark-evidence.tar.gz` and original versioned
results bundle remain intact. The follow-up VM is deleted after preservation;
cleanup evidence is included with this bundle.
