# zstd-capacity evidence

See the [results](../../search-benchmark-dictionary-results.md) and
[performance review](../../search-performance-review.md). This snapshot contains
252 hash-verified files: raw complete-workload timings, correctness
records, resource logs, machine/index manifests, adapters, local ARM evidence,
and changed build inputs for both measured builds. Unchanged sources come from
base `ce2c96b945fccc4bac58ccc45bb4bc23b809773e`. Later merge-admission and density-
admission changes are outside this snapshot.

`raw-results.zip`: 3,611,615 bytes; SHA-256 `00c264824e5df673a6e802cd848f1ec17867e83840a29ec5062066ecd9ef8155`. Individual hashes and
sizes are in `manifest.json`. The complete executable archive remains in the
workspace as `.context/zstd-capacity-cloud-evidence.tar.gz` (18,519,983 bytes), SHA-256
`60a9c95e44c5a52f7682ca444a4dacae25da3bcac7edf60c4d3b2c81a439c11a`. Every member and unchanged index manifest was verified after download.
The machine remains available for follow-up experiments.
