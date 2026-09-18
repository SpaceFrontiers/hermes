# Posting validation cache evidence

See the [results](../../search-benchmark-validation-results.md). `raw-results.zip`
contains 203 hash-verified files: raw samples, counts, build/test logs,
profiles, machine/index manifests, analysis scripts and the exact changed build
inputs for both measured builds. Unchanged sources come from base
`ce2c96b945fccc4bac58ccc45bb4bc23b809773e`. The two supplemental mmap tests are
included separately. Subsequent dictionary changes are outside this snapshot.

ZIP: 2,197,447 bytes; SHA-256 `309d49ae6e10ddf5e8e23e0d28b8555d9d8d01d78f292096d165e139a1cf44d0`. Individual entries are listed in
`manifest.json`. Full executable/perf-data archive is retained in the workspace
as `.context/validation-cloud-evidence.tar.gz` (27,262,114 bytes), SHA-256
`6593272b203f11654d38ad00200bc7175d45fd4d5207a81e8a3ee1f8e6b13b45`.
The cloud VM remains active for the next experiment.
