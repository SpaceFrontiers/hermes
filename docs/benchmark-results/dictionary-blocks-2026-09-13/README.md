# Dictionary block evidence

See the [results](../../search-benchmark-dictionary-results.md). `raw-results.zip`
contains 384 hash-verified files: raw cloud and ARM samples, counts,
resource logs, fixtures and machine/index manifests, preparation and analysis
scripts, full validation logs, and exact changed build inputs for the preceding
validation build and both dictionary builds. This includes the default-format
binary golden fixture. Unchanged sources come from base
`ce2c96b945fccc4bac58ccc45bb4bc23b809773e`.

ZIP: 5,581,334 bytes; SHA-256 `bf9ab3a6c36397694038abc4ce7222a21b04b8054fd0182c2994cd787f8f319d`. Individual entries are listed in
`manifest.json`. The executable archive is retained in the workspace as
`.context/dictionary-cloud-evidence.tar.gz` (27,178,425 bytes), SHA-256
`5a2129da5db4769146be22af82325cc5b15c3a8604b239263f5a4ff09ea35550`.
All 136 members and unchanged index manifests were verified after capture.
Later Zstd allocation and conjunction changes are separate experiments; their
results and build inputs are outside this snapshot. The cloud machine remains in
use for those experiments.
