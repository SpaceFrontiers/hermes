# Standalone RGB benchmark evidence — September 16, 2026

See [results and diagnosis](../../search-rgb-benchmark.md). This experiment keeps
the selected search reader frozen and changes only the index layout. It does not
replace the [RGB-off parity result](../block-execution-2026-09-16/README.md).

## Artifacts

results.zip (local archive `results.zip`) contains per-query timings, query lists, exact-reference summaries,
source/adapter provenance, construction and check logs, work counters, file hashes,
memory mappings and the downloaded public benchmark snapshot. [manifest.json](manifest.json)
identifies logical paths, deduplicated ZIP members, byte counts and SHA-256.
When extracting, use each manifest entry's `archive_path` to restore its logical
path; identical files are stored once. Binaries and indexes are excluded; their hashes identify them. Every member and
the ZIP are verified after packaging.

The selected core overlay is `arm/packed-source.tar.gz` in the RGB-off archive.
Apply `arm/adapter.patch` from this archive to expose the existing RGB API in the
benchmark adapter; core code is unchanged. `arm/source-audit.json` records the
comparison. The timing binary remains the frozen `packed` executable, not the
newly built adapter or diagnostics binary.

## Reproduction

Use the same canonical corpus and query set as the RGB-off comparison: full
5,032,104-document x86 corpus, first 100,000 documents on ARM. Preserve the
original compact index and its exhaustive reference before building RGB fixtures.

1. Build the adapter using Rust 1.98.1, release LTO and
   `RUSTFLAGS='-C target-cpu=native'`; source/compiler/binary hashes are recorded.
2. `arm/build-fixture.py` builds an explicitly eligible identity-map index,
   preserves its hashes, clones it, invokes `IndexWriter::reorder` through the
   adapter, and verifies both layouts against the frozen oracle.
3. `arm/measure.py arm|cloud` runs seven rotated official ranked passes and five
   supplemental ranked passes. ARM also measures counted commands with seven
   official/five supplemental passes. The incomplete x86 counted repetitions
   were excluded; their configurations, logs and scope adjustments are retained.
   `cloud/rgb-memory.py` runs separate fresh-process Linux residency checks with
   one complete pass each of top-10 and top-1000 for every engine. Do not overlap
   latency with builds, diagnostics or audits.
4. `arm/diagnose-arm.py` uses a separate `query-diagnostics` build and saves the
   second pass of every command. These work counters are not latency samples.
5. `arm/render.py` regenerates the report; `arm/package.py` verifies the evidence
   archive. Scripts use captured workspace/cloud paths; adapt them deliberately.

The cloud coordinator retains logs and hashes from an initial reorder with a
mismatched dictionary target, then rebuilds from the identity control using the
matched 4 KiB setting. The reported measurements use only the corrected fixture.
Files prefixed `rejected-dictionary-settings-` are excluded experiments. The
rejected full-corpus verification was interrupted after publication to avoid
finishing a redundant check; corrected-fixture verification remains mandatory. The
interrupted non-LTO adapter build did not produce a benchmark fixture.

The existing RGB writer emits its older posting/position formats. Consequently
this experiment measures the complete current feature, including its format
change and mapping overhead, not the permutation algorithm in isolation.
