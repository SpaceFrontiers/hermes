# Forward dimension codec comparison

These are offline benchmark tools, not production migration code. Input is the
same single-field, four-run SPLADE 1M version-5 fixture used in the
[summary compression study](../../seismic-compact-summaries.md). The converter
calls the production dimension encoder and checks every reconstructed ID;
weights and nomination payloads are preserved. Output uses version 6. The
production reader supports only version 6.

Build the runner in both baseline commit `782227808a54c560b231d19e0a4d46662ecca334`
(a clean main checkout at the start of this experiment) and the candidate using
the same lockfile, Rust toolchain, machine, and flags. Copy their binaries to
`$FORWARD_EVIDENCE/{before,after}-bin`. The measured runner used release mode,
thin LTO, one codegen unit and `RUSTFLAGS='-C target-cpu=native'`.

```sh
cargo build --release --manifest-path docs/benchmark-results/seismic-forward/runner/Cargo.toml
rustc --edition=2024 --cfg 'feature="native"' -C opt-level=3 -C target-cpu=native docs/benchmark-results/seismic-forward/convert.rs -o "$FORWARD_EVIDENCE/convert"
```

Prepare `raw-index`, `u24-index`, and `dot-index` under the evidence directory:

```sh
"$FORWARD_EVIDENCE/convert" "$FORWARD_FIXTURE" "$FORWARD_EVIDENCE/raw-index" raw
"$FORWARD_EVIDENCE/convert" "$FORWARD_FIXTURE" "$FORWARD_EVIDENCE/u24-index" u24
"$FORWARD_EVIDENCE/convert" "$FORWARD_FIXTURE" "$FORWARD_EVIDENCE/dot-index" dot
python3 docs/benchmark-results/seismic-forward/matrix.py
python3 docs/benchmark-results/seismic-forward/summarize.py "$FORWARD_EVIDENCE"
```

`FORWARD_QUERIES` points to the original CSR queries. `FORWARD_PRESSURE=1` adds
Linux cgroup probes capped at 1 GiB with swap disabled, repeated in reverse
order; these commands use the
benchmark machine's `pasha` account. Warm trials have balanced ordering, 200 queries,
three passes with the first excluded, four search workers, and a 64 MiB metadata
pin budget per segment. Independent top-10 and exhaustive probes compare IDs
and scores with the clean baseline. Compiler overlap is monitored and rejected.
Peak RSS and physical read/fault counters accompany latency.

`wide_fixture.py` preserves the first 100k real vectors and their weight bytes,
adds 65536 to every document/query coordinate, and sets vocabulary size to 100000. It exercises 17-bit IDs and the existing sorted-query scoring path.
This relabeling preserves gap distribution: it is a correctness and performance
probe for wide IDs, not a representative 100k-token embedding model. Build that
fixture using the baseline runner's `build INDEX CORPUS 100000 100000 seismic`
command, then repeat conversion and warm comparisons with shifted queries.

Raw U16 can also be emitted with mode `u16` for size comparisons on narrow
vocabularies. The production policy uses adaptive widths and gap encoding;
forced U24 in this experiment isolates narrowing from delta decoding.

Final warm runs first discard clean cached file ranges and then prewarm them
sequentially, after a `sync`, so file creation history does not define the cache
layout. The initial prototype measurements are retained separately; they expose
the outlined-weight-decoder regression and are not the final performance claim.

Recorded results are under [2026-09-19](2026-09-19/README.md). Full raw hits,
logs, benchmark binaries, and source archives are retained in the workspace's
ignored `.context/forward-compression/forward-final-evidence.tar.gz`; the tracked
manifest records their source/fixture hashes. Each format returned identical
IDs and scores for all measured queries. Throughput under concurrent load and
recall against relevance judgments were not measured.
