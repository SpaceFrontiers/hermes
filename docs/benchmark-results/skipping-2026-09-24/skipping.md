# Skipping ablations — September 24, 2026

Ratios are relative to the unchanged 1.9.1 control on the same immutable index. Higher QPS ratios are better; lower CPU ratios are better. Each cell combines three repetitions from each of two reversed-order rounds. The summary gives an unweighted geometric mean across ranked family/operation cells, not an overall engine throughput score.

Phrase variants cover 197 exact/sloppy phrase queries (12 ranked cells per layout); no-rare-seek covers 147 conjunction queries (six ranked cells per layout). Exact-count cells are controls and do not enter the ranked averages. [All paired cells](skipping.json) retain QPS ranges, per-round ratios, CPU/request, driver utilization, and anonymous/total resident memory.

| Variant        | Layout | Ranked QPS ratio | CPU/request ratio | Ranked cell ratio range | Count QPS ratio |
| -------------- | ------ | ---------------: | ----------------: | ----------------------: | --------------: |
| no-pilot       | plain  |           1.020× |            0.983× |            1.006–1.045× |          1.004× |
| no-pilot       | rgb    |           1.009× |            0.993× |            0.964–1.034× |          1.013× |
| short-pilot    | plain  |           1.010× |            0.990× |            1.006–1.016× |          0.995× |
| short-pilot    | rgb    |           1.003× |            1.000× |            0.983–1.022× |          0.998× |
| bound-priority | plain  |           1.003× |            0.998× |            0.993–1.009× |          0.996× |
| bound-priority | rgb    |           1.001× |            1.001× |            0.964–1.154× |          1.000× |
| single-block   | plain  |           1.007× |            0.994× |            0.877–1.066× |          1.000× |
| single-block   | rgb    |           0.990× |            1.011× |            0.787–1.050× |          1.010× |
| no-rare-seek   | plain  |           1.004× |            0.996× |            0.985–1.024× |          0.993× |
| no-rare-seek   | rgb    |           1.014× |            0.986× |            0.989–1.047× |          1.004× |

## Control drift

| Layout | Median round-2 / round-1 QPS | Minimum cell | Maximum cell |
| ------ | ---------------------------: | -----------: | -----------: |
| plain  |                       1.010× |       0.998× |       1.036× |
| rgb    |                       1.007× |       0.984× |       1.035× |

All variants pass exhaustive count and top-100 ID/score-bit comparisons on their selected queries, plus HTTP count/top-10/top-100 ID validation in both rounds. The current control audits all 344 conjunction/phrase queries per layout. [Audit totals](audits.json) and [build/test provenance](validation.json) accompany these measurements.

These are short screening runs on one CPU architecture. No production algorithm, schema, index format, or default is changed. The bound-priority pilot nominates a bounded number of blocks; it is not a general priority traversal over aligned document intervals.
