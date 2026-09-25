# ARM traversal controls

Same 32,768-document fixture, release compiler and native flags; A/B/B/A order.
Each run uses 30 Criterion samples, one second warmup and two seconds measurement.
Fixture construction and exhaustive result checks occur outside timing.
These controls do not establish a corpus-wide result. Full intervals and process memory are in [arm.json](arm.json).

| Layout | Query  |   k | Before µs | After µs | Speedup |
| ------ | ------ | --: | --------: | -------: | ------: |
| early  | and    |  10 |      9.56 |     9.49 |   1.01× |
| early  | and    | 100 |     12.08 |    11.99 |   1.01× |
| early  | filter |  10 |    193.67 |     4.65 |  41.61× |
| early  | filter | 100 |    194.25 |     6.01 |  32.32× |
| early  | term   |  10 |      9.31 |     5.31 |   1.75× |
| early  | term   | 100 |     11.80 |     6.94 |   1.70× |
| late   | and    |  10 |    290.12 |   287.01 |   1.01× |
| late   | and    | 100 |    293.84 |   291.31 |   1.01× |
| late   | filter |  10 |    193.23 |     4.60 |  42.02× |
| late   | filter | 100 |    193.82 |     6.03 |  32.17× |
| late   | term   |  10 |    177.17 |   128.01 |   1.38× |
| late   | term   | 100 |    180.18 |   130.18 |   1.38× |
| mixed  | and    |  10 |    145.56 |   145.02 |   1.00× |
| mixed  | and    | 100 |    262.11 |   259.90 |   1.01× |
| mixed  | filter |  10 |    194.18 |     4.63 |  41.95× |
| mixed  | filter | 100 |    193.88 |     6.03 |  32.18× |
| mixed  | term   |  10 |     69.69 |    55.94 |   1.25× |
| mixed  | term   | 100 |    156.20 |   116.22 |   1.34× |

Memory is measured for the entire benchmark process, including fixture construction and Criterion.
Maximum RSS ranges overlap: 96.4–108.4 MiB before and 100.7–105.7 MiB after.
Peak physical footprint is 29.5–29.7 MiB before and 31.7–33.6 MiB after.
No claim of reduced process memory follows from these samples.
