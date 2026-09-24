# Wildcard queries

## Core API

`WildcardQuery` is added to the core query API. It matches whole indexed UTF-8 terms:
`*` matches zero or more Unicode scalar values, `?` matches one, and backslash
escapes the following character. A dangling escape is an error. Patterns are
not tokenized or stemmed. The explicit text constructor lowercases like
`PrefixQuery::text`; the raw constructor preserves case. Regex metacharacters
other than wildcard operators are literal.

The invariant is the union of matching terms' documents, with one constant 1.0
score per document. Prefix and wildcard share union materialization and scorer
execution. This is a filter query, not a sum of term BM25 scores. Existing
chunked-field rejection remains explicit until logical-document mapping is
implemented for multi-term filters. No persisted format or legacy branch is added.

Term dictionary scans belong to structures; segment readers own field-key
composition, term metadata and posting reads; queries own pattern compilation
and document unions. An initial literal prefix restricts the dictionary scan.
Leading wildcards scan the selected field's vocabulary, never stored documents.
The existing SSTable block index is not a full-vocabulary FST: efficient arbitrary
automaton traversal would require a separate measured dictionary enhancement.

Patterns are limited to 1,024 bytes and compiled-regex/DFA caches to 2 MiB each before dictionary access.
At most 1,000,000 dictionary terms are examined per segment. Matched terms and
posting expansion retain the prefix limits of 1,024 terms and 5,000,000 postings; budget exhaustion is an error, never partial results. Matching adds
bounded work per visited term, plus the existing posting-union cost. Async,
native sync and WASM use the same pattern and union implementation. Existing
prefix defaults and scoring stay unchanged. Both expanded-term filters translate
RGB physical IDs through the existing document map before returning logical IDs;
the previous prefix path omitted this mapping.

The type is exposed through the public core API, explicit query-language syntax
`field:wildcard("pattern")` or `field:foo*bar?`, and the benchmark's declared wildcard families.
A dedicated production protobuf variant is a separate interface.
[RegexQuery](regex-query.md) now supports whole-term regular expressions through
the core API and explicit query-language syntax. This does not make the 826
benchmark queries comparable: analyzer rebuilding, sloppy-phrase matching and
escaped query syntax are still required. Broad patterns may exhaust explicit
expansion budgets, as prefixes already do.

## Validation

Use a raw-text fixture to test full-term versus substring matching, interior and
leading stars, Unicode `?`, escaped operators, empty/no-match patterns, overlap
and exact counts, Boolean composition, and native/async equivalence. Compare a
trailing-star wildcard with the existing prefix query. Exercise scan and match
budget boundaries without constructing large corpora. Native checks and the
portable build must pass; runtime matching never scans or rewrites index metadata.

```rust
use hermes_core::WildcardQuery;
let query = WildcardQuery::text(body_field, "foo*bar?")?;
```

The benchmark HTTP adapter accepts `wildcard`, `wildcard_scan` and
`wildcard_lead` through this API. Its regex family now uses the separate `RegexQuery`.
The query-string API accepts `field:wildcard("pattern")` or unqualified
`wildcard("pattern")` over the default fields. Pattern arguments use JSON string
escaping, so a literal star is written as `wildcard("a\\*b")`. Bare patterns are
consumed as one query: `th*e` is no longer split into prefix `th*` OR term `e`.
A single trailing star retains `PrefixQuery` execution. No full-corpus
wildcard throughput or count agreement is claimed yet.

The initial post-wildcard capability check submitted all 826 published expressions to the real
HTTP adapter over a four-document fixture: **801 accepted, 25 explicit errors**
(13 regex and 12 escaped-query syntax cases). All 145 wildcard expressions are
accepted. This fixture deliberately does not establish 10M-corpus count agreement
or test full-vocabulary expansion budgets; the throughput gate remains 15/826.
Raw responses and binary/query hashes are retained in
`.context/yonik-benchmark/gap/wildcard-http-capabilities-826.json`.
The native-only preflight separately accepts 730 expressions; its 83 syntax
rejections include 71 sloppy phrases handled by the HTTP adapter's existing
`PhraseQuery` translation, plus those same 12 escaped expressions.

The subsequent [regex and escaped-literal follow-up](regex-query.md) accepts
826/826 expressions on the same small fixture. This does not remove expansion
limits or establish full-corpus parity.
