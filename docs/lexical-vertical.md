# Lexical vertical: positions, pruning, reordering, tokenization

Status: research and design (2026-09-03). Nothing here is implemented yet;
the sequencing at the end is the proposed order. Companion documents:
`dynamic-tokenizer-and-phrase.md`, `chunked-text-fields.md`,
`bmp-grid-compression.md`, `merge-time-reorder.md`, `budgeted-reorder.md`,
`hot-metadata-pinning.md`, `seismic-research.md`, and in azeroth
`docs/design-docs/2026-09-03-bm25-phrase-retrieval.md` and
`2026-09-03-fulltext-billion-scale-research.md`.

## Why a separate vertical

The sparse vertical (SPLADE over BMP V19) and the lexical vertical (BM25 over
stemmed tokens) share executors but not data shapes:

|                              | SPLADE sparse                                                       | BM25 lexical                                                        |
| ---------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| vocabulary                   | 105,879 fixed dims                                                  | open, tens of millions of stems and identifiers per corpus, Zipfian |
| non-zeros per unit           | learned, roughly 100–300 per chunk, bounded by FLOPS regularisation | every token of the chunk; df ranges from 1 to N                     |
| weight                       | learned float, quantised u8                                         | integer tf plus a length norm; score needs k1, b, avg length        |
| positions                    | none                                                                | required (phrases, proximity, highlighting)                         |
| forward grid (dims × blocks) | feasible (`bmp_grid_compression.md`)                                | impossible at a 50M-term vocabulary                                 |
| pruning unit                 | block and superblock maxima over a dense grid                       | per-list block maxima on doc-ordered postings (MaxScore, BMW)       |
| query                        | 30–100 weighted dims, no phrases                                    | 2–15 tokens, quotes, per-query tokenizer hint                       |

Everything in this document is therefore inverted-list based and doc-ordered,
and reuses the sparse machinery only where the shape allows it (executor,
codecs, BP, pinning, cold IO, merge streaming).

## Audit of the text path today

Facts, with the file that establishes each one.

- Doc postings (`structures/postings/posting.rs`): 128-posting blocks, header
  `[count u16][first_doc u32][doc_bits u8][tf_bits u8]`, rounded-width bit
  packing of doc deltas and tfs, L0 skip 16 B per block `(first_doc,
last_doc, offset, max_weight f32 = block max tf)`, L1 `last_doc` per 8
  blocks, 24 B footer. Read zero-copy from the mmap (`deserialize_zero_copy`),
  blocks decoded lazily with deferred tf decode. Good.
- Block-Max MaxScore for text exists (`query/scoring.rs`, `TermCursor::text`,
  `current_block_max_score`), with conjunction skipping and block skipping;
  the upper bound is `bm25_upper_bound(max_tf, idf)` with the shortest
  possible length (`1 - b`). Since chunked fields (`chunked-text-fields.md`)
  MaxScore stays on when positions are requested.
- Positions (`structures/postings/positions.rs`): a separate list per term,
  blocks of 128 documents, block prefix `count u32 + first_doc u32`, then per
  document a vint doc delta, a vint position count and **absolute vint
  positions** encoded `(ordinal << 20) | token_position`. Skip entries are
  20 B per block. `SegmentReader::get_positions` reads the term's whole
  position range and `PositionPostingList::deserialize` copies it into the
  heap; `get_positions_into` then linearly vint-decodes a 128-document block
  to reach one document. Every phrase term pays a whole-list read and copy.
- Phrase (`query/phrase.rs`): conjunction over doc postings, then position
  check with `expected_pos += 1` per term (no gaps), score =
  `BM25(sum of term tfs) × 1.5` (not phrase frequency). One term collapses
  to `TermQuery`; no positions collapse to a MUST of terms.
- Server conversion (`hermes-server/src/converters.rs::field_tokens`): phrase
  text is tokenized with the field tokenizer and hint, `Token.position` is
  discarded, so the query side cannot express gaps.
- Tokenizer (`tokenizer/mod.rs`): `tokenize_and_clean` splits on whitespace,
  strips every non-alphanumeric character, lower-cases; `DynamicStemmer`
  routes each token to the first hinted Snowball language of its script
  (Latin, Cyrillic, Greek, Arabic, Tamil). `StopWordTokenizer` exists as a
  wrapper but is not reachable from a schema spec: `TokenizerSpec::parse`
  accepts only `by` and `default`. CJK text is not segmented; hyphenated
  identifiers collapse (`float-zero` → `floatzero`); no diacritic folding.
- Lengths: chunked fields persist real chunk lengths (`.chunks`); non-chunked
  text fields persist only the segment average (`SegmentMeta::avg_field_len`)
  and score with tf as the length.
- Term dictionary (`structures/sstable.rs`): `TermInfo::Inline` for df ≤ 3
  without positions, otherwise `External { posting_offset, posting_len,
doc_freq, position_offset, position_len }`. FST or raw mmap index. Not in
  `PinPolicy`.
- Reordering (`segment/reorder.rs`, `merge-time-reorder.md`): BP permutes
  only BMP sparse blobs through their own doc maps; postings, positions,
  store and fast fields are copied unchanged, so text postings are in
  insertion order.
- IDF (`query/global_stats.rs`): lazily aggregated over the segments of one
  searcher; no cross-shard statistics (broker phase 2 merges by RRF).

## Position list format v2

Status: implemented (2026-09-03), `structures/postings/positions_v2.rs`.

Goal: positions cost bytes only where they exist, and a phrase query touches
only the positions of documents that survived the doc-level conjunction.

### Layout

One position stream per term, addressed through the doc postings instead of
through its own skip list:

```text
.pos  per term:  [block 0]...[block n-1][block offsets: u32 × n]
                 [footer: n u32, total_positions u64, magic "POS2"]
block:           [count u16][bits u8][pad u8][packed values: count × bytes(bits)]
                 128 values per block, the last one partial
```

- Values are **deltas**: a document's positions are sorted, the first is
  stored as is, the rest relative to the previous one. For a chunked field
  positions restart per chunk and the virtual id already carries the
  ordinal, so there are no ordinal bits. A non-chunked multi-valued field
  keeps the `(ordinal << 20) | position` values; the ordinal step is one
  large delta that widens that block only.
- The packer is the rounded-width codec of the doc blocks (`simd::
pack_rounded`, 0/8/16/32 bits), so decode is the same SIMD widening.
  OptPFD-style exception packing would shave another 30–40% and is the
  next codec step once sizes are measured on a real generation.
- Doc postings gain one `u64` cursor per L0 block: the number of values
  before the block's first document (cumulative tf). Inside a block the
  position of document _j_ is `cursor + Σ tf[0..j]`, which the iterator
  keeps as a running prefix (`BlockPostingIterator::position_cursor`).
  `cursor / 128` is the block, `cursor % 128` the offset in it.
- The doc-posting footer grew by `total_positions u64 + flags u32 + magic
u32` ("BPL2"); a list ending in the magic has the extension, a legacy
  list ends with its `max_tf` (≤ 65 535 by the builder's u16 tf), so both
  stay readable. `TermInfo` is unchanged.

### Query execution

`PhraseScorer` runs the conjunction over doc postings as before and, for a
candidate, asks each term's `TermPositions` for
`[cursor, cursor + tf)`: one or two blocks decoded from the mmap into a
reused scratch buffer, delta-summed into the term's buffer, then the
offset-aware sliding match. Nothing is copied to the heap up front.
`TermPositions::Legacy` wraps the pre-v2 list for segments written before
the change. Still to do: `MADV_RANDOM` on `.pos` and a bounded `WILLNEED`
on the candidate's ranges.

### Size

With stop words removed and per-chunk restarts, in-document deltas are
roughly `chunk_len / tf`, so most blocks use 8-bit values: about 1 byte per
position plus 4 bytes per 128, against 2–3 bytes of absolute vint plus a
count per document and 20 B of skip entry per 128 documents before. The
`u64` cursor costs 8 B per 128 postings.

### Merge

Values are doc-agnostic, so a merge re-packs every source's blocks into one
fresh stream (`PositionStreamEncoder::push_values`), and the block copy of
the doc postings shifts each source's cursors by the values of the sources
before it (`Footer::total_positions`). A legacy source is decoded per
document and re-encoded, and its doc postings take the decode-and-rebuild
path so the merged term always addresses a v2 stream.

### Compatibility

No production index carries positioned text fields yet; a legacy positioned
segment is still readable and is converted by its first merge.

## Doc postings and skip metadata

- **Block bounds with real lengths** (implemented 2026-09-03). The fourth
  L0 word packs `(max_tf u16, min_len u16)` and the footer carries the
  list minimum (`FLAG_LEN_BOUNDS`), so a bound is
  `bm25(max_tf, idf, min_len, avg)` instead of the `1 - b` floor; k1/b stay
  a query-time choice. The builder derives `min_len` from chunk lengths or
  the document norms; merges copy the words and take the minimum; a list
  rebuilt without lengths stores 1, which every real unit satisfies.
  Legacy lists keep their `f32` word and the old bound. A cursor uses the
  stored minimum only when it also scores with real lengths: a `tf`-as-
  length score is not bounded by a real-length bound.
- **Norms for plain fields** (implemented 2026-09-03). `.chunks` version 2
  adds per-field sections of kind 1: one `u16` token count per document of
  the segment (0 = no value), written by the builder for every plain
  indexed text field with tokens and concatenated on merge with zero fill
  for sources lacking the column. `short_document` now scores with its
  real length in MaxScore, `TermScorer` and `PhraseScorer`; multi-valued
  fields sum their values' lengths.
- **L1 superblock maxima.** The L1 entry (one per 8 blocks) gains `max_tf`
  and `min_len`, giving two-level block-max skipping (Mallia & Porciani,
  "Faster BlockMax WAND with Longer Skipping", ECIR 2019). This is the text
  analogue of the BMP D/E hierarchy and costs 4 B per 1,024 postings.
- **Codec.** Keep 128-block OptPFD by default; after BP reordering (below)
  measure partitioned Elias-Fano (`structures/postings/partitioned_ef.rs`)
  for lists longer than a few hundred thousand postings, where PEF wins on
  clustered gaps (Ottaviano & Venturini, SIGIR 2014; Pibiri & Venturini,
  ACM CSUR 2020 survey).
- **Inline positions.** Terms with df ≤ 3 are external today only because
  positions force it; allow inline positions when the packed bytes fit the
  16 B inline slot (most tail identifiers), saving a `.pos` read for the
  rare-token queries the needle body bucket is made of.

## Pruning

- Keep Block-Max MaxScore as the rank-safe default. Fixed on the way
  (2026-09-03): the block-skip branch moved every cursor at the minimum
  document to its next block even when another essential cursor still held
  a document inside that block, so that document lost the skipped cursors'
  scores and could drop out of the top-k (regression test
  `block_max_skip_never_jumps_over_another_essential_cursor`; the same loop
  serves sparse MaxScore). The skip is now bounded by the next essential
  document, the Block-Max MaxScore rule. Still open: the L1 skip in the same
  branch (skip 8 blocks at once when the superblock bound fails).
- **Filters and phrases as executor predicates** (implemented 2026-09-03).
  The query shape clients send is `MUST [phrases, filters] + SHOULD
[terms]`. When every SHOULD clause is a text term and the MUST/MUST_NOT
  clauses combine into one document bitset (`build_combined_bitset`:
  narrowest clause materialised, the rest probed; `PhraseQuery::as_doc_bitset`
  drains the phrase, resolving chunk ids to documents), the text MaxScore
  executors run with the bitset as their predicate (`finish_text_maxscore`,
  `finish_chunked_text_maxscore`), one group per field. The top-k is exact
  over the filtered documents because a filter never changes a bound, and
  documents matching only the filters fill the tail with score 0
  (`BitsetFillScorer`) when fewer than `limit` scored documents survive.
  Before this, text SHOULD went through an over-fetched unfiltered top-k and
  a `PredicatedScorer`, which could lose phrase matches that ranked below the
  candidate budget. A chunked phrase now keeps every matching document
  instead of its top `limit`, which a MUST constraint requires.
- **Phrase as a scoring clause.** Still open: a `PhraseQuery` in SHOULD is a
  verifier; its block bound would be the minimum of its terms' bounds.
- **Approximate and anytime modes.** Reuse the sparse `heap_factor`
  threshold scaling for text, and add a postings-or-time budget that
  evaluates BP-clustered virtual-id ranges in order of their superblock bound and
  stops at the deadline (Mackenzie, Petri, Moffat, "Anytime Ranking on
  Document-Ordered Indexes", TOIS 2022). The ~4–10% corpus budget in
  arXiv:2608.00229 (Hierarchical BM25) recovers 0.85–0.92 of the exhaustive
  score on a topically ordered corpus; that is the same knob without a
  second cluster index.
- **Long queries.** Beyond ~24 terms MaxScore's essential set collapses;
  cap by idf-ranked static query pruning and, if measured to matter, run
  Block-Max WAND for the residual (Ding & Suel, SIGIR 2011; variable-sized
  blocks, Mallia et al., SIGIR 2017).
- Query term de-duplication with query tf weighting, and per-term boosts
  from the request (needed by the search-api field weighting).

## Field-level reordering

There is no document-level permutation in Hermes and this design keeps it
that way: doc ids, the store, the fast fields and the dense vector maps
never move. A field that wants locality owns its permutation and a map back
to `(doc_id, ordinal)`, exactly as a BMP sparse field does today
(`merge-time-reorder.md`, `block-level-reorder.md`).

For text the map already exists. A chunked text field keys postings and
positions by a field-local virtual id and resolves hits through `.chunks`
(`chunked-text-fields.md`). Today virtual ids are assigned in indexing
order, which makes `doc_ids` in the map non-decreasing; nothing in query
execution depends on that order (`ChunkMap::resolve`, `length` are array
lookups). Reordering a text field therefore means:

- Schema: the existing `reorder` attribute on a chunked text field,
  `field content: text<...> [indexed<chunked, token_position>, reorder]`.
  Fields without it keep indexing order and block-copy on merge, as sparse
  fields without the attribute do.
- Objective: BP (`graph_bisection`, `BpBudget`, optimizer tiering) over the
  field's own postings, terms restricted to a mid-df band. Each reordered
  field computes its own order; several text or sparse fields never have
  to agree.
- Output: postings re-encoded in the permuted virtual-id order (the same
  rewrite a BP pass does for a BMP blob), `.chunks` written in that order
  with `(doc_id, ordinal, length)` per virtual id, and the position stream
  re-packed with each chunk's run of values moved to its new place. The
  runs are intra-chunk deltas, so no delta decoding is needed, but the
  128-value blocks and the cursors are rebuilt.
- Merge: with `reorder_on_merge`, BP runs inside the merge over the
  concatenated field; otherwise virtual ids are concatenated with per-source
  offsets as now.
- Predicates: doc-id filters resolve through the map before the check, the
  path the predicate-aware sparse MaxScore executor already takes; the
  chunked text executor gets the same hook. Fusion keys on
  `(segment, doc, ordinal)` after resolve and is unaffected.
- `short_document` becomes `chunked` too (one value per document), so it
  gains real lengths, ordinals and the same reorder path; non-chunked
  positioned fields stay in doc-id order and are not reorderable.

Expected gains per reordered field (Dhulipala et al., KDD 2016; Mackenzie
et al., ECIR 2019 reproducibility): 10–30% smaller delta-coded postings,
tighter block maxima, superblock skipping that actually fires. The cost is
the map lookup per hit that chunked fields already pay.

## Tokenization, stemming, stop words

- **`stop_words` spec parameter (blocker).** `TokenizerSpec::parse` must
  accept `stop_words: true|false`, `DynamicStemmer` must drop the language's
  stop words after cleaning and before stemming (list chosen by the same
  language routing as the stemmer, `stop_words` crate as today), and the
  surviving tokens must **keep their original positions** (Tantivy
  semantics). Azeroth's templates already declare it and are blocked on it.
- **Gap-aware phrases.** `PhraseQuery` carries `(offset, term)` pairs; the
  server fills offsets from `Token.position` of the hinted tokenization, so
  `"state of the art"` becomes `state@0 art@3` and must not match
  `state art`. A phrase whose tokens are all stop words yields no terms:
  degrade to the plain match of the raw words and report it in the response
  diagnostics rather than failing the request.
- **Word segmentation, CJK, folding** (implemented 2026-09-03 as
  `segmenter: unicode` of the `stem(...)` spec; `simple` stays the default
  so existing fields keep their tokenization). UAX #29 word boundaries
  (`unicode-segmentation`): `float`@0 `zero`@1 for `float-zero`, digits
  stay inside tokens (`p53`, `co2`, `10.1007`), punctuation inside a word
  is stripped. Runs of Han, Hiragana and Katakana become character bigrams
  with one position each (Lucene `CJKBigramFilter` semantics; a lone
  ideograph stays a unigram); Hangul and Thai are left to UAX #29. NFKD plus
  removal of combining marks for Latin, Cyrillic and Greek tokens, applied
  after stemming so Snowball and the stop lists see the original letters.
  The same segmenter runs at query time.
- **Stemming.** Snowball stays (18 languages, script-routed). Lemmatisation
  is worth it only for morphologically rich languages (Czech, Estonian,
  Finnish) and needs dictionaries; defer. Decompounding for German, Dutch,
  Finnish, Swedish likewise deferred. Do not index both surface and stem at
  the same position (Lucene keyword-repeat): it doubles postings; prefer a
  query-time exact-form boost if precision on surface forms is ever needed.
- **Numbers and identifiers.** Never stem tokens containing digits; keep
  DOIs, arXiv ids and gene names as single tokens under UAX #29 rules with
  the `.` and `/` inside identifiers treated as non-breaking when between
  alphanumerics.

## Scoring

- BM25 constants become per-field schema options (`k1`, `b`) with the Lucene
  formulation as default (Kamphuis et al., "Which BM25 Do You Mean?", ECIR
  2020: the variants are practically equivalent, so keep the common one).
- Phrase score = BM25 over the **phrase frequency** (number of matched
  occurrences) with the phrase's summed idf, not `1.5 × BM25(Σ tf)`
  (implemented 2026-09-03).
- **Proximity rescoring.** Optional second stage over the top-k of the
  MaxScore pass: sequential-dependence ordered and unordered window counts
  for adjacent query term pairs (Metzler & Croft, SIGIR 2005; proximity
  BM25 variants, Tao & Zhai 2007) computed from the now-cheap positions.
  A `proximity_weight` on `MatchQuery`, default 0.
- **Fields.** `short_document` and `content` are fused by RRF today. Inside
  the lexical channel a per-field boost with a doc-level max/sum
  (`bm25f_score` already exists) is the better combination once norms exist
  for `short_document`.
- Global IDF for chunked fields must use chunk counts consistently across
  segments (`LazyGlobalStats::text_idf` uses document totals; verify before
  the first multi-segment chunked index).

## Statistics, sharding, residency

- Cross-shard IDF: a per-query DF exchange in broker phase 2, or a periodic
  merged DF table for terms above a df threshold with per-shard fallback
  (the "global DF table" fix of arXiv:2608.00229).
- `PinPolicy`: pin the term dictionary index (FST or raw index) and the
  L0/L1 skip sections of terms above a df threshold; `.pos` stays evictable
  with random-access advice.
- Segment size: virtual ids and doc ids are u32 per segment; `pos_cursor`
  is u64 so one term may exceed 4G positions in a segment.

## What is reused from the sparse vertical

| sparse component                                                      | lexical use                                  |
| --------------------------------------------------------------------- | -------------------------------------------- |
| `MaxScoreExecutor` / `TermCursor`                                     | unchanged; gains phrase cursors and L1 skips |
| OptPFD / rounded-width packers, SIMD unpack                           | doc deltas, tfs, and now positions           |
| `graph_bisection`, `BpBudget`, optimizer tiering, `reorder` attribute | per-field permutation over the chunk map     |
| `PinPolicy`, cold-IO writers, `MADV_*` helpers                        | term index, skips, `.pos`                    |
| streaming merge with offset rebasing                                  | pos stream concatenation                     |
| `heap_factor` approximate threshold                                   | approximate BM25                             |
| chunk maps and ordinal fusion keys                                    | unchanged                                    |

Not reused: BMP grids, LSP hierarchy, forward index, Seismic-style
per-list clustering (all need a bounded vocabulary).

## Sequencing

1. `stop_words` spec + gap-preserving positions + `PhraseQuery` offsets.
   Small; unblocks azeroth's templates.
2. Position format v2 + lazy phrase scorer, `(max_tf, min_len)` block
   bounds, norms for plain fields (all done). Must precede the first
   text-bearing index generation.
3. UAX #29 tokenizer, folding, CJK bigrams (done, opt-in `segmenter:
unicode`).
4. Filters and phrases as executor predicates, phrase-frequency scoring
   (done); L1 superblock maxima, proximity rescoring, per-field k1/b (open).
5. Field-level BP reordering of chunked text fields through their chunk maps.
6. Anytime/budgeted BM25 and long-query handling.
7. Cross-shard DF with broker phase 2.

## Evaluation

- Rank-safety differential tests: MaxScore vs exhaustive Boolean on
  synthetic Zipfian corpora with real lengths, for every pruning change.
- Phrase semantics tests: gaps, stop-word-only phrases, chunk boundaries,
  multi-valued ordinals, slop.
- Size accounting per file (`hermes-tool diagnose`): `.post`, `.pos`,
  `.terms`, `.chunks` bytes per document before and after each step.
- Latency by query length (1, 2, 4, 8, 16, 32 terms) and by phrase count,
  cold and warm cache.
- azeroth `needle scholar` (`body` bucket) after each index-time change.

## References

- Deshpande & Sundararaman, Hierarchical BM25, arXiv:2608.00229 (2026).
- Ding & Suel, Faster top-k document retrieval using block-max indexes,
  SIGIR 2011. Mallia et al., Faster BlockMax WAND with variable-sized
  blocks, SIGIR 2017. Mallia & Porciani, Faster BlockMax WAND with longer
  skipping, ECIR 2019.
- Mackenzie, Petri, Moffat, Anytime ranking on document-ordered indexes,
  TOIS 2022 (arXiv:2104.08976).
- Dhulipala et al., Compressing graphs and indexes with recursive graph
  bisection, KDD 2016; Mackenzie et al., reproducibility, ECIR 2019.
- Ottaviano & Venturini, Partitioned Elias-Fano indexes, SIGIR 2014;
  Pibiri & Venturini, Techniques for inverted index compression, ACM CSUR
  2020; Yan, Ding, Suel, Inverted index compression and query processing
  with optimized document ordering, WWW 2009 (OptPFD, positions).
- Metzler & Croft, A Markov random field model for term dependencies,
  SIGIR 2005; Tao & Zhai, An exploration of proximity measures in IR,
  SIGIR 2007.
- Kamphuis, de Vries, Boytsov, Lin, Which BM25 do you mean?, ECIR 2020.
- Mallia, Suel, Tonellotto, Faster learned sparse retrieval with Block-Max
  Pruning, SIGIR 2024; Carlson et al., Dynamic superblock pruning, SIGIR
  2025; Bruch et al., Seismic, SIGIR 2024 (sparse-side context only).
- Williams, Zobel, Bahle, Fast phrase querying with combined indexes, TOIS
  2004 (next-word indexes; not planned while stop words are dropped).
