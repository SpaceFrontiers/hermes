//! The dynamic text tokenizer: segmentation, normalisation, stemming and
//! same-position variants, declared in SDL as
//!
//! ```text
//! text<stem(by: <field>, default: <language|simple>, stop_words: <bool>,
//!           segmenter: <simple|unicode|icu>, stem: <none|light|snowball>,
//!           keep_original: <bool>, fold: <bool>, max_token_length: <n>,
//!           t2s: <bool>, morph: <bool>)>
//! ```
//!
//! Three layers, identical at index and query time:
//!
//! 1. **Segmentation and normalisation** (language-agnostic). `simple` splits
//!    on whitespace and strips punctuation; `unicode` uses UAX #29 word
//!    boundaries with character bigrams over CJK runs; `icu` uses ICU4X's
//!    word segmenter (dictionary words for Chinese and Japanese, LSTM word
//!    breaks for Thai, Lao, Khmer and Burmese, UAX #29 elsewhere), keeping
//!    the bigram fallback for characters the dictionary does not know and
//!    emitting the bigrams of every dictionary word as same-position
//!    variants so a differently segmented query still matches. Every token
//!    is NFKC-normalised and lowercased; Arabic tokens get the Lucene
//!    orthographic normalisation, Cyrillic `ё` becomes `е`. Tokens longer
//!    than `max_token_length` characters are dropped (position kept).
//! 2. **Morphology** per language, routed by the token's script to the first
//!    hinted language of that script (`by` reads the hint from a sibling
//!    field at index time, the query passes `tokenizer_hint`): `light`
//!    strips inflection only (see [`super::light_stem`]), `snowball` is the
//!    full Snowball algorithm, `none` keeps the word. With `keep_original`
//!    the stem is indexed as a **variant at the same position** as the
//!    original, so phrases and exact terms match the original while match
//!    queries use the stem; without it the stem replaces the word.
//! 3. **Folding**: with `fold`, the diacritic-free form of Latin, Cyrillic
//!    and Greek tokens is a same-position variant (with `keep_original`) or
//!    replaces the token (without).
//!
//! Query tokenization ([`super::Tokenizer::tokenize_query`]) emits one form
//! per word: the stem for a match query, the original for a phrase or exact
//! term, never variants. Variants are marked [`super::Token::variant`] and
//! do not count towards field length.

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use super::{
    Language, Script, Token, Tokenizer, clean_word, language_code, parse_language_opt,
    split_whitespace_with_offsets, with_stemmers,
};
use super::{cjk_morph, light_stem};

/// Default `max_token_length`: longer "words" are hashes, sequences and
/// URLs, which no query types and which bloat the dictionary.
pub const DEFAULT_MAX_TOKEN_LENGTH: usize = 64;

/// Word segmentation of a [`DynamicStemmer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Segmenter {
    /// Split on whitespace, strip every non-alphanumeric character
    /// (`float-zero` → `floatzero`, no CJK segmentation).
    #[default]
    Simple,
    /// UAX #29 word boundaries (`float-zero` → `float`, `zero`; `p53`,
    /// `co2` and `10.1007` stay whole) and character bigrams over runs of
    /// Han, Hiragana and Katakana.
    Unicode,
    /// ICU4X word segmentation: dictionary words for Chinese and Japanese
    /// (with their bigrams as variants), LSTM word breaks for Thai, Lao,
    /// Khmer and Burmese, UAX #29 for everything else.
    Icu,
}

/// Morphological normalisation of a [`DynamicStemmer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StemMode {
    /// Keep every word as written.
    None,
    /// Inflection only (plural, case, gender), Lucene's light stemmers.
    Light,
    /// Full Snowball stemming.
    #[default]
    Snowball,
}

/// What a tokenization is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Emit {
    /// Indexing: originals plus variants.
    Index,
    /// Match query: one form per word, the stem when the language is known.
    Match,
    /// Phrase or exact term: the original form.
    Exact,
}

/// Stemmer whose language is selected per call from the tokenizer hint.
///
/// The hint is a comma-separated list of language codes or names (`"ru,en"`).
/// Each token is stemmed with the first hinted language whose script matches
/// the token's script; tokens of a script no hinted language covers are kept
/// as cleaned (lowercased, punctuation stripped) text. Without a hint the
/// `default` language applies, or plain cleaning when it is `None`.
///
/// With `stop_words` enabled, the stop words of the language a token is
/// routed to are dropped before stemming. Dropped words keep consuming
/// positions, so phrase distances are preserved (`"quantum of the art"`
/// becomes `quantum@0 art@3`).
#[derive(Debug, Clone)]
pub struct DynamicStemmer {
    default: Option<Language>,
    stop_words: bool,
    segmenter: Segmenter,
    /// A spec without `by` has no per-document language and treats query
    /// hints the same way: every text goes through `default`.
    ignore_hints: bool,
    stem: StemMode,
    keep_original: bool,
    fold: bool,
    max_token_length: usize,
    /// Convert Han tokens to simplified Chinese (OpenCC `t2s`), index and
    /// query alike, so traditional and simplified spellings meet.
    t2s: bool,
    /// Japanese and Korean morphology through dictionaries (`cjk-dict`
    /// feature): Korean text always, Japanese text when hinted `ja` (kana
    /// runs regardless); content morphemes carry their base form as a
    /// variant, particles and endings are dropped like stop words.
    morph: bool,
}

impl Default for DynamicStemmer {
    fn default() -> Self {
        Self::new(None)
    }
}

impl DynamicStemmer {
    /// Create a dynamic stemmer; `default` applies when no hint is present.
    /// Stop words are kept, words are split by [`Segmenter::Simple`], stems
    /// are Snowball and replace the word, folding is on, tokens longer than
    /// [`DEFAULT_MAX_TOKEN_LENGTH`] are dropped.
    pub fn new(default: Option<Language>) -> Self {
        Self {
            default,
            stop_words: false,
            segmenter: Segmenter::Simple,
            ignore_hints: false,
            stem: StemMode::Snowball,
            keep_original: false,
            fold: true,
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
            t2s: false,
            morph: false,
        }
    }

    /// Japanese and Korean dictionary morphology (see [`cjk_morph`]).
    pub fn with_morph(mut self, enabled: bool) -> Self {
        self.morph = enabled;
        self
    }

    /// Whether Japanese and Korean dictionary morphology is on.
    pub fn morphs_cjk(&self) -> bool {
        self.morph
    }

    /// Convert Han tokens to simplified Chinese (index and query alike),
    /// character by character with OpenCC's table.
    pub fn with_t2s(mut self, enabled: bool) -> Self {
        self.t2s = enabled;
        self
    }

    /// Whether Han tokens are converted to simplified Chinese.
    pub fn converts_to_simplified(&self) -> bool {
        self.t2s
    }

    /// Ignore language hints (index-time field values and query hints
    /// alike): the tokenizer always applies `default`. This is the
    /// `stem(...)` spec without `by`.
    pub fn with_ignored_hints(mut self, ignore: bool) -> Self {
        self.ignore_hints = ignore;
        self
    }

    /// Whether hints are ignored (see [`DynamicStemmer::with_ignored_hints`]).
    pub fn ignores_hints(&self) -> bool {
        self.ignore_hints
    }

    /// Choose the word segmentation (see [`Segmenter`]).
    pub fn with_segmenter(mut self, segmenter: Segmenter) -> Self {
        self.segmenter = segmenter;
        self
    }

    /// The word segmentation in use.
    pub fn segmenter(&self) -> Segmenter {
        self.segmenter
    }

    /// Drop the routed language's stop words (positions are preserved).
    pub fn with_stop_words(mut self, enabled: bool) -> Self {
        self.stop_words = enabled;
        self
    }

    /// Whether stop words are dropped.
    pub fn strips_stop_words(&self) -> bool {
        self.stop_words
    }

    /// Choose the morphological normalisation (see [`StemMode`]).
    pub fn with_stem(mut self, stem: StemMode) -> Self {
        self.stem = stem;
        self
    }

    /// The morphological normalisation in use.
    pub fn stem_mode(&self) -> StemMode {
        self.stem
    }

    /// Index stems and folded forms as same-position variants of the
    /// original word instead of replacing it.
    pub fn with_keep_original(mut self, keep: bool) -> Self {
        self.keep_original = keep;
        self
    }

    /// Whether originals are kept next to their variants.
    pub fn keeps_original(&self) -> bool {
        self.keep_original
    }

    /// Fold diacritics of Latin, Cyrillic and Greek tokens.
    pub fn with_fold(mut self, fold: bool) -> Self {
        self.fold = fold;
        self
    }

    /// Whether diacritics are folded.
    pub fn folds(&self) -> bool {
        self.fold
    }

    /// Drop tokens longer than `chars` characters (0 = unlimited).
    pub fn with_max_token_length(mut self, chars: usize) -> Self {
        self.max_token_length = chars;
        self
    }

    /// The token length limit in characters (0 = unlimited).
    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }

    /// Default language used when no hint is given.
    pub fn default_language(&self) -> Option<Language> {
        self.default
    }

    /// Parse a hint into the ordered list of recognised languages.
    pub fn parse_hint(hint: &str) -> Vec<Language> {
        let mut languages = Vec::new();
        for part in hint.split(',') {
            if let Some(language) = parse_language_opt(part)
                && !languages.contains(&language)
            {
                languages.push(language);
            }
        }
        languages
    }

    /// Whether the hint names Japanese (`ja`) or Korean (`ko`); these are
    /// not Snowball languages, so they ride next to `languages_for`.
    fn cjk_hints_for(&self, hint: Option<&str>) -> CjkHints {
        let mut hints = CjkHints::default();
        if self.ignore_hints {
            return hints;
        }
        if let Some(hint) = hint {
            for part in hint.split(',') {
                match part.trim().to_ascii_lowercase().as_str() {
                    "ja" | "jpn" | "japanese" => hints.japanese = true,
                    "ko" | "kor" | "korean" => hints.korean = true,
                    _ => {}
                }
            }
        }
        hints
    }

    fn languages_for(&self, hint: Option<&str>) -> Vec<Language> {
        if !self.ignore_hints
            && let Some(hint) = hint.map(str::trim).filter(|hint| !hint.is_empty())
        {
            let languages = Self::parse_hint(hint);
            if !languages.is_empty() {
                return languages;
            }
        }
        self.default.into_iter().collect()
    }

    fn run(&self, text: &str, languages: &[Language], cjk: CjkHints, emit: Emit) -> Vec<Token> {
        let stops: Vec<Option<&'static HashSet<String>>> = languages
            .iter()
            .map(|language| self.stop_words.then(|| stop_word_set(*language)).flatten())
            .collect();
        let snowball = languages.is_empty() || self.stem != StemMode::Snowball;
        if snowball {
            let ctx = Ctx {
                languages,
                stops: &stops,
                stemmers: &[],
                cjk,
            };
            self.walk(text, &ctx, emit)
        } else {
            with_stemmers(languages, |stemmers| {
                let ctx = Ctx {
                    languages,
                    stops: &stops,
                    stemmers,
                    cjk,
                };
                self.walk(text, &ctx, emit)
            })
        }
    }

    fn walk(&self, text: &str, ctx: &Ctx<'_>, emit: Emit) -> Vec<Token> {
        let mut emitter = Emitter {
            cfg: self,
            ctx,
            emit,
            tokens: Vec::with_capacity(text.len() / 5),
            position: 0,
            run: Vec::new(),
            run_end: 0,
        };
        match self.segmenter {
            Segmenter::Simple => {
                for (offset, word) in split_whitespace_with_offsets(text) {
                    emitter.word(offset, word);
                }
            }
            Segmenter::Unicode => {
                use unicode_segmentation::UnicodeSegmentation;
                for (offset, word) in text.unicode_word_indices() {
                    if word.chars().all(is_cjk_char) {
                        // UAX #29 yields one word per ideograph but keeps
                        // kana runs together; either way the characters
                        // join the current run when adjacent in the text.
                        emitter.cjk_chars(offset, word);
                    } else {
                        emitter.word(offset, word);
                    }
                }
            }
            Segmenter::Icu => {
                if self.morph {
                    for (start, end, kind) in morph_spans(text, ctx.cjk) {
                        match kind {
                            SpanKind::Japanese => {
                                emitter.morph_run(start, &text[start..end], cjk_morph::japanese)
                            }
                            SpanKind::Korean => {
                                emitter.morph_run(start, &text[start..end], cjk_morph::korean)
                            }
                            SpanKind::Icu => emitter.icu_span(start, &text[start..end]),
                        }
                    }
                } else {
                    emitter.icu_span(0, text);
                }
            }
        }
        emitter.flush_run();
        emitter.tokens
    }
}

impl Tokenizer for DynamicStemmer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let languages: Vec<Language> = self.default.into_iter().collect();
        self.run(text, &languages, CjkHints::default(), Emit::Index)
    }

    fn tokenize_hinted(&self, text: &str, hint: Option<&str>) -> Vec<Token> {
        let languages = self.languages_for(hint);
        self.run(text, &languages, self.cjk_hints_for(hint), Emit::Index)
    }

    fn tokenize_query(&self, text: &str, hint: Option<&str>, exact: bool) -> Vec<Token> {
        let languages = self.languages_for(hint);
        let emit = if exact { Emit::Exact } else { Emit::Match };
        self.run(text, &languages, self.cjk_hints_for(hint), emit)
    }
}

/// Per-call language context.
struct Ctx<'a> {
    languages: &'a [Language],
    stops: &'a [Option<&'static HashSet<String>>],
    /// Snowball stemmers aligned with `languages` (empty unless the mode is
    /// Snowball).
    stemmers: &'a [&'a rust_stemmers::Stemmer],
    cjk: CjkHints,
}

/// Japanese / Korean flags of a hint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CjkHints {
    japanese: bool,
    korean: bool,
}

/// How a span of text is segmented under `morph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanKind {
    Japanese,
    Korean,
    Icu,
}

#[inline]
fn is_hangul(c: char) -> bool {
    matches!(c as u32, 0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xD7B0..=0xD7FF)
}

#[inline]
fn is_kana(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F)
}

/// Split `text` into maximal spans: Hangul runs go to the Korean
/// dictionary, kana runs (plus Han when the document or query is hinted
/// Japanese) to the Japanese one, everything else to ICU. Whitespace ends a
/// run, so each Japanese sentence or Korean word group is analysed whole.
fn morph_spans(text: &str, cjk: CjkHints) -> Vec<(usize, usize, SpanKind)> {
    let classify = |c: char| {
        if is_hangul(c) {
            SpanKind::Korean
        } else if is_kana(c) || (cjk.japanese && is_han_char(c)) {
            SpanKind::Japanese
        } else {
            SpanKind::Icu
        }
    };
    let mut spans: Vec<(usize, usize, SpanKind)> = Vec::new();
    for (offset, c) in text.char_indices() {
        let kind = classify(c);
        let end = offset + c.len_utf8();
        match spans.last_mut() {
            Some((_, last_end, last_kind)) if *last_kind == kind && *last_end == offset => {
                *last_end = end;
            }
            _ => spans.push((offset, end, kind)),
        }
    }
    // Merge every non-morph span with its neighbours of the same kind (the
    // loop above already does), and let ICU see runs whole.
    spans
}

struct Emitter<'a> {
    cfg: &'a DynamicStemmer,
    ctx: &'a Ctx<'a>,
    emit: Emit,
    tokens: Vec<Token>,
    position: u32,
    /// Contiguous run of single CJK characters: (byte offset, char).
    run: Vec<(usize, char)>,
    run_end: usize,
}

impl Emitter<'_> {
    /// Segment `text` (at byte `base` of the document) with ICU and emit
    /// its words.
    fn icu_span(&mut self, base: usize, text: &str) {
        let segmenter = icu_word_segmenter();
        let mut start = 0usize;
        for (end, kind) in segmenter.segment_str(text).iter_with_word_type() {
            let segment = &text[start..end];
            let offset = base + start;
            start = end;
            if !kind.is_word_like() {
                continue;
            }
            if segment.chars().all(is_cjk_char) {
                self.cjk_word(offset, segment);
            } else {
                self.word(offset, segment);
            }
        }
    }

    /// Morphemes of a Japanese or Korean run: function morphemes keep
    /// their position and are dropped; content morphemes are emitted with
    /// their base form (and, when indexing, their bigrams) as variants.
    fn morph_run(&mut self, base: usize, text: &str, analyse: fn(&str) -> Vec<cjk_morph::Morph>) {
        self.flush_run();
        for morph in analyse(text) {
            if !morph.content {
                self.position += 1;
                continue;
            }
            use unicode_normalization::UnicodeNormalization;
            let surface: String = morph.surface.nfkc().collect();
            if self.cfg.max_token_length > 0 && surface.chars().count() > self.cfg.max_token_length
            {
                self.position += 1;
                continue;
            }
            let (from, to) = (base + morph.start, base + morph.end);
            let position = self.position;
            match self.emit {
                Emit::Index => {
                    let chars: Vec<char> = surface.chars().collect();
                    self.tokens
                        .push(Token::new(surface.clone(), position, from, to));
                    if let Some(lemma) = morph.lemma.filter(|lemma| *lemma != surface) {
                        self.tokens.push(Token::variant(lemma, position, from, to));
                    }
                    if chars.len() >= 3 && chars.iter().all(|c| is_cjk_char(*c)) {
                        for pair in chars.windows(2) {
                            let mut bigram = String::with_capacity(6);
                            bigram.push(pair[0]);
                            bigram.push(pair[1]);
                            self.tokens.push(Token::variant(bigram, position, from, to));
                        }
                    }
                }
                Emit::Match => {
                    let form = morph.lemma.unwrap_or(surface);
                    self.tokens.push(Token::new(form, position, from, to));
                }
                Emit::Exact => {
                    self.tokens.push(Token::new(surface, position, from, to));
                }
            }
            self.position += 1;
        }
    }

    /// A word from the segmenter (non-CJK).
    fn word(&mut self, offset: usize, raw: &str) {
        self.flush_run();
        if raw.is_empty() {
            return;
        }
        let cleaned = if self.cfg.segmenter == Segmenter::Simple || raw.is_ascii() {
            clean_word(raw)
        } else {
            use unicode_normalization::UnicodeNormalization;
            let normalized: String = raw.nfkc().collect();
            clean_word(&normalized)
        };
        if cleaned.is_empty() {
            return;
        }
        self.emit_word(cleaned, offset, offset + raw.len());
    }

    /// Characters of a CJK segment that has no dictionary word boundary
    /// (UAX #29 output): they join the current run and are bigrammed.
    fn cjk_chars(&mut self, offset: usize, word: &str) {
        if !self.run.is_empty() && offset != self.run_end {
            self.flush_run();
        }
        let mut at = offset;
        for c in word.chars() {
            self.run.push((at, c));
            at += c.len_utf8();
        }
        self.run_end = at;
    }

    /// A CJK segment from the ICU dictionary segmenter: a single character
    /// joins the bigram run; a word is one token, with its bigrams as
    /// same-position variants when indexing.
    fn cjk_word(&mut self, offset: usize, word: &str) {
        let count = word.chars().count();
        if count == 1 {
            self.cjk_chars(offset, word);
            return;
        }
        self.flush_run();
        use unicode_normalization::UnicodeNormalization;
        let text: String = word.nfkc().collect();
        let text = self.simplify(text);
        let end = offset + word.len();
        let chars: Vec<char> = text.chars().collect();
        self.tokens
            .push(Token::new(text, self.position, offset, end));
        if self.emit == Emit::Index && chars.len() >= 3 {
            for pair in chars.windows(2) {
                let mut bigram = String::with_capacity(pair[0].len_utf8() + pair[1].len_utf8());
                bigram.push(pair[0]);
                bigram.push(pair[1]);
                self.tokens
                    .push(Token::variant(bigram, self.position, offset, end));
            }
        }
        self.position += 1;
    }

    fn flush_run(&mut self) {
        match self.run.len() {
            0 => {}
            1 => {
                let (offset, c) = self.run[0];
                let text = self.simplify(c.to_string());
                self.tokens.push(Token::new(
                    text,
                    self.position,
                    offset,
                    offset + c.len_utf8(),
                ));
                self.position += 1;
            }
            _ => {
                for pair in self.run.windows(2) {
                    let (start, a) = pair[0];
                    let (next, b) = pair[1];
                    let mut text = String::with_capacity(a.len_utf8() + b.len_utf8());
                    text.push(a);
                    text.push(b);
                    let text = self.simplify(text);
                    self.tokens
                        .push(Token::new(text, self.position, start, next + b.len_utf8()));
                    self.position += 1;
                }
            }
        }
        self.run.clear();
    }

    /// Traditional-to-simplified conversion of a Han token when enabled.
    fn simplify(&self, text: String) -> String {
        if self.cfg.t2s && text.chars().any(is_han_char) {
            han_to_simplified(&text)
        } else {
            text
        }
    }

    /// Route a cleaned word to its language and emit the forms the mode
    /// asks for. Every path consumes one position.
    fn emit_word(&mut self, word: String, from: usize, to: usize) {
        let cfg = self.cfg;
        let script = Script::of_token(&word);
        let route = self
            .ctx
            .languages
            .iter()
            .position(|language| language.script() == script);

        // Orthographic normalisation of the original itself.
        let word = match script {
            Script::Arabic => light_stem::arabic_normalize(&word).unwrap_or(word),
            Script::Cyrillic if word.contains('ё') => word.replace('ё', "е"),
            _ => word,
        };

        if let Some(index) = route
            && self.ctx.stops[index].is_some_and(|set| set.contains(word.as_str()))
        {
            self.position += 1;
            return;
        }
        if cfg.max_token_length > 0 && word.chars().count() > cfg.max_token_length {
            self.position += 1;
            return;
        }

        let stem: Option<String> = route.and_then(|index| match cfg.stem {
            StemMode::None => None,
            StemMode::Light => light_stem::light_stem(self.ctx.languages[index], &word),
            StemMode::Snowball => {
                let stemmer = self.ctx.stemmers.get(index)?;
                match stemmer.stem(&word) {
                    std::borrow::Cow::Borrowed(_) => None,
                    std::borrow::Cow::Owned(stemmed) => (stemmed != word).then_some(stemmed),
                }
            }
        });

        let position = self.position;
        match self.emit {
            Emit::Index if cfg.keep_original => {
                let mut seen: Vec<String> = Vec::with_capacity(4);
                let mut push = |text: String, tokens: &mut Vec<Token>, variant: bool| {
                    if seen.contains(&text) {
                        return;
                    }
                    tokens.push(if variant {
                        Token::variant(text.clone(), position, from, to)
                    } else {
                        Token::new(text.clone(), position, from, to)
                    });
                    seen.push(text);
                };
                let folded = cfg.fold.then(|| fold_diacritics(&word)).flatten();
                let folded_stem = cfg
                    .fold
                    .then(|| stem.as_deref().and_then(fold_diacritics))
                    .flatten();
                push(word, &mut self.tokens, false);
                if let Some(stem) = stem {
                    push(stem, &mut self.tokens, true);
                }
                if let Some(folded) = folded {
                    push(folded, &mut self.tokens, true);
                }
                if let Some(folded) = folded_stem {
                    push(folded, &mut self.tokens, true);
                }
            }
            Emit::Exact if cfg.keep_original => {
                // The written form is indexed as the original token; its
                // stem and folded form are variants at the same position.
                self.tokens.push(Token::new(word, position, from, to));
            }
            Emit::Index | Emit::Match | Emit::Exact => {
                // Without `keep_original` the index holds one form per
                // word, the (folded) stem, so every query form is that.
                let base = stem.unwrap_or(word);
                let out = if cfg.fold && !cfg.keep_original {
                    fold_diacritics(&base).unwrap_or(base)
                } else {
                    base
                };
                self.tokens.push(Token::new(out, position, from, to));
            }
        }
        self.position += 1;
    }
}

/// Whether `c` belongs to a script that is bigrammed instead of stemmed.
#[inline]
pub(super) fn is_cjk_char(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x30FF      // Hiragana, Katakana
            | 0x31F0..=0x31FF // Katakana phonetic extensions
            | 0x3400..=0x4DBF // CJK extension A
            | 0x4E00..=0x9FFF // CJK unified ideographs
            | 0xF900..=0xFAFF // CJK compatibility ideographs
            | 0xFF66..=0xFF9F // half-width Katakana
            | 0x20000..=0x2FA1F // CJK extensions B..F, compatibility supplement
    )
}

/// Whether `c` is a Han ideograph (the scripts OpenCC converts).
#[inline]
fn is_han_char(c: char) -> bool {
    matches!(
        c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F
    )
}

/// Character-level traditional-to-simplified conversion (OpenCC
/// `TSCharacters` table, see [`super::han_t2s`]).
fn han_to_simplified(text: &str) -> String {
    text.chars()
        .map(|c| super::han_t2s::to_simplified(c).unwrap_or(c))
        .collect()
}

/// Diacritic-free form of a Latin, Cyrillic or Greek word (compatibility
/// decomposition, combining marks dropped, lowercased), or `None` when the
/// word has no diacritics or belongs to another script (whose combining
/// marks are letters in their own right).
pub(super) fn fold_diacritics(word: &str) -> Option<String> {
    if word.is_ascii() {
        return None;
    }
    if !matches!(
        Script::of_token(word),
        Script::Latin | Script::Cyrillic | Script::Greek
    ) {
        return None;
    }
    use unicode_normalization::UnicodeNormalization;
    use unicode_normalization::char::is_combining_mark;
    let folded: String = word
        .nfkd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(|c| c.to_lowercase())
        .collect();
    (folded != word).then_some(folded)
}

/// The process-wide ICU4X word segmenter (compiled data).
fn icu_word_segmenter() -> &'static icu_segmenter::WordSegmenterBorrowed<'static> {
    static SEGMENTER: std::sync::OnceLock<icu_segmenter::WordSegmenterBorrowed<'static>> =
        std::sync::OnceLock::new();
    SEGMENTER.get_or_init(|| {
        icu_segmenter::WordSegmenter::new_auto(
            icu_segmenter::options::WordBreakInvariantOptions::default(),
        )
    })
}

/// Stop words of a language (NLTK lists), shared for the process lifetime.
pub(super) fn stop_word_set(language: Language) -> Option<&'static HashSet<String>> {
    static SETS: std::sync::OnceLock<RwLock<HashMap<Language, &'static HashSet<String>>>> =
        std::sync::OnceLock::new();
    let sets = SETS.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(set) = sets.read().get(&language) {
        return Some(set);
    }
    let set: &'static HashSet<String> = Box::leak(Box::new(
        stop_words::get(language.to_stop_words_language())
            .iter()
            .map(|word| word.to_string())
            .collect(),
    ));
    Some(*sets.write().entry(language).or_insert(set))
}

/// Parsed form of a tokenizer name in the schema.
///
/// Plain names refer to a registered tokenizer (`simple`, `en_stem`, ...).
/// `stem(...)` declares a [`DynamicStemmer`] (see the module docs). The
/// canonical string form is stored in `FieldEntry::tokenizer`, so index
/// metadata needs no new field; options at their defaults are not rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSpec {
    /// A registered tokenizer name.
    Named(String),
    /// Dynamic stemmer hinted by another field of the document, or, without
    /// `by`, a fixed tokenizer with the same options that ignores hints.
    DynamicStem {
        /// Field whose text values supply the language hint; `None` = no
        /// per-document language, hints are ignored, `default` always applies.
        by: Option<String>,
        /// Language applied when the hint field is absent; `None` = simple.
        default: Option<Language>,
        /// Drop the routed language's stop words (positions keep their gaps).
        stop_words: bool,
        /// Word segmentation; `Simple` is the historic behaviour.
        segmenter: Segmenter,
        /// Morphological normalisation; `Snowball` is the historic behaviour.
        stem: StemMode,
        /// Keep the original next to its stem and folded form (variants).
        keep_original: bool,
        /// Fold diacritics (Latin, Cyrillic, Greek).
        fold: bool,
        /// Drop tokens longer than this many characters (0 = unlimited).
        max_token_length: usize,
        /// Convert Han tokens to simplified Chinese.
        t2s: bool,
        /// Japanese and Korean dictionary morphology (`cjk-dict` feature).
        morph: bool,
    },
}

impl TokenizerSpec {
    /// Parse a tokenizer name or `stem(...)` spec.
    pub fn parse(spec: &str) -> Result<TokenizerSpec, String> {
        let spec = spec.trim();
        let Some(rest) = spec.strip_prefix("stem(") else {
            if spec.is_empty() || spec.contains(['(', ')', ':', ',']) {
                return Err(format!("invalid tokenizer spec '{spec}'"));
            }
            return Ok(TokenizerSpec::Named(spec.to_string()));
        };
        let Some(params) = rest.strip_suffix(')') else {
            return Err(format!("tokenizer spec '{spec}' is missing ')'"));
        };
        let mut by = None;
        let mut default = None;
        let mut stop_words = false;
        let mut segmenter = Segmenter::Simple;
        let mut stem = StemMode::Snowball;
        let mut keep_original = false;
        let mut fold = true;
        let mut max_token_length = DEFAULT_MAX_TOKEN_LENGTH;
        let mut t2s = false;
        let mut morph = false;
        let parse_bool = |key: &str, value: &str| -> Result<bool, String> {
            match value {
                "true" => Ok(true),
                "false" => Ok(false),
                other => Err(format!(
                    "tokenizer spec '{spec}': '{key}' must be true or false, got '{other}'"
                )),
            }
        };
        for param in params.split(',') {
            let param = param.trim();
            if param.is_empty() {
                continue;
            }
            let Some((key, value)) = param.split_once(':') else {
                return Err(format!(
                    "tokenizer spec '{spec}': parameter '{param}' must be 'key: value'"
                ));
            };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "by" if !value.is_empty() => by = Some(value.to_string()),
                "by" => return Err(format!("tokenizer spec '{spec}': 'by' needs a field name")),
                "default" => {
                    default = match value {
                        "simple" | "none" => None,
                        other => Some(parse_language_opt(other).ok_or_else(|| {
                            format!("tokenizer spec '{spec}': unknown default language '{other}'")
                        })?),
                    };
                }
                "stop_words" => stop_words = parse_bool(key, value)?,
                "keep_original" => keep_original = parse_bool(key, value)?,
                "fold" => fold = parse_bool(key, value)?,
                "t2s" => t2s = parse_bool(key, value)?,
                "morph" => {
                    morph = parse_bool(key, value)?;
                    if morph && !cjk_morph::available() {
                        return Err(format!(
                            "tokenizer spec '{spec}': 'morph' needs a build with the cjk-dict feature (Japanese and Korean dictionaries)"
                        ));
                    }
                }
                "segmenter" => {
                    segmenter = match value {
                        "simple" => Segmenter::Simple,
                        "unicode" => Segmenter::Unicode,
                        "icu" => Segmenter::Icu,
                        other => {
                            return Err(format!(
                                "tokenizer spec '{spec}': 'segmenter' must be simple, unicode or icu, got '{other}'"
                            ));
                        }
                    };
                }
                "stem" => {
                    stem = match value {
                        "none" => StemMode::None,
                        "light" => StemMode::Light,
                        "snowball" => StemMode::Snowball,
                        other => {
                            return Err(format!(
                                "tokenizer spec '{spec}': 'stem' must be none, light or snowball, got '{other}'"
                            ));
                        }
                    };
                }
                "max_token_length" => {
                    max_token_length = value.parse::<usize>().map_err(|_| {
                        format!(
                            "tokenizer spec '{spec}': 'max_token_length' must be a number, got '{value}'"
                        )
                    })?;
                }
                other => {
                    return Err(format!(
                        "tokenizer spec '{spec}': unknown parameter '{other}'"
                    ));
                }
            }
        }
        Ok(TokenizerSpec::DynamicStem {
            by,
            default,
            stop_words,
            segmenter,
            stem,
            keep_original,
            fold,
            max_token_length,
            t2s,
            morph,
        })
    }

    /// Field whose values hint the tokenizer, for dynamic specs.
    pub fn hint_field(&self) -> Option<&str> {
        match self {
            TokenizerSpec::Named(_) => None,
            TokenizerSpec::DynamicStem { by, .. } => by.as_deref(),
        }
    }

    /// Whether the spec indexes originals next to their variants, so exact
    /// (phrase, term) queries match the written form and match queries the
    /// stem.
    pub fn keeps_original(&self) -> bool {
        match self {
            TokenizerSpec::Named(_) => false,
            TokenizerSpec::DynamicStem { keep_original, .. } => *keep_original,
        }
    }

    /// Build the tokenizer described by a dynamic spec.
    pub fn dynamic_tokenizer(&self) -> Option<super::BoxedTokenizer> {
        match self {
            TokenizerSpec::Named(_) => None,
            TokenizerSpec::DynamicStem {
                by,
                default,
                stop_words,
                segmenter,
                stem,
                keep_original,
                fold,
                max_token_length,
                t2s,
                morph,
            } => Some(Box::new(
                DynamicStemmer::new(*default)
                    .with_stop_words(*stop_words)
                    .with_segmenter(*segmenter)
                    .with_ignored_hints(by.is_none())
                    .with_stem(*stem)
                    .with_keep_original(*keep_original)
                    .with_fold(*fold)
                    .with_max_token_length(*max_token_length)
                    .with_t2s(*t2s)
                    .with_morph(*morph),
            )),
        }
    }
}

impl std::fmt::Display for TokenizerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerSpec::Named(name) => f.write_str(name),
            TokenizerSpec::DynamicStem {
                by,
                default,
                stop_words,
                segmenter,
                stem,
                keep_original,
                fold,
                max_token_length,
                t2s,
                morph,
            } => {
                let default = match default {
                    None => "simple".to_string(),
                    Some(language) => language_code(*language).to_string(),
                };
                match by {
                    Some(by) => write!(f, "stem(by: {by}, default: {default}")?,
                    None => write!(f, "stem(default: {default}")?,
                }
                if *stop_words {
                    write!(f, ", stop_words: true")?;
                }
                match segmenter {
                    Segmenter::Simple => {}
                    Segmenter::Unicode => write!(f, ", segmenter: unicode")?,
                    Segmenter::Icu => write!(f, ", segmenter: icu")?,
                }
                match stem {
                    StemMode::Snowball => {}
                    StemMode::Light => write!(f, ", stem: light")?,
                    StemMode::None => write!(f, ", stem: none")?,
                }
                if *keep_original {
                    write!(f, ", keep_original: true")?;
                }
                if !*fold {
                    write!(f, ", fold: false")?;
                }
                if *max_token_length != DEFAULT_MAX_TOKEN_LENGTH {
                    write!(f, ", max_token_length: {max_token_length}")?;
                }
                if *t2s {
                    write!(f, ", t2s: true")?;
                }
                if *morph {
                    write!(f, ", morph: true")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(tokens: &[Token]) -> Vec<(u32, String, bool)> {
        tokens
            .iter()
            .map(|t| (t.position, t.text.clone(), t.variant))
            .collect()
    }

    fn light_en() -> DynamicStemmer {
        DynamicStemmer::new(Some(Language::English))
            .with_segmenter(Segmenter::Icu)
            .with_stem(StemMode::Light)
            .with_keep_original(true)
            .with_stop_words(true)
    }

    #[test]
    fn keep_original_indexes_stem_and_folded_forms_as_variants() {
        let tokens = light_en().tokenize("The cell membranes of résumés");
        assert_eq!(
            texts(&tokens),
            vec![
                (1, "cell".to_string(), false),
                (2, "membranes".to_string(), false),
                (2, "membrane".to_string(), true),
                (4, "résumés".to_string(), false),
                (4, "résumé".to_string(), true),
                (4, "resumes".to_string(), true),
                (4, "resume".to_string(), true),
            ]
        );
        // A match query uses the stem, a phrase the original; never variants.
        let matched = light_en().tokenize_query("cell membranes", Some("en"), false);
        assert_eq!(
            texts(&matched),
            vec![
                (0, "cell".to_string(), false),
                (1, "membrane".to_string(), false)
            ]
        );
        let exact = light_en().tokenize_query("cell membranes", Some("en"), true);
        assert_eq!(
            texts(&exact),
            vec![
                (0, "cell".to_string(), false),
                (1, "membranes".to_string(), false)
            ]
        );
        // Unknown language: the match form is the original.
        let unknown =
            light_en()
                .with_ignored_hints(false)
                .tokenize_query("membranes", Some("xx"), false);
        // "xx" parses to nothing, so the default (English) applies.
        assert_eq!(unknown[0].text, "membrane");
        let no_default = DynamicStemmer::new(None)
            .with_stem(StemMode::Light)
            .with_keep_original(true)
            .tokenize_query("membranes", None, false);
        assert_eq!(no_default[0].text, "membranes");
    }

    #[test]
    fn icu_segments_cjk_words_with_bigram_variants_and_thai() {
        let tokenizer = DynamicStemmer::new(None).with_segmenter(Segmenter::Icu);
        let tokens = tokenizer.tokenize("量子コンピュータの研究");
        let words: Vec<(u32, &str, bool)> = tokens
            .iter()
            .map(|t| (t.position, t.text.as_str(), t.variant))
            .collect();
        assert_eq!(words[0], (0, "量子", false));
        // A dictionary word of three or more characters carries its bigrams.
        let computer: Vec<&(u32, &str, bool)> = words.iter().filter(|(p, _, _)| *p == 1).collect();
        assert_eq!(computer[0], &(1, "コンピュータ", false));
        assert!(computer.iter().skip(1).all(|(_, _, v)| *v));
        assert!(computer.iter().any(|(_, t, _)| *t == "コン"));
        assert!(words.contains(&(2, "の", false)));
        assert!(words.contains(&(3, "研究", false)));
        // Queries emit the words only.
        let query = tokenizer.tokenize_query("量子コンピュータ", None, false);
        assert!(query.iter().all(|t| !t.variant));
        assert_eq!(query.len(), 2);
        // Thai is split into words by the LSTM model.
        let thai = tokenizer.tokenize("สวัสดีครับ");
        assert!(thai.len() >= 2);
        assert!(thai.iter().all(|t| !t.variant));
    }

    #[test]
    fn long_tokens_are_dropped_but_keep_their_position() {
        let tokenizer = DynamicStemmer::new(None)
            .with_segmenter(Segmenter::Icu)
            .with_stem(StemMode::None)
            .with_max_token_length(8);
        let tokens = tokenizer.tokenize("short averyveryverylongtoken next");
        assert_eq!(
            texts(&tokens),
            vec![
                (0, "short".to_string(), false),
                (2, "next".to_string(), false)
            ]
        );
        let unlimited = tokenizer.clone().with_max_token_length(0);
        assert_eq!(unlimited.tokenize("averyveryverylongtoken").len(), 1);
    }

    #[test]
    fn stem_modes_and_arabic_normalisation() {
        let word = "running";
        let none = DynamicStemmer::new(Some(Language::English))
            .with_stem(StemMode::None)
            .tokenize(word);
        assert_eq!(none[0].text, "running");
        let light = DynamicStemmer::new(Some(Language::English))
            .with_stem(StemMode::Light)
            .tokenize(word);
        assert_eq!(light[0].text, "running");
        let snowball = DynamicStemmer::new(Some(Language::English))
            .with_stem(StemMode::Snowball)
            .tokenize(word);
        assert_eq!(snowball[0].text, "run");

        let arabic = DynamicStemmer::new(Some(Language::Arabic))
            .with_segmenter(Segmenter::Icu)
            .with_stem(StemMode::Light)
            .with_keep_original(true)
            .tokenize("الْكِتَابُ");
        assert_eq!(arabic[0].text, "الكتاب");
        assert!(!arabic[0].variant);
        assert_eq!(arabic[1].text, "كتاب");
        assert!(arabic[1].variant);
    }

    #[test]
    fn spec_round_trips_every_option() {
        let text = "stem(by: languages, default: en, stop_words: true, segmenter: icu, stem: light, keep_original: true, fold: false, max_token_length: 32, t2s: true)";
        let spec = TokenizerSpec::parse(text).unwrap();
        assert_eq!(spec.to_string(), text);
        assert!(spec.keeps_original());
        let TokenizerSpec::DynamicStem {
            stem,
            keep_original,
            fold,
            max_token_length,
            segmenter,
            t2s,
            morph,
            ..
        } = spec
        else {
            panic!()
        };
        assert_eq!(stem, StemMode::Light);
        assert!(keep_original && !fold);
        assert_eq!(max_token_length, 32);
        assert_eq!(segmenter, Segmenter::Icu);
        assert!(t2s && !morph);
        // Defaults are not rendered.
        assert_eq!(
            TokenizerSpec::parse("stem(by: languages, default: simple, stem: snowball, fold: true, max_token_length: 64)")
                .unwrap()
                .to_string(),
            "stem(by: languages, default: simple)"
        );
        assert!(TokenizerSpec::parse("stem(stem: aggressive)").is_err());
        assert!(TokenizerSpec::parse("stem(max_token_length: many)").is_err());
    }

    #[test]
    fn traditional_chinese_is_indexed_and_queried_as_simplified() {
        let tokenizer = DynamicStemmer::new(None)
            .with_segmenter(Segmenter::Icu)
            .with_t2s(true);
        let traditional: Vec<String> = tokenizer
            .tokenize("電腦網絡")
            .into_iter()
            .filter(|t| !t.variant)
            .map(|t| t.text)
            .collect();
        let simplified: Vec<String> = tokenizer
            .tokenize("电脑网络")
            .into_iter()
            .filter(|t| !t.variant)
            .map(|t| t.text)
            .collect();
        assert_eq!(traditional, simplified);
        assert!(traditional.concat().contains("电脑"));
        let query: Vec<String> = tokenizer
            .tokenize_query("電腦", None, false)
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert_eq!(query, vec!["电脑"]);
        // Japanese kana runs are untouched.
        let kana: Vec<String> = tokenizer
            .tokenize("コンピュータ")
            .into_iter()
            .filter(|t| !t.variant)
            .map(|t| t.text)
            .collect();
        assert_eq!(kana, vec!["コンピュータ"]);
    }

    #[cfg(feature = "cjk-dict")]
    #[test]
    fn morph_indexes_japanese_lemmas_and_korean_stems() {
        let tokenizer = DynamicStemmer::new(None)
            .with_segmenter(Segmenter::Icu)
            .with_morph(true);
        // Japanese needs the `ja` hint for Han runs; kana runs always go
        // through the dictionary.
        let ja = tokenizer.tokenize_hinted("研究を食べました", Some("ja"));
        assert_eq!(
            texts(&ja),
            vec![
                (0, "研究".to_string(), false),
                (2, "食べ".to_string(), false),
                (2, "食べる".to_string(), true),
            ]
        );
        let matched = tokenizer.tokenize_query("食べました", Some("ja"), false);
        assert_eq!(texts(&matched), vec![(0, "食べる".to_string(), false)]);
        let exact = tokenizer.tokenize_query("食べました", Some("ja"), true);
        assert_eq!(texts(&exact), vec![(0, "食べ".to_string(), false)]);

        // Korean needs no hint: particles and endings keep their positions.
        let ko = tokenizer.tokenize("학교에서 친구들과 공부했습니다");
        assert_eq!(
            texts(&ko),
            vec![
                (0, "학교".to_string(), false),
                (2, "친구".to_string(), false),
                (5, "공부".to_string(), false),
            ]
        );
        // Mixed text: Latin words still go through ICU and the stemmers.
        let mixed = DynamicStemmer::new(Some(Language::English))
            .with_segmenter(Segmenter::Icu)
            .with_stem(StemMode::Light)
            .with_keep_original(true)
            .with_morph(true)
            .tokenize("cells 학교에서");
        assert_eq!(
            texts(&mixed),
            vec![
                (0, "cells".to_string(), false),
                (0, "cell".to_string(), true),
                (1, "학교".to_string(), false),
            ]
        );
    }

    #[test]
    fn morph_spec_requires_the_feature() {
        let parsed = TokenizerSpec::parse("stem(default: simple, segmenter: icu, morph: true)");
        assert_eq!(parsed.is_ok(), cjk_morph::available());
        if let Ok(spec) = parsed {
            assert_eq!(
                spec.to_string(),
                "stem(default: simple, segmenter: icu, morph: true)"
            );
        }
    }
}
