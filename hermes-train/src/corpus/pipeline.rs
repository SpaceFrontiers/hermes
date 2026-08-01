use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    ClassificationRule, CorpusBuildConfig, DeduplicationConfig, DiscoveryHit, MaterializedRecord,
    NormalizationConfig, RecordMaterializer, RecordPredicate, RepetitionConfig, SearchBackend,
    SourceSnapshot, TransformationConfig,
};

pub trait CorpusTokenizer: Send + Sync {
    fn snapshot(&self) -> TokenizerSnapshot;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSnapshot {
    pub implementation: String,
    pub revision: String,
    pub vocabulary_size: usize,
}

pub struct HermesCorpusTokenizer<'a> {
    tokenizer: &'a Tokenizer,
    revision: String,
}

impl<'a> HermesCorpusTokenizer<'a> {
    pub fn new(tokenizer: &'a Tokenizer, revision: impl Into<String>) -> Result<Self> {
        let revision = revision.into();
        ensure!(
            !revision.trim().is_empty(),
            "tokenizer revision must not be empty"
        );
        Ok(Self {
            tokenizer,
            revision,
        })
    }
}

impl CorpusTokenizer for HermesCorpusTokenizer<'_> {
    fn snapshot(&self) -> TokenizerSnapshot {
        TokenizerSnapshot {
            implementation: "hermes-tokenizer".to_owned(),
            revision: self.revision.clone(),
            vocabulary_size: self.tokenizer.vocab_size(),
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }
}

pub trait Deduplicator {
    /// Returns `true` when a discovery key has already been scheduled for
    /// materialization. This makes discovery streaming bounded-memory even
    /// when many queries return the same records.
    fn seen_discovery_key(&mut self, record_key: &str) -> Result<bool>;
    /// Returns `true` if the record key or normalized text was already seen.
    fn seen_or_insert(&mut self, record_key: &str, normalized_text: &str) -> Result<bool>;
    fn configuration(&self) -> Value;
}

pub struct InMemoryDeduplicator {
    config: DeduplicationConfig,
    discovery_keys: HashSet<String>,
    keys: HashSet<String>,
    texts: HashSet<[u8; 32]>,
}

impl InMemoryDeduplicator {
    pub fn new(config: DeduplicationConfig) -> Result<Self> {
        config.validate_for_pipeline()?;
        Ok(Self {
            config,
            discovery_keys: HashSet::new(),
            keys: HashSet::new(),
            texts: HashSet::new(),
        })
    }
}

impl Deduplicator for InMemoryDeduplicator {
    fn seen_discovery_key(&mut self, record_key: &str) -> Result<bool> {
        Ok(!self.discovery_keys.insert(record_key.to_owned()))
    }

    fn seen_or_insert(&mut self, record_key: &str, normalized_text: &str) -> Result<bool> {
        let text_hash = sha256(normalized_text.as_bytes());
        let duplicate = (self.config.by_record_key && self.keys.contains(record_key))
            || (self.config.by_normalized_text && self.texts.contains(&text_hash));
        if duplicate {
            return Ok(true);
        }
        if self.config.by_record_key {
            self.keys.insert(record_key.to_owned());
        }
        if self.config.by_normalized_text {
            self.texts.insert(text_hash);
        }
        Ok(false)
    }

    fn configuration(&self) -> Value {
        serde_json::json!({
            "type": "in_memory",
            "by_record_key": self.config.by_record_key,
            "by_normalized_text": self.config.by_normalized_text,
        })
    }
}

/// Disk-backed exact deduplication for production-scale builds. Its database
/// belongs to an in-progress build and is not part of the immutable corpus.
pub struct SqliteDeduplicator {
    config: DeduplicationConfig,
    connection: Connection,
}

impl SqliteDeduplicator {
    pub fn open(path: &Path, config: DeduplicationConfig) -> Result<Self> {
        config.validate_for_pipeline()?;
        let connection = Connection::open(path)
            .with_context(|| format!("failed to open deduplication catalog {}", path.display()))?;
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS corpus_keys (
                 record_key TEXT PRIMARY KEY NOT NULL
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS discovery_keys (
                 record_key TEXT PRIMARY KEY NOT NULL
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS corpus_texts (
                 text_sha256 BLOB PRIMARY KEY NOT NULL
             ) WITHOUT ROWID;",
        )?;
        Ok(Self { config, connection })
    }
}

impl Deduplicator for SqliteDeduplicator {
    fn seen_discovery_key(&mut self, record_key: &str) -> Result<bool> {
        Ok(self.connection.execute(
            "INSERT OR IGNORE INTO discovery_keys(record_key) VALUES (?1)",
            [record_key],
        )? == 0)
    }

    fn seen_or_insert(&mut self, record_key: &str, normalized_text: &str) -> Result<bool> {
        let text_hash = sha256(normalized_text.as_bytes());
        let transaction = self.connection.transaction()?;
        let duplicate_key = if self.config.by_record_key {
            transaction
                .query_row(
                    "SELECT 1 FROM corpus_keys WHERE record_key = ?1",
                    [record_key],
                    |_| Ok(()),
                )
                .optional()?
                .is_some()
        } else {
            false
        };
        let duplicate_text = if self.config.by_normalized_text {
            transaction
                .query_row(
                    "SELECT 1 FROM corpus_texts WHERE text_sha256 = ?1",
                    [text_hash.as_slice()],
                    |_| Ok(()),
                )
                .optional()?
                .is_some()
        } else {
            false
        };
        if duplicate_key || duplicate_text {
            return Ok(true);
        }
        if self.config.by_record_key {
            transaction.execute(
                "INSERT INTO corpus_keys(record_key) VALUES (?1)",
                [record_key],
            )?;
        }
        if self.config.by_normalized_text {
            transaction.execute(
                "INSERT INTO corpus_texts(text_sha256) VALUES (?1)",
                params![text_hash.as_slice()],
            )?;
        }
        transaction.commit()?;
        Ok(false)
    }

    fn configuration(&self) -> Value {
        serde_json::json!({
            "type": "sqlite_exact",
            "by_record_key": self.config.by_record_key,
            "by_normalized_text": self.config.by_normalized_text,
        })
    }
}

// Keep validation private in config.rs while allowing construction of
// pipeline-owned implementations.
trait DeduplicationValidation {
    fn validate_for_pipeline(&self) -> Result<()>;
}

impl DeduplicationValidation for DeduplicationConfig {
    fn validate_for_pipeline(&self) -> Result<()> {
        ensure!(
            self.by_record_key || self.by_normalized_text,
            "deduplication must enable record-key or normalized-text matching"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComponentManifest {
    pub name: String,
    pub configuration: Value,
    pub snapshot: SourceSnapshot,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorpusStageStats {
    pub discovery_pages: u64,
    pub discovered_hits: u64,
    pub duplicate_discovery_keys: u64,
    pub materialized_records: u64,
    pub normalization_rejections: u64,
    pub duplicate_records: u64,
    pub unique_records: u64,
    pub unique_tokens: u64,
    pub emitted_views: u64,
    pub exposure_tokens: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ShardManifest {
    pub path: String,
    pub records: u64,
    pub tokens: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifestBody {
    pub version: u32,
    pub build_id: String,
    pub config_sha256: String,
    pub config: CorpusBuildConfig,
    pub discovery: ComponentManifest,
    pub materializer: ComponentManifest,
    pub deduplicator: Value,
    pub tokenizer: TokenizerSnapshot,
    pub stats: CorpusStageStats,
    pub desired_token_target_reached: bool,
    pub topic_counts: BTreeMap<String, u64>,
    pub difficulty_counts: BTreeMap<String, u64>,
    pub shards: Vec<ShardManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    pub manifest_sha256: String,
    pub build: CorpusManifestBody,
}

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenizedCorpusRecord<'a> {
    record_key: &'a str,
    uris: &'a [String],
    topic: Option<&'a str>,
    difficulty: Option<&'a str>,
    view: &'a str,
    copy: usize,
    metadata: &'a BTreeMap<String, Value>,
    tokens: &'a [u32],
}

pub struct CorpusPipeline<'a> {
    config: CorpusBuildConfig,
    search: &'a dyn SearchBackend,
    materializer: &'a dyn RecordMaterializer,
    tokenizer: &'a dyn CorpusTokenizer,
    deduplicator: &'a mut dyn Deduplicator,
}

impl<'a> CorpusPipeline<'a> {
    pub fn new(
        config: CorpusBuildConfig,
        search: &'a dyn SearchBackend,
        materializer: &'a dyn RecordMaterializer,
        tokenizer: &'a dyn CorpusTokenizer,
        deduplicator: &'a mut dyn Deduplicator,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            search.page_size() > 0,
            "search backend page size must be positive"
        );
        Ok(Self {
            config,
            search,
            materializer,
            tokenizer,
            deduplicator,
        })
    }

    /// Build `output_root/<build_id>` exactly once. A manifest is written last
    /// in a staging directory, then the complete directory is renamed into
    /// place. Existing completed builds are never overwritten.
    pub fn run(&mut self, output_root: &Path) -> Result<(PathBuf, CorpusManifest)> {
        let discovery_snapshot = self.search.snapshot()?;
        let materializer_snapshot = self.materializer.snapshot()?;
        let discovery_configuration = self.search.configuration()?;
        let materializer_configuration = self.materializer.configuration()?;
        let config_value = serde_json::to_value(&self.config)?;
        let config_sha256 = canonical_json_sha256(&serde_json::json!({
            "corpus": config_value,
            "search": discovery_configuration,
            "materializer": materializer_configuration,
            "tokenizer": self.tokenizer.snapshot(),
        }))?;

        fs::create_dir_all(output_root).with_context(|| {
            format!(
                "failed to create corpus output root {}",
                output_root.display()
            )
        })?;
        let final_path = output_root.join(&self.config.build_id);
        ensure!(
            !final_path.exists(),
            "immutable corpus build {} already exists",
            final_path.display()
        );
        let staging_path = output_root.join(format!(".{}.building", self.config.build_id));
        ensure!(
            !staging_path.exists(),
            "in-progress corpus build {} already exists",
            staging_path.display()
        );
        fs::create_dir(&staging_path).with_context(|| {
            format!(
                "failed to create corpus staging path {}",
                staging_path.display()
            )
        })?;

        let mut stats = CorpusStageStats::default();
        let mut writer =
            ImmutableShardWriter::new(&staging_path, self.config.sharding.max_tokens_per_shard);
        let mut topic_counts = BTreeMap::new();
        let mut difficulty_counts = BTreeMap::new();

        // Pages flow directly into bounded materialization batches. The exact
        // deduplicator owns discovery-key state, so corpus size does not turn
        // into an in-memory hit list.
        let queries = self.config.discovery.queries.clone();
        let batch_size = self.config.discovery.materialization_batch_size;
        let mut pending = Vec::with_capacity(batch_size);
        for query in &queries {
            let mut offset = 0usize;
            while offset < query.limit {
                let request_limit = self.search.page_size().min(query.limit - offset);
                let page = self
                    .search
                    .discover(query, offset, request_limit)
                    .with_context(|| format!("discovery query `{}` failed", query.name))?;
                stats.discovery_pages = checked_add(stats.discovery_pages, 1)?;
                stats.discovered_hits = checked_add(stats.discovered_hits, page.hits.len())?;
                let returned = page.hits.len();
                for mut hit in page.hits {
                    if self.deduplicator.seen_discovery_key(&hit.record_key)? {
                        stats.duplicate_discovery_keys =
                            checked_add(stats.duplicate_discovery_keys, 1)?;
                        continue;
                    }
                    hit.metadata.insert(
                        "discovery_query".to_owned(),
                        Value::String(query.name.clone()),
                    );
                    pending.push(hit);
                    if pending.len() == batch_size {
                        let batch = std::mem::replace(&mut pending, Vec::with_capacity(batch_size));
                        self.process_batch(
                            &batch,
                            &mut writer,
                            &mut stats,
                            &mut topic_counts,
                            &mut difficulty_counts,
                        )?;
                    }
                }
                if returned == 0 {
                    break;
                }
                offset = offset
                    .checked_add(returned)
                    .context("search pagination offset overflows usize")?;
                if returned < request_limit
                    || page.total_hits.is_some_and(|total| offset as u64 >= total)
                {
                    break;
                }
            }
        }
        if !pending.is_empty() {
            self.process_batch(
                &pending,
                &mut writer,
                &mut stats,
                &mut topic_counts,
                &mut difficulty_counts,
            )?;
        }
        let shards = writer.finish()?;
        ensure!(
            self.config.token_target.accepts(stats.unique_tokens),
            "corpus unique-token count {} is outside configured [{}, {}] target",
            stats.unique_tokens,
            self.config.token_target.minimum,
            self.config.token_target.maximum
        );
        ensure!(
            self.search.snapshot()? == discovery_snapshot,
            "search backend snapshot changed during corpus build"
        );
        ensure!(
            self.materializer.snapshot()? == materializer_snapshot,
            "record materializer snapshot changed during corpus build"
        );

        let body = CorpusManifestBody {
            version: self.config.version,
            build_id: self.config.build_id.clone(),
            config_sha256,
            config: self.config.clone(),
            discovery: ComponentManifest {
                name: self.search.name().to_owned(),
                configuration: discovery_configuration,
                snapshot: discovery_snapshot,
            },
            materializer: ComponentManifest {
                name: self.materializer.name().to_owned(),
                configuration: materializer_configuration,
                snapshot: materializer_snapshot,
            },
            deduplicator: self.deduplicator.configuration(),
            tokenizer: self.tokenizer.snapshot(),
            stats: stats.clone(),
            desired_token_target_reached: stats.unique_tokens >= self.config.token_target.desired,
            topic_counts,
            difficulty_counts,
            shards,
        };
        let manifest = CorpusManifest {
            manifest_sha256: canonical_json_sha256(&serde_json::to_value(&body)?)?,
            build: body,
        };
        let manifest_path = staging_path.join("manifest.json");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&manifest_path)?;
        serde_json::to_writer_pretty(&mut file, &manifest)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&staging_path, &final_path).with_context(|| {
            format!(
                "failed to publish immutable corpus {}",
                final_path.display()
            )
        })?;
        Ok((final_path, manifest))
    }

    fn process_batch(
        &mut self,
        hit_batch: &[DiscoveryHit],
        writer: &mut ImmutableShardWriter,
        stats: &mut CorpusStageStats,
        topic_counts: &mut BTreeMap<String, u64>,
        difficulty_counts: &mut BTreeMap<String, u64>,
    ) -> Result<()> {
        let materialized = self.materializer.materialize(hit_batch).with_context(|| {
            format!(
                "{} failed to materialize {} discovery hits",
                self.materializer.name(),
                hit_batch.len()
            )
        })?;
        stats.materialized_records = checked_add(stats.materialized_records, materialized.len())?;
        let ordered = order_materialized(hit_batch, materialized)?;
        for mut record in ordered {
            let Some(normalized) = normalize(&record.text, &self.config.normalization) else {
                stats.normalization_rejections = checked_add(stats.normalization_rejections, 1)?;
                continue;
            };
            if self
                .deduplicator
                .seen_or_insert(&record.record_key, &normalized)?
            {
                stats.duplicate_records = checked_add(stats.duplicate_records, 1)?;
                continue;
            }
            record.text = normalized;
            let topic = classify(
                &record,
                &self.config.classification.topic_rules,
                self.config.classification.default_topic.as_deref(),
            );
            let difficulty = classify(
                &record,
                &self.config.classification.difficulty_rules,
                self.config.classification.default_difficulty.as_deref(),
            );
            if let Some(topic) = &topic {
                increment_label(topic_counts, topic)?;
            }
            if let Some(difficulty) = &difficulty {
                increment_label(difficulty_counts, difficulty)?;
            }

            let canonical_tokens = self
                .tokenizer
                .encode(&record.text)
                .with_context(|| format!("failed to tokenize `{}`", record.record_key))?;
            ensure!(
                !canonical_tokens.is_empty(),
                "tokenizer emitted no tokens for `{}`",
                record.record_key
            );
            stats.unique_records = checked_add(stats.unique_records, 1)?;
            stats.unique_tokens = checked_add(stats.unique_tokens, canonical_tokens.len())?;
            ensure!(
                stats.unique_tokens <= self.config.token_target.maximum,
                "corpus exceeded maximum token target {} with {} unique tokens",
                self.config.token_target.maximum,
                stats.unique_tokens
            );

            let views = render_views(
                &record,
                topic.as_deref(),
                difficulty.as_deref(),
                &self.config.transformations,
                &self.config.repetition,
            );
            for view in views {
                let tokens = if view.name == "source" && view.copy == 0 {
                    canonical_tokens.clone()
                } else {
                    self.tokenizer.encode(&view.text).with_context(|| {
                        format!(
                            "failed to tokenize view `{}` for `{}`",
                            view.name, record.record_key
                        )
                    })?
                };
                ensure!(
                    !tokens.is_empty(),
                    "tokenizer emitted no tokens for view `{}` of `{}`",
                    view.name,
                    record.record_key
                );
                writer.push(TokenizedCorpusRecord {
                    record_key: &record.record_key,
                    uris: &record.uris,
                    topic: topic.as_deref(),
                    difficulty: difficulty.as_deref(),
                    view: &view.name,
                    copy: view.copy,
                    metadata: &record.metadata,
                    tokens: &tokens,
                })?;
                stats.emitted_views = checked_add(stats.emitted_views, 1)?;
                stats.exposure_tokens = checked_add(stats.exposure_tokens, tokens.len())?;
            }
        }
        Ok(())
    }
}

fn order_materialized(
    hits: &[DiscoveryHit],
    records: Vec<MaterializedRecord>,
) -> Result<Vec<MaterializedRecord>> {
    let mut by_key: HashMap<_, _> = records
        .into_iter()
        .map(|record| (record.record_key.clone(), record))
        .collect();
    let mut ordered = Vec::with_capacity(by_key.len());
    for hit in hits {
        if let Some(record) = by_key.remove(&hit.record_key) {
            ordered.push(record);
        }
    }
    ensure!(
        by_key.is_empty(),
        "materializer returned records absent from its discovery batch"
    );
    Ok(ordered)
}

fn normalize(text: &str, config: &NormalizationConfig) -> Option<String> {
    if config.reject_replacement_character && text.contains('\u{fffd}') {
        return None;
    }
    let mut text = if config.normalize_newlines {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_owned()
    };
    if config.collapse_horizontal_whitespace {
        let mut normalized = String::with_capacity(text.len());
        let mut horizontal = false;
        for character in text.chars() {
            if character != '\n' && character.is_whitespace() {
                horizontal = true;
            } else {
                if horizontal && !normalized.ends_with([' ', '\n']) {
                    normalized.push(' ');
                }
                horizontal = false;
                normalized.push(character);
            }
        }
        text = normalized;
    }
    if config.normalize_newlines {
        let maximum_newlines = config.max_consecutive_blank_lines.saturating_add(1);
        let mut normalized = String::with_capacity(text.len());
        let mut newlines = 0usize;
        for character in text.chars() {
            if character == '\n' {
                newlines += 1;
                if newlines <= maximum_newlines {
                    normalized.push(character);
                }
            } else {
                newlines = 0;
                normalized.push(character);
            }
        }
        text = normalized;
    }
    if config.trim {
        text = text.trim().to_owned();
    }
    (text.chars().count() >= config.minimum_characters).then_some(text)
}

fn classify(
    record: &MaterializedRecord,
    rules: &[ClassificationRule],
    default: Option<&str>,
) -> Option<String> {
    let lowercase = record.text.to_lowercase();
    rules
        .iter()
        .find(|rule| {
            (rule.any_terms.is_empty()
                || rule
                    .any_terms
                    .iter()
                    .any(|term| lowercase.contains(&term.to_lowercase())))
                && rule
                    .metadata_equals
                    .iter()
                    .all(|(name, expected)| record.metadata.get(name) == Some(expected))
        })
        .map(|rule| rule.label.clone())
        .or_else(|| default.map(str::to_owned))
}

struct RenderedView {
    name: String,
    copy: usize,
    text: String,
}

fn render_views(
    record: &MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
    transforms: &[TransformationConfig],
    repetition: &RepetitionConfig,
) -> Vec<RenderedView> {
    let base_copies = repetition
        .topic_copies
        .get(topic.unwrap_or_default())
        .into_iter()
        .chain(
            repetition
                .difficulty_copies
                .get(difficulty.unwrap_or_default()),
        )
        .copied()
        .fold(repetition.base_copies, usize::max);
    let mut views = (0..base_copies)
        .map(|copy| RenderedView {
            name: "source".to_owned(),
            copy,
            text: record.text.clone(),
        })
        .collect::<Vec<_>>();
    for transform in transforms {
        if !predicate_matches(&transform.when, record, topic, difficulty) {
            continue;
        }
        for copy in 0..transform.copies {
            views.push(RenderedView {
                name: transform.name.clone(),
                copy,
                text: render_template(&transform.template, record, topic, difficulty),
            });
        }
    }
    views.truncate(repetition.max_copies_per_record);
    views
}

fn predicate_matches(
    predicate: &RecordPredicate,
    record: &MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
) -> bool {
    (predicate.topics.is_empty()
        || topic.is_some_and(|value| predicate.topics.iter().any(|v| v == value)))
        && (predicate.difficulties.is_empty()
            || difficulty.is_some_and(|value| predicate.difficulties.iter().any(|v| v == value)))
        && predicate
            .metadata_equals
            .iter()
            .all(|(name, expected)| record.metadata.get(name) == Some(expected))
}

fn render_template(
    template: &str,
    record: &MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
) -> String {
    let mut rendered = template
        .replace("${text}", &record.text)
        .replace("${topic}", topic.unwrap_or_default())
        .replace("${difficulty}", difficulty.unwrap_or_default());
    for (name, value) in &record.metadata {
        let replacement = value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string());
        rendered = rendered.replace(&format!("${{metadata.{name}}}"), &replacement);
    }
    rendered
}

struct OpenShard {
    path: String,
    writer: BufWriter<File>,
    hasher: Sha256,
    records: u64,
    tokens: u64,
}

struct ImmutableShardWriter {
    directory: PathBuf,
    max_tokens: u64,
    next_index: usize,
    open: Option<OpenShard>,
    completed: Vec<ShardManifest>,
}

impl ImmutableShardWriter {
    fn new(directory: &Path, max_tokens: u64) -> Self {
        Self {
            directory: directory.to_owned(),
            max_tokens,
            next_index: 0,
            open: None,
            completed: Vec::new(),
        }
    }

    fn push(&mut self, record: TokenizedCorpusRecord<'_>) -> Result<()> {
        let token_count = u64::try_from(record.tokens.len())?;
        if self.open.as_ref().is_some_and(|shard| {
            shard.records > 0 && shard.tokens.saturating_add(token_count) > self.max_tokens
        }) {
            self.close()?;
        }
        if self.open.is_none() {
            self.open()?;
        }
        let bytes = serde_json::to_vec(&record)?;
        let shard = self.open.as_mut().expect("shard was opened");
        shard.writer.write_all(&bytes)?;
        shard.writer.write_all(b"\n")?;
        shard.hasher.update(&bytes);
        shard.hasher.update(b"\n");
        shard.records = checked_add(shard.records, 1)?;
        shard.tokens = checked_add(shard.tokens, record.tokens.len())?;
        Ok(())
    }

    fn open(&mut self) -> Result<()> {
        let path = format!("shard-{:05}.tokens.jsonl", self.next_index);
        self.next_index += 1;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(&path))?;
        self.open = Some(OpenShard {
            path,
            writer: BufWriter::new(file),
            hasher: Sha256::new(),
            records: 0,
            tokens: 0,
        });
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        let Some(mut shard) = self.open.take() else {
            return Ok(());
        };
        shard.writer.flush()?;
        shard.writer.get_ref().sync_all()?;
        self.completed.push(ShardManifest {
            path: shard.path,
            records: shard.records,
            tokens: shard.tokens,
            sha256: hex(&shard.hasher.finalize()),
        });
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<ShardManifest>> {
        self.close()?;
        ensure!(!self.completed.is_empty(), "corpus emitted no shards");
        Ok(self.completed)
    }
}

fn canonical_json_sha256(value: &Value) -> Result<String> {
    let canonical = canonicalize(value);
    Ok(hex(&Sha256::digest(serde_json::to_vec(&canonical)?)))
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect();
            serde_json::to_value(sorted).expect("BTreeMap JSON serialization cannot fail")
        }
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn checked_add(current: u64, amount: impl TryInto<u64>) -> Result<u64> {
    let amount = amount
        .try_into()
        .map_err(|_| anyhow::anyhow!("corpus counter value does not fit u64"))?;
    current
        .checked_add(amount)
        .context("corpus counter overflows u64")
}

fn increment_label(counts: &mut BTreeMap<String, u64>, label: &str) -> Result<()> {
    let count = counts.entry(label.to_owned()).or_default();
    *count = count
        .checked_add(1)
        .context("classification count overflows u64")?;
    Ok(())
}
