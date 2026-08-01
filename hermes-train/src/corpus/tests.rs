use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use serde_json::{Value, json};

use super::*;

struct MockSearchTransport {
    responses: Mutex<VecDeque<Value>>,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl MockSearchTransport {
    fn new(responses: Vec<Value>) -> (Self, Arc<Mutex<Vec<Value>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                responses: Mutex::new(responses.into()),
                requests: Arc::clone(&requests),
            },
            requests,
        )
    }
}

impl SearchApiTransport for MockSearchTransport {
    fn post_json(&self, _endpoint: &str, body: &Value) -> Result<Value> {
        self.requests.lock().unwrap().push(body.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("unexpected search request"))
    }
}

fn search_api_config() -> SearchApiConfig {
    SearchApiConfig {
        endpoint: "https://search.invalid/v2/search".to_owned(),
        request_template: json!({
            "payload": {
                "words": "",
                "start": 0,
                "count": 1,
                "kind": "sparse_dense_fusion",
                "types": [],
                "reranker": { "enabled": true },
                "cross_rerank": true,
                "hydrate": true
            }
        }),
        request_mapping: SearchApiRequestMapping {
            query_pointer: "/payload/words".to_owned(),
            offset_pointer: "/payload/start".to_owned(),
            limit_pointer: "/payload/count".to_owned(),
            parameter_pointers: BTreeMap::from([("types".to_owned(), "/payload/types".to_owned())]),
            disabled_reranker_pointers: vec![
                "/payload/reranker/enabled".to_owned(),
                "/payload/cross_rerank".to_owned(),
            ],
            return_documents_pointer: Some("/payload/hydrate".to_owned()),
        },
        response_mapping: SearchApiResponseMapping {
            hits_pointer: "/results/items".to_owned(),
            total_hits_pointer: Some("/results/count".to_owned()),
            record_key_pointer: "/primary_key".to_owned(),
            score_pointer: Some("/rank".to_owned()),
            uris_pointer: Some("/opaque_locations".to_owned()),
            inline_text_pointer: None,
            metadata_pointers: BTreeMap::from([(
                "level".to_owned(),
                "/attributes/level".to_owned(),
            )]),
        },
        fusion: SearchApiFusionContract {
            marker_pointer: "/payload/kind".to_owned(),
            marker_value: json!("sparse_dense_fusion"),
            sparse_vector_field: "schema_specific_sparse".to_owned(),
            dense_vector_field: "schema_specific_dense".to_owned(),
        },
        snapshot: SourceSnapshot {
            provider: "search-test".to_owned(),
            revision: "index-42".to_owned(),
        },
        page_size: 50,
        timeout_seconds: 5,
        max_retries: 0,
        retry_initial_ms: 1,
        retry_max_ms: 1,
        auth: None,
    }
}

#[test]
fn search_api_uses_configured_fields_and_forces_rerankers_off() {
    let (transport, requests) = MockSearchTransport::new(vec![json!({
        "results": {
            "count": 1,
            "items": [{
                "primary_key": 17,
                "rank": 0.75,
                "opaque_locations": ["urn:book:17", "custom+store://bucket/key"],
                "attributes": { "level": "school" }
            }]
        }
    })]);
    let client = SearchApiClient::with_transport(search_api_config(), transport).unwrap();
    let query = DiscoveryQuery {
        name: "school".to_owned(),
        text: "basic geometry".to_owned(),
        limit: 10,
        parameters: BTreeMap::from([("types".to_owned(), json!(["manual"]))]),
    };
    let page = client.discover(&query, 7, 10).unwrap();
    assert_eq!(page.total_hits, Some(1));
    assert_eq!(page.hits[0].record_key, "17");
    assert_eq!(
        page.hits[0].uris,
        ["urn:book:17", "custom+store://bucket/key"]
    );
    assert_eq!(page.hits[0].metadata["level"], "school");

    let requests = requests.lock().unwrap();
    let request = &requests[0];
    assert_eq!(
        request.pointer("/payload/words"),
        Some(&json!("basic geometry"))
    );
    assert_eq!(request.pointer("/payload/start"), Some(&json!(7)));
    assert_eq!(request.pointer("/payload/count"), Some(&json!(10)));
    assert_eq!(request.pointer("/payload/types"), Some(&json!(["manual"])));
    assert_eq!(
        request.pointer("/payload/reranker/enabled"),
        Some(&json!(false))
    );
    assert_eq!(
        request.pointer("/payload/cross_rerank"),
        Some(&json!(false))
    );
    assert_eq!(request.pointer("/payload/hydrate"), Some(&json!(false)));
}

struct MockPostgresExecutor {
    rows: Vec<PostgresRecordRow>,
}

impl PostgresExecutor for MockPostgresExecutor {
    fn snapshot_revision(&self) -> Result<String> {
        Ok("10:20:".to_owned())
    }

    fn fetch(
        &self,
        statement: &str,
        columns: &PostgresColumnMapping,
        record_keys: &[String],
    ) -> Result<Vec<PostgresRecordRow>> {
        ensure!(statement == "SELECT configured");
        ensure!(columns.text == "unusual_text_alias");
        Ok(self
            .rows
            .iter()
            .filter(|row| record_keys.contains(&row.record_key))
            .cloned()
            .collect())
    }
}

fn postgres_config() -> PostgresRecordMaterializerConfig {
    PostgresRecordMaterializerConfig {
        connection_environment: "TEST_POSTGRES_DSN".to_owned(),
        statement: "SELECT configured".to_owned(),
        columns: PostgresColumnMapping {
            record_key: "unusual_key_alias".to_owned(),
            text: "unusual_text_alias".to_owned(),
            uris: Some("unusual_uri_alias".to_owned()),
            metadata: Some("unusual_metadata_alias".to_owned()),
        },
        snapshot_statement: "SELECT snapshot".to_owned(),
        require_every_record: true,
    }
}

#[test]
fn postgres_materializer_uses_canonical_rows_and_opaque_uris() {
    let executor = MockPostgresExecutor {
        rows: vec![PostgresRecordRow {
            record_key: "a".to_owned(),
            text: "canonical database text".to_owned(),
            uris: vec!["not-a-standard-prefix!value".to_owned()],
            metadata: BTreeMap::from([("kind".to_owned(), json!("book"))]),
        }],
    };
    let materializer =
        PostgresRecordMaterializer::with_executor(postgres_config(), executor).unwrap();
    let records = materializer
        .materialize(&[DiscoveryHit {
            record_key: "a".to_owned(),
            score: 1.0,
            uris: vec!["search:uri".to_owned()],
            metadata: BTreeMap::from([("query".to_owned(), json!("grammar"))]),
            inline_text: Some("search snippets are not canonical".to_owned()),
        }])
        .unwrap();
    assert_eq!(records[0].text, "canonical database text");
    assert_eq!(records[0].uris, ["not-a-standard-prefix!value"]);
    assert_eq!(records[0].metadata["kind"], "book");
    assert_eq!(records[0].metadata["query"], "grammar");
    assert_eq!(materializer.snapshot().unwrap().revision, "10:20:");
    let serialized = materializer.configuration().unwrap().to_string();
    assert!(serialized.contains("TEST_POSTGRES_DSN"));
    assert!(!serialized.contains("password"));
}

struct MockSearchBackend {
    calls: Mutex<Vec<(usize, usize)>>,
}

impl SearchBackend for MockSearchBackend {
    fn name(&self) -> &str {
        "mock_search"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"fields": {"key": "pk-z", "vectors": ["sv-z", "dv-z"]}}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "mock".to_owned(),
            revision: "stable".to_owned(),
        })
    }

    fn page_size(&self) -> usize {
        2
    }

    fn discover(
        &self,
        _query: &DiscoveryQuery,
        offset: usize,
        limit: usize,
    ) -> Result<DiscoveryPage> {
        self.calls.lock().unwrap().push((offset, limit));
        let hit = |record_key: &str| DiscoveryHit {
            record_key: record_key.to_owned(),
            score: 1.0,
            uris: vec![format!("opaque::{record_key}")],
            metadata: BTreeMap::new(),
            inline_text: None,
        };
        let hits = match offset {
            0 => vec![hit("a"), hit("b")],
            2 => vec![hit("b"), hit("c")],
            _ => vec![],
        };
        Ok(DiscoveryPage {
            hits,
            total_hits: Some(4),
        })
    }
}

struct MockMaterializer;

impl RecordMaterializer for MockMaterializer {
    fn name(&self) -> &str {
        "mock_store"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"text_column": "canonical-z"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "mock_store".to_owned(),
            revision: "stable".to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        // Return reverse order to verify that pipeline output is re-aligned to
        // deterministic discovery order.
        Ok(hits
            .iter()
            .rev()
            .map(|hit| {
                let (text, level) = match hit.record_key.as_str() {
                    "a" => ("Alpha   grammar\r\nlesson", "foundation"),
                    "b" => ("Alpha grammar\nlesson", "foundation"),
                    "c" => ("Advanced mathematics guide", "university"),
                    _ => unreachable!(),
                };
                MaterializedRecord {
                    record_key: hit.record_key.clone(),
                    text: text.to_owned(),
                    uris: hit.uris.clone(),
                    metadata: BTreeMap::from([("level".to_owned(), json!(level))]),
                }
            })
            .collect())
    }
}

struct MockTokenizer;

impl CorpusTokenizer for MockTokenizer {
    fn snapshot(&self) -> TokenizerSnapshot {
        TokenizerSnapshot {
            implementation: "mock".to_owned(),
            revision: "1".to_owned(),
            vocabulary_size: 100,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(index, _)| index as u32 + 1)
            .collect())
    }
}

fn pipeline_config() -> CorpusBuildConfig {
    CorpusBuildConfig {
        version: CORPUS_SCHEMA_VERSION,
        build_id: "test-build".to_owned(),
        discovery: DiscoveryConfig {
            queries: vec![DiscoveryQuery {
                name: "all".to_owned(),
                text: "all records".to_owned(),
                limit: 4,
                parameters: BTreeMap::new(),
            }],
            materialization_batch_size: 2,
        },
        normalization: NormalizationConfig::default(),
        deduplication: DeduplicationConfig::default(),
        classification: ClassificationConfig {
            topic_rules: vec![
                ClassificationRule {
                    label: "language".to_owned(),
                    any_terms: vec!["grammar".to_owned()],
                    metadata_equals: BTreeMap::new(),
                },
                ClassificationRule {
                    label: "math".to_owned(),
                    any_terms: vec!["mathematics".to_owned()],
                    metadata_equals: BTreeMap::new(),
                },
            ],
            difficulty_rules: vec![ClassificationRule {
                label: "advanced".to_owned(),
                any_terms: Vec::new(),
                metadata_equals: BTreeMap::from([("level".to_owned(), json!("university"))]),
            }],
            default_topic: Some("general".to_owned()),
            default_difficulty: Some("basic".to_owned()),
        },
        transformations: vec![TransformationConfig {
            name: "study".to_owned(),
            template: "Study ${topic}: ${text}".to_owned(),
            copies: 1,
            when: RecordPredicate::default(),
        }],
        repetition: RepetitionConfig {
            base_copies: 1,
            max_copies_per_record: 2,
            topic_copies: BTreeMap::new(),
            difficulty_copies: BTreeMap::new(),
        },
        token_target: TokenTarget {
            minimum: 6,
            desired: 6,
            maximum: 6,
        },
        sharding: ShardingConfig {
            max_tokens_per_shard: 7,
        },
    }
}

#[test]
fn pipeline_runs_all_stages_and_publishes_immutable_manifest() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let mut pipeline = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap();
    let (path, manifest) = pipeline.run(root.path()).unwrap();

    assert_eq!(*backend.calls.lock().unwrap(), [(0, 2), (2, 2)]);
    assert_eq!(manifest.build.stats.discovered_hits, 4);
    assert_eq!(manifest.build.stats.duplicate_discovery_keys, 1);
    assert_eq!(manifest.build.stats.materialized_records, 3);
    assert_eq!(manifest.build.stats.duplicate_records, 1);
    assert_eq!(manifest.build.stats.unique_records, 2);
    assert_eq!(manifest.build.stats.unique_tokens, 6);
    assert_eq!(manifest.build.stats.emitted_views, 4);
    assert_eq!(manifest.build.topic_counts["language"], 1);
    assert_eq!(manifest.build.topic_counts["math"], 1);
    assert_eq!(manifest.build.difficulty_counts["advanced"], 1);
    assert!(manifest.build.desired_token_target_reached);
    assert!(path.join("manifest.json").is_file());
    assert!(!root.path().join(".test-build.building").exists());

    let lines: Vec<Value> = manifest
        .build
        .shards
        .iter()
        .flat_map(|shard| {
            fs::read_to_string(path.join(&shard.path))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["record_key"], "a");
    assert_eq!(lines[0]["uris"], json!(["opaque::a"]));
    assert_eq!(lines[2]["record_key"], "c");

    let mut second_dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut second_dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err()
    .to_string();
    assert!(error.contains("already exists"));

    let second_root = tempfile::tempdir().unwrap();
    let second_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
    };
    let mut reproducible_dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (_, reproduced) = CorpusPipeline::new(
        pipeline_config(),
        &second_backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut reproducible_dedup,
    )
    .unwrap()
    .run(second_root.path())
    .unwrap();
    assert_eq!(reproduced.manifest_sha256, manifest.manifest_sha256);
    assert_eq!(reproduced.build.shards, manifest.build.shards);
}

#[test]
fn sqlite_deduplicator_detects_key_and_text_duplicates() {
    let directory = tempfile::tempdir().unwrap();
    let mut dedup = SqliteDeduplicator::open(
        &directory.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    assert!(!dedup.seen_discovery_key("a").unwrap());
    assert!(dedup.seen_discovery_key("a").unwrap());
    assert!(!dedup.seen_or_insert("a", "first").unwrap());
    assert!(dedup.seen_or_insert("a", "different").unwrap());
    assert!(dedup.seen_or_insert("b", "first").unwrap());
    assert!(!dedup.seen_or_insert("b", "second").unwrap());
}

#[test]
fn production_recipe_is_strict_and_validates_without_live_services() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus.production.example.json");
    let recipe = SearchApiPostgresCorpusRecipe::load(&path).unwrap();
    assert_eq!(recipe.corpus.token_target.minimum, 10_000_000_000);
    assert_eq!(recipe.corpus.token_target.maximum, 20_000_000_000);
    assert_eq!(recipe.search_api.fusion.marker_value, "hybrid");
    assert_eq!(
        recipe.postgres.connection_environment,
        "CORPUS_POSTGRES_DSN"
    );
}
