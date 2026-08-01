use std::collections::{BTreeMap, HashMap};
use std::env;
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use postgres::{Client, NoTls, Row};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{DiscoveryHit, SourceSnapshot};

#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedRecord {
    pub record_key: String,
    pub text: String,
    /// URI values remain opaque; materializers never impose a scheme.
    pub uris: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
}

pub trait RecordMaterializer: Send + Sync {
    fn name(&self) -> &str;
    /// Serializable, secret-free materializer configuration for the build
    /// manifest.
    fn configuration(&self) -> Result<Value>;
    fn snapshot(&self) -> Result<SourceSnapshot>;
    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>>;
}

/// Materializer for discovery engines which are themselves the canonical text
/// store. It fails when any hit lacks inline content.
#[derive(Default)]
pub struct InlineRecordMaterializer;

impl RecordMaterializer for InlineRecordMaterializer {
    fn name(&self) -> &str {
        "inline"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(serde_json::json!({ "type": "inline" }))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "inline".to_owned(),
            revision: "discovery-snapshot".to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        hits.iter()
            .map(|hit| {
                let text = hit.inline_text.clone().with_context(|| {
                    format!("discovery hit `{}` has no inline text", hit.record_key)
                })?;
                Ok(MaterializedRecord {
                    record_key: hit.record_key.clone(),
                    text,
                    uris: hit.uris.clone(),
                    metadata: hit.metadata.clone(),
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresRecordMaterializerConfig {
    /// Environment variable containing the PostgreSQL DSN. Only this variable
    /// name is serialized; its value never enters the corpus manifest.
    pub connection_environment: String,
    /// Prepared-style statement accepting one `text[]` parameter containing
    /// discovery keys. Column aliases are mapped below and are not inferred.
    pub statement: String,
    pub columns: PostgresColumnMapping,
    #[serde(default = "default_snapshot_statement")]
    pub snapshot_statement: String,
    #[serde(default = "default_true")]
    pub require_every_record: bool,
}

fn default_snapshot_statement() -> String {
    "SELECT txid_current_snapshot()::text".to_owned()
}

fn default_true() -> bool {
    true
}

impl PostgresRecordMaterializerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.connection_environment.trim().is_empty(),
            "postgres connection_environment must not be empty"
        );
        ensure!(
            !self.statement.trim().is_empty(),
            "postgres materialization statement must not be empty"
        );
        ensure!(
            !self.snapshot_statement.trim().is_empty(),
            "postgres snapshot_statement must not be empty"
        );
        self.columns.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresColumnMapping {
    pub record_key: String,
    pub text: String,
    pub uris: Option<String>,
    pub metadata: Option<String>,
}

impl PostgresColumnMapping {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.record_key.trim().is_empty() && !self.text.trim().is_empty(),
            "postgres record_key and text column mappings must not be empty"
        );
        ensure!(
            self.uris
                .as_ref()
                .is_none_or(|name| !name.trim().is_empty())
                && self
                    .metadata
                    .as_ref()
                    .is_none_or(|name| !name.trim().is_empty()),
            "optional postgres column mappings must not be empty"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PostgresRecordRow {
    pub record_key: String,
    pub text: String,
    pub uris: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
}

pub trait PostgresExecutor: Send + Sync {
    fn snapshot_revision(&self) -> Result<String>;
    fn fetch(
        &self,
        statement: &str,
        columns: &PostgresColumnMapping,
        record_keys: &[String],
    ) -> Result<Vec<PostgresRecordRow>>;
}

/// One read-only, repeatable-read transaction keeps every materialization
/// query on the same PostgreSQL snapshot for the lifetime of a build.
pub struct LivePostgresExecutor {
    client: Mutex<Client>,
    snapshot_revision: String,
}

impl LivePostgresExecutor {
    pub fn connect(config: &PostgresRecordMaterializerConfig) -> Result<Self> {
        config.validate()?;
        let dsn = env::var(&config.connection_environment).with_context(|| {
            format!(
                "postgres connection environment variable `{}` is not set",
                config.connection_environment
            )
        })?;
        let mut client = Client::connect(&dsn, NoTls).with_context(|| {
            format!(
                "failed to connect to postgres using `{}`",
                config.connection_environment
            )
        })?;
        client
            .batch_execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            .context("failed to start postgres corpus snapshot")?;
        let row = client
            .query_one(&config.snapshot_statement, &[])
            .context("failed to capture postgres source snapshot")?;
        let snapshot_revision = scalar_column(&row, 0)
            .context("postgres snapshot_statement must return one scalar value")?;
        Ok(Self {
            client: Mutex::new(client),
            snapshot_revision,
        })
    }
}

impl PostgresExecutor for LivePostgresExecutor {
    fn snapshot_revision(&self) -> Result<String> {
        Ok(self.snapshot_revision.clone())
    }

    fn fetch(
        &self,
        statement: &str,
        columns: &PostgresColumnMapping,
        record_keys: &[String],
    ) -> Result<Vec<PostgresRecordRow>> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| anyhow::anyhow!("postgres materializer lock is poisoned"))?;
        let rows = client
            .query(statement, &[&record_keys])
            .context("postgres record materialization query failed")?;
        rows.iter().map(|row| decode_row(row, columns)).collect()
    }
}

fn decode_row(row: &Row, columns: &PostgresColumnMapping) -> Result<PostgresRecordRow> {
    let record_key = row
        .try_get::<_, String>(columns.record_key.as_str())
        .with_context(|| format!("invalid postgres `{}` column", columns.record_key))?;
    let text = row
        .try_get::<_, String>(columns.text.as_str())
        .with_context(|| format!("invalid postgres `{}` column", columns.text))?;
    let uris = match &columns.uris {
        Some(column) => decode_uris(row, column)?,
        None => Vec::new(),
    };
    let metadata = match &columns.metadata {
        Some(column) => {
            let value = row
                .try_get::<_, Value>(column.as_str())
                .with_context(|| format!("invalid postgres `{column}` JSON column"))?;
            let Value::Object(values) = value else {
                bail!("postgres metadata column `{column}` must be a JSON object");
            };
            values.into_iter().collect()
        }
        None => BTreeMap::new(),
    };
    Ok(PostgresRecordRow {
        record_key,
        text,
        uris,
        metadata,
    })
}

fn decode_uris(row: &Row, column: &str) -> Result<Vec<String>> {
    if let Ok(values) = row.try_get::<_, Vec<String>>(column) {
        return Ok(values);
    }
    let value = row
        .try_get::<_, Value>(column)
        .with_context(|| format!("postgres URI column `{column}` must be text[] or JSON"))?;
    match value {
        Value::Null => Ok(Vec::new()),
        Value::String(value) => Ok(vec![value]),
        Value::Array(values) => values
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("postgres URI column `{column}` has non-string item"))
            })
            .collect(),
        _ => bail!("postgres URI column `{column}` must be a string or array"),
    }
}

fn scalar_column(row: &Row, index: usize) -> Option<String> {
    row.try_get::<_, String>(index)
        .ok()
        .or_else(|| {
            row.try_get::<_, i64>(index)
                .ok()
                .map(|value| value.to_string())
        })
        .or_else(|| {
            row.try_get::<_, i32>(index)
                .ok()
                .map(|value| value.to_string())
        })
}

pub struct PostgresRecordMaterializer<E = LivePostgresExecutor> {
    config: PostgresRecordMaterializerConfig,
    executor: E,
}

impl PostgresRecordMaterializer<LivePostgresExecutor> {
    pub fn connect(config: PostgresRecordMaterializerConfig) -> Result<Self> {
        let executor = LivePostgresExecutor::connect(&config)?;
        Ok(Self { config, executor })
    }
}

impl<E> PostgresRecordMaterializer<E>
where
    E: PostgresExecutor,
{
    pub fn with_executor(config: PostgresRecordMaterializerConfig, executor: E) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, executor })
    }
}

impl<E> RecordMaterializer for PostgresRecordMaterializer<E>
where
    E: PostgresExecutor,
{
    fn name(&self) -> &str {
        "postgres"
    }

    fn configuration(&self) -> Result<Value> {
        serde_json::to_value(&self.config)
            .context("failed to serialize postgres materializer configuration")
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "postgres".to_owned(),
            revision: self.executor.snapshot_revision()?,
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        let keys: Vec<_> = hits.iter().map(|hit| hit.record_key.clone()).collect();
        let rows = self
            .executor
            .fetch(&self.config.statement, &self.config.columns, &keys)?;
        let hits_by_key: HashMap<_, _> = hits
            .iter()
            .map(|hit| (hit.record_key.as_str(), hit))
            .collect();
        let mut found = std::collections::HashSet::new();
        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let hit = hits_by_key.get(row.record_key.as_str()).with_context(|| {
                format!(
                    "postgres returned unrequested record key `{}`",
                    row.record_key
                )
            })?;
            ensure!(
                found.insert(row.record_key.clone()),
                "postgres returned duplicate record key `{}`",
                row.record_key
            );
            let mut metadata = hit.metadata.clone();
            metadata.extend(row.metadata);
            records.push(MaterializedRecord {
                record_key: row.record_key,
                text: row.text,
                uris: if row.uris.is_empty() {
                    hit.uris.clone()
                } else {
                    row.uris
                },
                metadata,
            });
        }
        if self.config.require_every_record {
            ensure!(
                found.len() == hits.len(),
                "postgres materialized {} of {} requested records",
                found.len(),
                hits.len()
            );
        }
        Ok(records)
    }
}
