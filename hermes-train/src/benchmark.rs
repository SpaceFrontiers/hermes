//! Deterministic execution of public and sealed model acceptance benchmarks.
//!
//! A suite manifest and every data artifact are content addressed.  The runner
//! never downloads data: callers materialize public or sealed suites locally,
//! then provide the expected manifest digest out of band.  Baseline and
//! candidate evaluations receive the same model seed, example-order seed, and
//! GPU-hour allowance.  Raw measurements can be converted directly into the
//! higher-is-better paired inputs consumed by [`crate::acceptance`]. The
//! content-addressed run is audit evidence and therefore retains sealed case
//! ids and scores, but never suite examples; downstream promotion reports
//! expose sealed results only as an aggregate gate.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::acceptance::{
    ACCEPTANCE_SUITE_VERSION, AcceptancePolicy, AcceptanceSuite, BenchmarkCase, BenchmarkFamily,
    CaseResult, ExactResumeArtifact, ExactResumeEvidence, PairedRun, PromotionReport,
    ResourceComparison, SuiteVisibility, VerifiedPromotionContext, evaluate_with_verified_context,
};
use crate::metrics::metric_log_digests_from_bytes;
use crate::qat_candidate::open_qat_candidate;

pub const BENCHMARK_MANIFEST_VERSION: u32 = 2;
pub const MINIMUM_PAIRED_SEEDS: usize = 3;

/// Whether a catalog entry gates current promotion or belongs to a later
/// language/long-context sweep from the sleep paper reproduction plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogStage {
    Promotion,
    Sleep,
    LaterSweep,
}

impl CatalogStage {
    pub fn required_now(self) -> bool {
        !matches!(self, Self::LaterSweep)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub family: BenchmarkFamily,
    pub stage: CatalogStage,
}

/// The minimum capability and in-model-sleep catalog.  `LaterSweep` entries
/// are intentionally registered now so result identifiers do not drift when
/// the more expensive language and long-context sweeps are enabled.
pub fn required_catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry {
            id: "pretraining_causal",
            family: BenchmarkFamily::Pretraining,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "summarization",
            family: BenchmarkFamily::Summarization,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "retrieval_representation_ranking",
            family: BenchmarkFamily::Retrieval,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "retrieval_planning",
            family: BenchmarkFamily::RetrievalPlanning,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "reasoning_qa",
            family: BenchmarkFamily::Reasoning,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "pairwise_preference",
            family: BenchmarkFamily::Preference,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "verifiable_rl",
            family: BenchmarkFamily::VerifiableRl,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "synthetic_retention_smoke",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "clinc_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "banking77_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "mk_niah",
            family: BenchmarkFamily::LongContext,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "qasper",
            family: BenchmarkFamily::LongContext,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "squad_no_context",
            family: BenchmarkFamily::FactualIncorporation,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "arc_dreaming",
            family: BenchmarkFamily::Dreaming,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "manchu_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::LaterSweep,
        },
        CatalogEntry {
            id: "kalamang_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::LaterSweep,
        },
        CatalogEntry {
            id: "babilong",
            family: BenchmarkFamily::LongContext,
            stage: CatalogStage::LaterSweep,
        },
    ]
}

/// Capacity-, compute-, and mechanism-matched variants required by the sleep
/// experiment matrix.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AblationId {
    WakeOnly,
    StaticReplay,
    CmsWithoutKnowledgeSeeding,
    FixedCapacityOnPolicyDistillation,
    ExpansionMatchedDistillation,
    ConsolidationWithoutDreaming,
    NoSemanticReward,
    NoImitationReward,
    NoGradientSelection,
    NoRandomExpert,
    StaticallyAvailableEqualCapacityMoe,
}

impl AblationId {
    pub const ALL: [Self; 11] = [
        Self::WakeOnly,
        Self::StaticReplay,
        Self::CmsWithoutKnowledgeSeeding,
        Self::FixedCapacityOnPolicyDistillation,
        Self::ExpansionMatchedDistillation,
        Self::ConsolidationWithoutDreaming,
        Self::NoSemanticReward,
        Self::NoImitationReward,
        Self::NoGradientSelection,
        Self::NoRandomExpert,
        Self::StaticallyAvailableEqualCapacityMoe,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WakeOnly => "wake_only",
            Self::StaticReplay => "static_replay",
            Self::CmsWithoutKnowledgeSeeding => "cms_without_knowledge_seeding",
            Self::FixedCapacityOnPolicyDistillation => "fixed_capacity_on_policy_distillation",
            Self::ExpansionMatchedDistillation => "expansion_matched_distillation",
            Self::ConsolidationWithoutDreaming => "consolidation_without_dreaming",
            Self::NoSemanticReward => "no_semantic_reward",
            Self::NoImitationReward => "no_imitation_reward",
            Self::NoGradientSelection => "no_gradient_selection",
            Self::NoRandomExpert => "no_random_expert",
            Self::StaticallyAvailableEqualCapacityMoe => "statically_available_equal_capacity_moe",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    #[default]
    Maximize,
    Minimize,
}

impl MetricDirection {
    fn acceptance_score(self, raw: f64) -> f64 {
        match self {
            Self::Maximize => raw,
            Self::Minimize => -raw,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkArtifact {
    /// Local file, resolved relative to the suite manifest.
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSpec {
    /// Globally unique result id.  It may identify a public/sealed variant.
    pub id: String,
    /// Stable catalog id; defaults to `id` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    pub family: BenchmarkFamily,
    pub metric: String,
    #[serde(default)]
    pub direction: MetricDirection,
    pub stable_anchor: bool,
    pub artifact: BenchmarkArtifact,
    /// Versioned evaluator options.  Trainer core does not interpret them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evaluator: BTreeMap<String, serde_json::Value>,
}

impl BenchmarkSpec {
    pub fn catalog_id(&self) -> &str {
        self.catalog_id.as_deref().unwrap_or(&self.id)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSuiteManifest {
    pub version: u32,
    pub suite_id: String,
    pub visibility: SuiteVisibility,
    pub cases: Vec<BenchmarkSpec>,
}

impl BenchmarkSuiteManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == BENCHMARK_MANIFEST_VERSION,
            "unsupported benchmark manifest version {}",
            self.version
        );
        ensure!(
            !self.suite_id.trim().is_empty(),
            "benchmark suite_id is empty"
        );
        ensure!(
            !self.cases.is_empty(),
            "benchmark suite `{}` has no cases",
            self.suite_id
        );
        let mut case_ids = BTreeSet::new();
        for case in &self.cases {
            ensure!(
                !case.id.trim().is_empty(),
                "benchmark suite `{}` contains an empty case id",
                self.suite_id
            );
            ensure!(
                case_ids.insert(case.id.as_str()),
                "benchmark suite `{}` repeats case `{}`",
                self.suite_id,
                case.id
            );
            ensure!(
                !case.catalog_id().trim().is_empty(),
                "benchmark case `{}` has an empty catalog_id",
                case.id
            );
            ensure!(
                !case.metric.trim().is_empty(),
                "benchmark case `{}` has an empty metric",
                case.id
            );
            ensure!(
                !case.artifact.path.as_os_str().is_empty(),
                "benchmark case `{}` has an empty artifact path",
                case.id
            );
            validate_sha256(&case.artifact.sha256)
                .with_context(|| format!("invalid artifact hash for `{}`", case.id))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VerifiedArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

/// A suite that passed both manifest and per-artifact digest verification.
#[derive(Clone, Debug)]
pub struct VerifiedBenchmarkSuite {
    pub manifest: BenchmarkSuiteManifest,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    artifacts: BTreeMap<String, VerifiedArtifact>,
}

impl VerifiedBenchmarkSuite {
    pub fn load(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        validate_sha256(expected_sha256).context("invalid expected manifest hash")?;
        let path = path.as_ref();
        let bytes = read_regular_file(path, "benchmark manifest")?;
        let actual_sha256 = sha256_hex(&bytes);
        ensure!(
            actual_sha256.eq_ignore_ascii_case(expected_sha256),
            "benchmark manifest hash mismatch for {}: expected {}, got {}",
            path.display(),
            expected_sha256,
            actual_sha256
        );
        let manifest: BenchmarkSuiteManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid benchmark manifest {}", path.display()))?;
        manifest.validate()?;

        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut artifacts = BTreeMap::new();
        for case in &manifest.cases {
            let artifact_path = if case.artifact.path.is_absolute() {
                case.artifact.path.clone()
            } else {
                base.join(&case.artifact.path)
            };
            let artifact_bytes = read_regular_file(
                &artifact_path,
                &format!("artifact for benchmark `{}`", case.id),
            )?;
            let actual = sha256_hex(&artifact_bytes);
            ensure!(
                actual.eq_ignore_ascii_case(&case.artifact.sha256),
                "artifact hash mismatch for benchmark `{}`: expected {}, got {}",
                case.id,
                case.artifact.sha256,
                actual
            );
            artifacts.insert(
                case.id.clone(),
                VerifiedArtifact {
                    path: artifact_path,
                    sha256: actual,
                    bytes: artifact_bytes.len() as u64,
                },
            );
        }
        Ok(Self {
            manifest,
            manifest_path: path.to_path_buf(),
            manifest_sha256: actual_sha256,
            artifacts,
        })
    }

    pub fn artifact(&self, case_id: &str) -> Option<&VerifiedArtifact> {
        self.artifacts.get(case_id)
    }

    /// Re-check files immediately before an expensive run, closing the gap
    /// between suite loading and evaluator access.
    pub fn verify_contents(&self) -> Result<()> {
        let manifest_bytes = read_regular_file(&self.manifest_path, "benchmark manifest")?;
        let actual_manifest = sha256_hex(&manifest_bytes);
        ensure!(
            actual_manifest == self.manifest_sha256,
            "benchmark manifest changed after verification: {}",
            self.manifest_path.display()
        );
        for (case_id, artifact) in &self.artifacts {
            let bytes = read_regular_file(
                &artifact.path,
                &format!("artifact for benchmark `{case_id}`"),
            )?;
            let actual = sha256_hex(&bytes);
            ensure!(
                actual == artifact.sha256 && bytes.len() as u64 == artifact.bytes,
                "artifact for benchmark `{case_id}` changed after verification"
            );
        }
        Ok(())
    }
}

/// Check that manifests cover the fixed catalog with the expected benchmark
/// family.  Later sweeps can be required explicitly for a full reproduction.
pub fn validate_catalog_coverage(
    suites: &[VerifiedBenchmarkSuite],
    include_later_sweeps: bool,
) -> Result<()> {
    let mut cases = BTreeMap::new();
    let mut visibilities = BTreeSet::new();
    for suite in suites {
        visibilities.insert(suite.manifest.visibility);
        for case in &suite.manifest.cases {
            let key = (case.catalog_id(), suite.manifest.visibility);
            if let Some(previous) = cases.insert(key, case) {
                ensure!(
                    previous.family == case.family,
                    "catalog id `{}` is assigned conflicting {:?} families {:?} and {:?}",
                    case.catalog_id(),
                    suite.manifest.visibility,
                    previous.family,
                    case.family
                );
            }
        }
    }
    ensure!(
        visibilities.contains(&SuiteVisibility::Public)
            && visibilities.contains(&SuiteVisibility::Sealed),
        "promotion benchmarks require both public and sealed suites"
    );
    for entry in required_catalog() {
        if !entry.stage.required_now() && !include_later_sweeps {
            continue;
        }
        for visibility in [SuiteVisibility::Public, SuiteVisibility::Sealed] {
            let case = cases.get(&(entry.id, visibility)).with_context(|| {
                format!(
                    "benchmark catalog is missing {} `{}`",
                    match visibility {
                        SuiteVisibility::Public => "public",
                        SuiteVisibility::Sealed => "sealed",
                    },
                    entry.id
                )
            })?;
            ensure!(
                case.family == entry.family,
                "benchmark catalog `{}` has {:?} family {:?}, expected {:?}",
                entry.id,
                visibility,
                case.family,
                entry.family
            );
        }
    }
    Ok(())
}

pub const TRAINING_ACCOUNTING_VERSION: u32 = 1;
pub const TRAINING_ACCOUNTING_FILE: &str = "training-accounting.json";
const CHECKPOINT_WEIGHTS_FILE: &str = "weights.safetensors";

/// Resource measurements sealed *inside* a checkpoint generation. This layer
/// intentionally does not contain the generation-manifest hash, avoiding a
/// cycle while letting the manifest authenticate every accounting field.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingAccounting {
    pub version: u32,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub weights_bytes: u64,
    pub weights_sha256: String,
}

impl TrainingAccounting {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == TRAINING_ACCOUNTING_VERSION,
            "unsupported training-accounting version {}",
            self.version
        );
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "training accounting has invalid measured GPU hours"
        );
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "training accounting has an empty parameter count"
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "training accounting routes more parameters than it stores"
        );
        ensure!(self.weights_bytes > 0, "training accounting has no weights");
        validate_sha256(&self.weights_sha256)
            .context("invalid model-weights hash in training accounting")?;
        Ok(())
    }
}

/// Content-addressed attestation emitted after sealing one immutable
/// checkpoint. The accounting itself is already inside that generation; this
/// outer artifact binds its manifest identity without a circular hash.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingEvidence {
    pub version: u32,
    pub checkpoint_manifest_sha256: String,
    pub accounting_sha256: String,
    /// Built-in single-device trainer: sum of committed optimizer-window
    /// elapsed time divided by 3600, including stalls (not GPU-busy time).
    pub training_gpu_hours: f64,
    /// Exact stored parameters in the instantiated model module.
    pub parameters: u64,
    /// All non-expert parameters plus routers/shared experts and ordinary
    /// top-k expert parameter-equivalent.
    pub routed_active_parameters: u64,
    /// Byte length of the model weights artifact only; optimizer/trainer state
    /// is excluded from model-capacity comparisons.
    pub stored_bytes: u64,
    pub weights_sha256: String,
}

impl TrainingEvidence {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported training-evidence version {}",
            self.version
        );
        validate_sha256(&self.checkpoint_manifest_sha256)
            .context("invalid checkpoint hash in training evidence")?;
        validate_sha256(&self.accounting_sha256)
            .context("invalid accounting hash in training evidence")?;
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "training evidence has invalid measured GPU hours"
        );
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "training evidence has an empty parameter count"
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "training evidence routes more parameters than it stores"
        );
        ensure!(
            self.stored_bytes > 0,
            "training evidence has no stored bytes"
        );
        validate_sha256(&self.weights_sha256)
            .context("invalid model-weights hash in training evidence")?;
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointManifest {
    version: u32,
    training_state_version: u32,
    global_step: usize,
    #[serde(rename = "phase")]
    _phase: usize,
    phase_id: String,
    files: Vec<CheckpointManifestFile>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

/// Concrete model representation evaluated by an acceptance backend. This is
/// required in the current schema: callers may not rely on a historical
/// implicit full-precision default.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelRepresentationTarget {
    FullPrecision,
    Hquant {
        /// Canonical `candidate.json` produced by the QAT candidate publisher.
        candidate_manifest: PathBuf,
        /// Prefixed content identity returned by that publisher.
        candidate_manifest_sha256: String,
    },
}

/// Path-free representation identity retained in resource and promotion
/// receipts. The QAT candidate manifest transitively authenticates its source
/// checkpoint, recipe, archive manifest, and every archive member.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelRepresentationIdentity {
    FullPrecision,
    Hquant { candidate_manifest_sha256: String },
}

/// Fully verified model transport supplied to the evaluator. HQUANT remains
/// an injected backend responsibility; this contract exposes the sealed
/// archive directly and never substitutes a dequantized FP evaluation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum VerifiedModelRepresentation {
    FullPrecision {
        weights: PathBuf,
        weights_sha256: String,
        stored_bytes: u64,
    },
    Hquant {
        candidate_manifest: PathBuf,
        candidate_manifest_sha256: String,
        source_weights: PathBuf,
        source_weights_sha256: String,
        archive: PathBuf,
        archive_manifest: PathBuf,
        archive_manifest_sha256: String,
        archive_content_sha256: String,
        stored_bytes: u64,
    },
}

/// Immutable model identity passed to an evaluator. The checkpoint and
/// training evidence describe how the model was trained; `representation`
/// names the exact FP or sealed HQUANT bytes whose quality is measured.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkTarget {
    pub id: String,
    pub checkpoint_manifest: PathBuf,
    pub checkpoint_manifest_sha256: String,
    pub training_evidence: PathBuf,
    /// Content identity of the immutable trainer metrics/accounting artifact.
    pub training_evidence_sha256: String,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub stored_bytes: u64,
    pub representation: ModelRepresentationTarget,
}

impl BenchmarkTarget {
    /// Resolve artifact paths against the target JSON that declared them.
    /// Callers must do this before verification so execution never depends on
    /// the process working directory.
    pub fn resolve_paths(&mut self, source: &Path) {
        let base = source.parent().unwrap_or_else(|| Path::new("."));
        if self.checkpoint_manifest.is_relative() {
            self.checkpoint_manifest = base.join(&self.checkpoint_manifest);
        }
        if self.training_evidence.is_relative() {
            self.training_evidence = base.join(&self.training_evidence);
        }
        if let ModelRepresentationTarget::Hquant {
            candidate_manifest, ..
        } = &mut self.representation
            && candidate_manifest.is_relative()
        {
            *candidate_manifest = base.join(&*candidate_manifest);
        }
    }

    pub fn representation_identity(&self) -> ModelRepresentationIdentity {
        match &self.representation {
            ModelRepresentationTarget::FullPrecision => ModelRepresentationIdentity::FullPrecision,
            ModelRepresentationTarget::Hquant {
                candidate_manifest_sha256,
                ..
            } => ModelRepresentationIdentity::Hquant {
                candidate_manifest_sha256: candidate_manifest_sha256.clone(),
            },
        }
    }

    fn validate_identity(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "benchmark target id is empty");
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "benchmark target `{}` has an empty parameter count",
            self.id
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "benchmark target `{}` routes more parameters than it stores",
            self.id
        );
        validate_sha256(&self.checkpoint_manifest_sha256)
            .with_context(|| format!("invalid checkpoint hash for `{}`", self.id))?;
        validate_sha256(&self.training_evidence_sha256)
            .with_context(|| format!("invalid training evidence hash for `{}`", self.id))?;
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "benchmark target `{}` has invalid measured training GPU hours",
            self.id
        );
        ensure!(
            self.stored_bytes > 0,
            "benchmark target `{}` has no stored-byte measurement",
            self.id
        );
        if let ModelRepresentationTarget::Hquant {
            candidate_manifest,
            candidate_manifest_sha256,
        } = &self.representation
        {
            ensure!(
                !candidate_manifest.as_os_str().is_empty(),
                "HQUANT benchmark target `{}` has no candidate manifest",
                self.id
            );
            validate_prefixed_sha256(candidate_manifest_sha256)
                .with_context(|| format!("invalid HQUANT candidate hash for `{}`", self.id))?;
        }
        Ok(())
    }

    pub fn verify(&self) -> Result<VerifiedModelRepresentation> {
        self.validate_identity()?;
        let manifest_bytes = read_regular_file(
            &self.checkpoint_manifest,
            &format!("checkpoint manifest for `{}`", self.id),
        )?;
        let actual = sha256_hex(&manifest_bytes);
        ensure!(
            actual.eq_ignore_ascii_case(&self.checkpoint_manifest_sha256),
            "checkpoint manifest hash mismatch for `{}`: expected {}, got {}",
            self.id,
            self.checkpoint_manifest_sha256,
            actual
        );
        ensure!(
            self.checkpoint_manifest
                .file_name()
                .and_then(|name| name.to_str())
                == Some("generation-manifest.json"),
            "checkpoint manifest for `{}` must be named generation-manifest.json",
            self.id
        );
        let generation = self
            .checkpoint_manifest
            .parent()
            .context("checkpoint manifest has no generation directory")?;
        ensure!(
            generation.file_name().and_then(|name| name.to_str())
                == Some(&format!("sha256-{}", self.checkpoint_manifest_sha256)),
            "checkpoint generation directory for `{}` does not match its manifest hash",
            self.id
        );
        let manifest: CheckpointManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("invalid checkpoint manifest for `{}`", self.id))?;
        validate_checkpoint_manifest(&manifest, &self.id)?;

        let evidence_bytes = read_regular_file(
            &self.training_evidence,
            &format!("training evidence for `{}`", self.id),
        )?;
        let evidence_sha256 = sha256_hex(&evidence_bytes);
        ensure!(
            evidence_sha256.eq_ignore_ascii_case(&self.training_evidence_sha256),
            "training evidence hash mismatch for `{}`: expected {}, got {}",
            self.id,
            self.training_evidence_sha256,
            evidence_sha256
        );
        ensure!(
            self.training_evidence
                .file_name()
                .and_then(|name| name.to_str())
                == Some(&format!("sha256-{evidence_sha256}.json")),
            "training evidence for `{}` is not stored under its content-addressed name",
            self.id
        );
        let evidence: TrainingEvidence =
            serde_json::from_slice(&evidence_bytes).with_context(|| {
                format!(
                    "invalid training evidence for `{}` at {}",
                    self.id,
                    self.training_evidence.display()
                )
            })?;
        evidence.validate()?;
        ensure!(
            evidence
                .checkpoint_manifest_sha256
                .eq_ignore_ascii_case(&self.checkpoint_manifest_sha256),
            "training evidence for `{}` addresses a different checkpoint",
            self.id
        );

        let accounting_entry = manifest_file(&manifest, TRAINING_ACCOUNTING_FILE, &self.id)?;
        let accounting_bytes = read_manifest_file(
            generation,
            accounting_entry,
            &format!("training accounting for `{}`", self.id),
        )?;
        let accounting_sha256 = sha256_hex(&accounting_bytes);
        ensure!(
            accounting_sha256.eq_ignore_ascii_case(&evidence.accounting_sha256),
            "training evidence for `{}` is not authenticated by its checkpoint manifest",
            self.id
        );
        let accounting: TrainingAccounting = serde_json::from_slice(&accounting_bytes)
            .with_context(|| format!("invalid training accounting for `{}`", self.id))?;
        accounting.validate()?;

        let weights_entry = manifest_file(&manifest, CHECKPOINT_WEIGHTS_FILE, &self.id)?;
        let weights = read_manifest_file(
            generation,
            weights_entry,
            &format!("model weights for `{}`", self.id),
        )?;
        let weights_sha256 = sha256_hex(&weights);
        let actual_parameters = safetensor_parameter_count(&weights)
            .with_context(|| format!("invalid model weights for `{}`", self.id))?;
        ensure!(
            accounting.weights_bytes == weights.len() as u64
                && accounting
                    .weights_sha256
                    .eq_ignore_ascii_case(&weights_sha256)
                && accounting.parameters == actual_parameters,
            "sealed training accounting for `{}` differs from its actual weights",
            self.id
        );
        ensure!(
            same_f64(evidence.training_gpu_hours, self.training_gpu_hours)
                && evidence.parameters == self.parameters
                && evidence.routed_active_parameters == self.routed_active_parameters
                && same_f64(evidence.training_gpu_hours, accounting.training_gpu_hours)
                && evidence.parameters == accounting.parameters
                && evidence.routed_active_parameters == accounting.routed_active_parameters
                && evidence.stored_bytes == accounting.weights_bytes
                && evidence
                    .weights_sha256
                    .eq_ignore_ascii_case(&accounting.weights_sha256),
            "benchmark target `{}` resource claims differ from its verified training evidence",
            self.id
        );
        match &self.representation {
            ModelRepresentationTarget::FullPrecision => {
                ensure!(
                    self.stored_bytes == accounting.weights_bytes,
                    "full-precision benchmark target `{}` stored bytes differ from its weights",
                    self.id
                );
                Ok(VerifiedModelRepresentation::FullPrecision {
                    weights: generation.join(CHECKPOINT_WEIGHTS_FILE),
                    weights_sha256,
                    stored_bytes: self.stored_bytes,
                })
            }
            ModelRepresentationTarget::Hquant {
                candidate_manifest,
                candidate_manifest_sha256,
            } => {
                ensure!(
                    candidate_manifest
                        .file_name()
                        .and_then(|name| name.to_str())
                        == Some("candidate.json"),
                    "HQUANT benchmark target `{}` must address candidate.json",
                    self.id
                );
                let candidate_root = candidate_manifest
                    .parent()
                    .context("HQUANT candidate manifest has no candidate directory")?;
                let publication = open_qat_candidate(candidate_root).with_context(|| {
                    format!("invalid sealed HQUANT candidate for `{}`", self.id)
                })?;
                ensure!(
                    publication.candidate_manifest_path == *candidate_manifest
                        && publication.candidate_manifest_sha256 == *candidate_manifest_sha256,
                    "HQUANT candidate manifest identity mismatch for `{}`",
                    self.id
                );
                ensure!(
                    publication.weights_sha256
                        == format!("sha256:{}", accounting.weights_sha256.to_ascii_lowercase()),
                    "HQUANT candidate for `{}` was produced from different source weights",
                    self.id
                );
                ensure!(
                    publication.metrics.archive_weight_elements == self.parameters,
                    "HQUANT candidate for `{}` has a different parameter inventory",
                    self.id
                );
                ensure!(
                    publication.metrics.archive_weight_bytes == self.stored_bytes,
                    "HQUANT benchmark target `{}` stored bytes differ from its validated archive members",
                    self.id
                );
                Ok(VerifiedModelRepresentation::Hquant {
                    candidate_manifest: publication.candidate_manifest_path,
                    candidate_manifest_sha256: publication.candidate_manifest_sha256,
                    source_weights: publication.weights_path,
                    source_weights_sha256: publication.weights_sha256,
                    archive: publication.archive_path,
                    archive_manifest: publication.archive_manifest_path,
                    archive_manifest_sha256: publication.archive_manifest_sha256,
                    archive_content_sha256: publication.archive_content_sha256,
                    stored_bytes: self.stored_bytes,
                })
            }
        }
    }
}

fn validate_checkpoint_manifest(manifest: &CheckpointManifest, target: &str) -> Result<()> {
    ensure!(
        manifest.version == 1,
        "unsupported checkpoint manifest version"
    );
    ensure!(
        manifest.training_state_version == 2,
        "checkpoint for `{target}` is not a WorkflowV2 generation"
    );
    ensure!(
        !manifest.phase_id.trim().is_empty(),
        "checkpoint for `{target}` has an empty phase id"
    );
    let mut paths = BTreeSet::new();
    for file in &manifest.files {
        validate_manifest_relative_path(&file.path)?;
        ensure!(file.bytes > 0, "checkpoint file `{}` is empty", file.path);
        validate_sha256(&file.sha256)
            .with_context(|| format!("invalid hash for checkpoint file `{}`", file.path))?;
        ensure!(
            paths.insert(file.path.as_str()),
            "checkpoint manifest repeats `{}`",
            file.path
        );
    }
    Ok(())
}

fn validate_manifest_relative_path(path: &str) -> Result<()> {
    ensure!(
        !path.is_empty(),
        "checkpoint manifest contains an empty path"
    );
    let path = Path::new(path);
    ensure!(
        !path.is_absolute(),
        "checkpoint manifest path must be relative"
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "checkpoint manifest path must not contain prefixes, `.` or `..`"
    );
    Ok(())
}

fn manifest_file<'a>(
    manifest: &'a CheckpointManifest,
    path: &str,
    target: &str,
) -> Result<&'a CheckpointManifestFile> {
    let matches = manifest
        .files
        .iter()
        .filter(|file| file.path == path)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "checkpoint for `{target}` must contain exactly one `{path}`"
    );
    Ok(matches[0])
}

fn read_manifest_file(
    generation: &Path,
    entry: &CheckpointManifestFile,
    label: &str,
) -> Result<Vec<u8>> {
    let path = generation.join(&entry.path);
    let bytes = read_regular_file(&path, label)?;
    ensure!(
        bytes.len() as u64 == entry.bytes && sha256_hex(&bytes).eq_ignore_ascii_case(&entry.sha256),
        "{label} does not match the checkpoint manifest"
    );
    Ok(bytes)
}

fn safetensor_parameter_count(bytes: &[u8]) -> Result<u64> {
    let tensors = safetensors::SafeTensors::deserialize(bytes)?;
    let mut total = 0_u64;
    for (_, tensor) in tensors.tensors() {
        let elements = tensor
            .shape()
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(
                        (*dimension)
                            .try_into()
                            .context("tensor dimension exceeds u64")?,
                    )
                    .context("tensor element count overflows u64")
            })?;
        total = total
            .checked_add(elements)
            .context("model parameter count overflows u64")?;
    }
    ensure!(total > 0, "model weights contain no tensors");
    Ok(total)
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let absolute;
    let path = if path.is_absolute() {
        path
    } else {
        absolute = std::env::current_dir()
            .context("failed to resolve current directory for benchmark artifact")?
            .join(path);
        absolute.as_path()
    };
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        let metadata = fs::symlink_metadata(parent)
            .with_context(|| format!("failed to inspect {label} parent {}", parent.display()))?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "{label} parent {} must be a real directory, not a symlink",
            parent.display()
        );
    }
    let before = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} at {}", path.display()))?;
    ensure!(
        before.is_file() && !before.file_type().is_symlink(),
        "{label} must be a regular non-symlink file"
    );
    let mut file = File::open(path)
        .with_context(|| format!("failed to open {label} at {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label} at {}", path.display()))?;
    ensure!(opened.is_file(), "{label} must open as a regular file");

    // Compare the lstat identity before and after open with fstat on the
    // actual handle. This rejects a regular-file -> symlink/other-file swap in
    // the check/open gap while retaining a stable handle for the read itself.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            before.dev() == opened.dev() && before.ino() == opened.ino(),
            "{label} changed while it was opened"
        );
    }
    let after = fs::symlink_metadata(path)
        .with_context(|| format!("failed to re-inspect {label} at {}", path.display()))?;
    ensure!(
        after.is_file() && !after.file_type().is_symlink(),
        "{label} became a symlink or non-file while it was opened"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            after.dev() == opened.dev() && after.ino() == opened.ino(),
            "{label} changed while it was opened"
        );
    }

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label} at {}", path.display()))?;
    let final_metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect read {label} at {}", path.display()))?;
    ensure!(
        final_metadata.len() == bytes.len() as u64,
        "{label} changed length while it was read"
    );
    Ok(bytes)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunConfig {
    pub evaluator_id: String,
    pub evaluator_version: String,
    /// Strictly increasing order; the order itself is part of run identity.
    pub paired_seeds: Vec<u64>,
    /// Controls example ordering inside every evaluator.  A case-specific seed
    /// is derived from it and shared by baseline and candidate.
    pub order_seed: u64,
    /// Equal hard allowance for each (case, seed, target) evaluation.
    pub gpu_hours_per_evaluation: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ablation: Option<AblationId>,
}

impl BenchmarkRunConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.evaluator_id.trim().is_empty() && !self.evaluator_version.trim().is_empty(),
            "benchmark evaluator id and version are required"
        );
        ensure!(
            self.paired_seeds.len() >= MINIMUM_PAIRED_SEEDS,
            "benchmarks require at least {MINIMUM_PAIRED_SEEDS} paired seeds"
        );
        ensure!(
            self.paired_seeds.windows(2).all(|pair| pair[0] < pair[1]),
            "paired benchmark seeds must be unique and strictly increasing"
        );
        ensure!(
            self.gpu_hours_per_evaluation.is_finite() && self.gpu_hours_per_evaluation > 0.0,
            "gpu_hours_per_evaluation must be finite and positive"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetRole {
    Baseline,
    Candidate,
}

/// Fully deterministic evaluator input. The evaluator must execute `model`,
/// consume examples in `example_order_seed` order, and stop at
/// `max_gpu_hours`. `target` supplies training lineage and resource identity;
/// it is not permission to substitute the checkpoint's FP weights for an
/// HQUANT `model`.
pub struct EvaluationRequest<'a> {
    pub suite_id: &'a str,
    pub visibility: SuiteVisibility,
    pub case: &'a BenchmarkSpec,
    pub artifact: &'a VerifiedArtifact,
    pub target: &'a BenchmarkTarget,
    /// Verified bytes the backend must execute for this measurement.
    pub model: &'a VerifiedModelRepresentation,
    pub role: TargetRole,
    pub model_seed: u64,
    pub example_order_seed: u64,
    pub pair_ordinal: usize,
    pub max_gpu_hours: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationMeasurement {
    pub score: f64,
    pub gpu_hours: f64,
    pub examples: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
}

impl EvaluationMeasurement {
    pub fn validate(&self, case_id: &str, limit: f64) -> Result<()> {
        ensure!(
            self.score.is_finite(),
            "benchmark `{case_id}` returned a non-finite score"
        );
        ensure!(
            self.gpu_hours.is_finite() && self.gpu_hours >= 0.0,
            "benchmark `{case_id}` returned invalid GPU hours"
        );
        // Accommodate a few floating point ulps at the accounting boundary.
        ensure!(
            self.gpu_hours <= limit + f64::EPSILON * limit.max(1.0) * 8.0,
            "benchmark `{case_id}` exceeded its fixed GPU-hour allowance: {} > {}",
            self.gpu_hours,
            limit
        );
        ensure!(
            self.examples > 0,
            "benchmark `{case_id}` evaluated no examples"
        );
        ensure!(
            self.metrics.values().all(|value| value.is_finite()),
            "benchmark `{case_id}` returned a non-finite auxiliary metric"
        );
        Ok(())
    }
}

/// Implementations can wrap an in-process harness or invoke a local worker.
/// They must reject a representation they cannot execute. Artifact
/// materialization and network access deliberately remain outside this
/// interface.
pub trait BenchmarkEvaluator {
    fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawPairedResult {
    pub seed: u64,
    pub example_order_seed: u64,
    pub baseline: EvaluationMeasurement,
    pub candidate: EvaluationMeasurement,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawCaseResult {
    pub suite_id: String,
    pub case_id: String,
    pub catalog_id: String,
    pub visibility: SuiteVisibility,
    pub family: BenchmarkFamily,
    pub metric: String,
    pub direction: MetricDirection,
    pub stable_anchor: bool,
    pub pairs: Vec<RawPairedResult>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunMetadata {
    pub evaluator_id: String,
    pub evaluator_version: String,
    pub paired_seeds: Vec<u64>,
    pub order_seed: u64,
    pub fixed_case_order: Vec<String>,
    pub gpu_hours_per_evaluation: f64,
    pub gpu_hour_budget_per_target: f64,
    pub baseline: BenchmarkTarget,
    pub candidate: BenchmarkTarget,
    pub suite_manifest_sha256: BTreeMap<String, String>,
    pub ablation: Option<AblationId>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRun {
    pub metadata: BenchmarkRunMetadata,
    pub cases: Vec<RawCaseResult>,
}

impl BenchmarkRun {
    pub fn capacity_matched(&self) -> bool {
        self.metadata.baseline.parameters == self.metadata.candidate.parameters
    }

    pub fn routed_active_parameters_matched(&self) -> bool {
        self.metadata.baseline.routed_active_parameters
            == self.metadata.candidate.routed_active_parameters
    }

    pub fn training_compute_matched(&self) -> bool {
        same_f64(
            self.metadata.baseline.training_gpu_hours,
            self.metadata.candidate.training_gpu_hours,
        )
    }

    /// Convert raw metrics into the higher-is-better convention required by
    /// the paired promotion gates.  Lower-is-better metrics are negated.
    pub fn acceptance_inputs(&self) -> (AcceptanceSuite, Vec<CaseResult>) {
        let suite = AcceptanceSuite {
            version: ACCEPTANCE_SUITE_VERSION,
            cases: self
                .cases
                .iter()
                .map(|case| BenchmarkCase {
                    id: case.case_id.clone(),
                    catalog_id: case.catalog_id.clone(),
                    family: case.family.clone(),
                    visibility: case.visibility,
                    metric: case.metric.clone(),
                    stable_anchor: case.stable_anchor,
                })
                .collect(),
        };
        let results = self
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.case_id.clone(),
                pairs: case
                    .pairs
                    .iter()
                    .map(|pair| PairedRun {
                        seed: pair.seed,
                        baseline: case.direction.acceptance_score(pair.baseline.score),
                        candidate: case.direction.acceptance_score(pair.candidate.score),
                    })
                    .collect(),
            })
            .collect();
        (suite, results)
    }

    pub fn measured_gpu_hours(&self, role: TargetRole) -> f64 {
        self.cases
            .iter()
            .flat_map(|case| &case.pairs)
            .map(|pair| match role {
                TargetRole::Baseline => pair.baseline.gpu_hours,
                TargetRole::Candidate => pair.candidate.gpu_hours,
            })
            .sum()
    }

    /// Validate every relationship that is encoded inside a run artifact.
    /// Checkpoint and suite files are verified separately by their
    /// content-addressed wrappers so this remains portable after a run is
    /// copied to an evaluation vault.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.metadata.evaluator_id.trim().is_empty()
                && !self.metadata.evaluator_version.trim().is_empty(),
            "benchmark run has no evaluator identity"
        );
        ensure!(
            self.metadata.paired_seeds.len() >= MINIMUM_PAIRED_SEEDS,
            "benchmark run requires at least {MINIMUM_PAIRED_SEEDS} paired seeds"
        );
        ensure!(
            self.metadata
                .paired_seeds
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "benchmark run seeds must be unique and strictly increasing"
        );
        ensure!(
            self.metadata.gpu_hours_per_evaluation.is_finite()
                && self.metadata.gpu_hours_per_evaluation > 0.0,
            "benchmark run has an invalid GPU-hour allowance"
        );
        self.metadata.baseline.validate_identity()?;
        self.metadata.candidate.validate_identity()?;
        ensure!(
            self.metadata.baseline.id != self.metadata.candidate.id,
            "benchmark baseline and candidate identities are equal"
        );
        ensure!(!self.cases.is_empty(), "benchmark run contains no cases");
        ensure!(
            self.metadata.fixed_case_order.len() == self.cases.len(),
            "fixed case order does not cover every result"
        );
        ensure!(
            self.metadata
                .fixed_case_order
                .iter()
                .map(String::as_str)
                .eq(self.cases.iter().map(|case| case.case_id.as_str())),
            "benchmark results do not follow the recorded fixed case order"
        );
        ensure!(
            !self.metadata.suite_manifest_sha256.is_empty(),
            "benchmark run contains no suite manifest hashes"
        );
        for (suite_id, digest) in &self.metadata.suite_manifest_sha256 {
            ensure!(
                !suite_id.trim().is_empty(),
                "empty suite id in run metadata"
            );
            validate_sha256(digest)
                .with_context(|| format!("invalid manifest hash for suite `{suite_id}`"))?;
        }

        let mut case_ids = BTreeSet::new();
        for case in &self.cases {
            ensure!(
                case_ids.insert(case.case_id.as_str()),
                "duplicate benchmark result `{}`",
                case.case_id
            );
            ensure!(
                !case.case_id.trim().is_empty()
                    && !case.catalog_id.trim().is_empty()
                    && !case.metric.trim().is_empty(),
                "benchmark result contains an empty identity or metric"
            );
            ensure!(
                self.metadata
                    .suite_manifest_sha256
                    .contains_key(&case.suite_id),
                "benchmark result `{}` refers to unknown suite `{}`",
                case.case_id,
                case.suite_id
            );
            ensure!(
                case.pairs.len() == self.metadata.paired_seeds.len(),
                "benchmark `{}` does not contain every paired seed",
                case.case_id
            );
            for (ordinal, (pair, expected_seed)) in case
                .pairs
                .iter()
                .zip(&self.metadata.paired_seeds)
                .enumerate()
            {
                ensure!(
                    pair.seed == *expected_seed,
                    "benchmark `{}` seed order differs from run metadata",
                    case.case_id
                );
                ensure!(
                    pair.example_order_seed
                        == derive_order_seed(
                            self.metadata.order_seed,
                            &case.suite_id,
                            &case.case_id,
                            pair.seed,
                        ),
                    "benchmark `{}` seed {} has the wrong example order",
                    case.case_id,
                    pair.seed
                );
                pair.baseline.validate(
                    &format!("{} baseline pair {ordinal}", case.case_id),
                    self.metadata.gpu_hours_per_evaluation,
                )?;
                pair.candidate.validate(
                    &format!("{} candidate pair {ordinal}", case.case_id),
                    self.metadata.gpu_hours_per_evaluation,
                )?;
                ensure!(
                    pair.baseline.examples == pair.candidate.examples,
                    "benchmark `{}` seed {} evaluated different example counts",
                    case.case_id,
                    pair.seed
                );
            }
        }
        let expected_budget = self.metadata.gpu_hours_per_evaluation
            * self.cases.len() as f64
            * self.metadata.paired_seeds.len() as f64;
        ensure!(
            same_f64(self.metadata.gpu_hour_budget_per_target, expected_budget),
            "benchmark run GPU-hour budget does not match its schedule"
        );
        Ok(())
    }
}

/// A benchmark result whose serialized bytes and internal relationships have
/// both been verified.
#[derive(Clone, Debug)]
pub struct VerifiedBenchmarkRun {
    pub(crate) run: BenchmarkRun,
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
}

impl VerifiedBenchmarkRun {
    pub fn run(&self) -> &BenchmarkRun {
        &self.run
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn load(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        validate_sha256(expected_sha256).context("invalid expected benchmark run hash")?;
        let path = path.as_ref();
        let bytes = read_regular_file(path, "benchmark run")?;
        let actual = sha256_hex(&bytes);
        ensure!(
            actual.eq_ignore_ascii_case(expected_sha256),
            "benchmark run hash mismatch for {}: expected {}, got {}",
            path.display(),
            expected_sha256,
            actual
        );
        let run: BenchmarkRun = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid benchmark run {}", path.display()))?;
        run.validate()?;
        Ok(Self {
            run,
            path: path.to_path_buf(),
            sha256: actual,
        })
    }

    pub fn verify_contents(&self) -> Result<()> {
        let bytes = read_regular_file(&self.path, "benchmark run")?;
        ensure!(
            sha256_hex(&bytes) == self.sha256,
            "benchmark run changed after verification: {}",
            self.path.display()
        );
        let disk_run: BenchmarkRun = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid benchmark run {}", self.path.display()))?;
        disk_run.validate()?;
        ensure!(
            serde_json::to_vec(&disk_run)? == serde_json::to_vec(&self.run)?,
            "in-memory benchmark run differs from its content-addressed artifact"
        );
        self.run.metadata.baseline.verify()?;
        self.run.metadata.candidate.verify()?;
        Ok(())
    }
}

/// Content-addressed performance, memory, kernel, and resume measurements.
#[derive(Clone, Debug)]
pub struct VerifiedResourceComparison {
    pub(crate) comparison: ResourceComparison,
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
    exact_resume_verified: bool,
}

impl VerifiedResourceComparison {
    pub fn comparison(&self) -> &ResourceComparison {
        &self.comparison
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn load(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        validate_sha256(expected_sha256).context("invalid expected resource evidence hash")?;
        let path = path.as_ref();
        let bytes = read_regular_file(path, "resource evidence")?;
        let actual = sha256_hex(&bytes);
        ensure!(
            actual.eq_ignore_ascii_case(expected_sha256),
            "resource evidence hash mismatch for {}: expected {}, got {}",
            path.display(),
            expected_sha256,
            actual
        );
        let comparison: ResourceComparison = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid resource evidence {}", path.display()))?;
        comparison.validate()?;
        verify_resource_comparison_artifacts(&comparison, path)?;
        Ok(Self {
            comparison,
            path: path.to_path_buf(),
            sha256: actual,
            exact_resume_verified: true,
        })
    }

    pub fn verify_contents(&self) -> Result<()> {
        let bytes = read_regular_file(&self.path, "resource evidence")?;
        ensure!(
            sha256_hex(&bytes) == self.sha256,
            "resource evidence changed after verification: {}",
            self.path.display()
        );
        let disk_comparison: ResourceComparison = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid resource evidence {}", self.path.display()))?;
        disk_comparison.validate()?;
        ensure!(
            serde_json::to_vec(&disk_comparison)? == serde_json::to_vec(&self.comparison)?,
            "in-memory resource evidence differs from its content-addressed artifact"
        );
        ensure!(
            self.exact_resume_verified,
            "exact-resume evidence was not verified"
        );
        verify_resource_comparison_artifacts(&self.comparison, &self.path)?;
        Ok(())
    }
}

pub(crate) fn verify_resource_comparison_artifacts(
    comparison: &ResourceComparison,
    source_path: &Path,
) -> Result<()> {
    comparison.validate()?;
    let mut resolved = comparison.clone();
    resolve_exact_resume_paths(&mut resolved.exact_resume, source_path);
    verify_exact_resume_artifacts(&resolved.exact_resume)
}

fn resolve_exact_resume_paths(evidence: &mut ExactResumeEvidence, source: &Path) {
    let base = source.parent().unwrap_or_else(|| Path::new("."));
    for artifact in [
        &mut evidence.interrupted_checkpoint,
        &mut evidence.uninterrupted_final_state,
        &mut evidence.resumed_final_state,
        &mut evidence.uninterrupted_metrics,
        &mut evidence.resumed_metrics,
    ] {
        if artifact.path.is_relative() {
            artifact.path = base.join(&artifact.path);
        }
    }
}

fn read_exact_artifact(artifact: &ExactResumeArtifact, label: &str) -> Result<Vec<u8>> {
    let bytes = read_regular_file(&artifact.path, label)?;
    let actual = sha256_hex(&bytes);
    ensure!(
        actual.eq_ignore_ascii_case(&artifact.sha256),
        "{label} hash mismatch: expected {}, got {actual}",
        artifact.sha256
    );
    Ok(bytes)
}

fn verify_checkpoint_artifact(
    artifact: &ExactResumeArtifact,
    label: &str,
) -> Result<CheckpointManifest> {
    ensure!(
        artifact.path.file_name().and_then(|name| name.to_str())
            == Some("generation-manifest.json"),
        "{label} must address generation-manifest.json"
    );
    let bytes = read_exact_artifact(artifact, label)?;
    let manifest: CheckpointManifest =
        serde_json::from_slice(&bytes).with_context(|| format!("invalid {label}"))?;
    validate_checkpoint_manifest(&manifest, label)?;
    let generation = artifact
        .path
        .parent()
        .with_context(|| format!("{label} has no generation directory"))?;
    ensure!(
        generation.file_name().and_then(|name| name.to_str())
            == Some(&format!("sha256-{}", artifact.sha256)),
        "{label} generation directory does not match its manifest hash"
    );
    for file in &manifest.files {
        read_manifest_file(generation, file, &format!("{label} file `{}`", file.path))?;
    }
    Ok(manifest)
}

fn verify_exact_resume_artifacts(evidence: &ExactResumeEvidence) -> Result<()> {
    ensure!(
        evidence.interruption_step == evidence.resumed_from_step,
        "resumed step does not equal the interrupted checkpoint step"
    );
    ensure!(
        evidence.uninterrupted_final_state.path != evidence.resumed_final_state.path
            && evidence.uninterrupted_metrics.path != evidence.resumed_metrics.path,
        "exact-resume comparison requires distinct uninterrupted and resumed artifacts"
    );
    let interrupted =
        verify_checkpoint_artifact(&evidence.interrupted_checkpoint, "interrupted checkpoint")?;
    let uninterrupted = verify_checkpoint_artifact(
        &evidence.uninterrupted_final_state,
        "uninterrupted final state",
    )?;
    let resumed = verify_checkpoint_artifact(&evidence.resumed_final_state, "resumed final state")?;
    ensure!(
        interrupted.global_step as u64 == evidence.interruption_step,
        "interrupted checkpoint global step does not match resume evidence"
    );
    ensure!(
        uninterrupted.global_step == resumed.global_step
            && uninterrupted.global_step > interrupted.global_step,
        "exact-resume final checkpoints do not describe the same later step"
    );
    ensure!(
        evidence
            .uninterrupted_final_state
            .sha256
            .eq_ignore_ascii_case(&evidence.resumed_final_state.sha256),
        "interrupted and uninterrupted runs did not reach byte-identical final state"
    );

    let uninterrupted_metrics = read_exact_artifact(
        &evidence.uninterrupted_metrics,
        "uninterrupted metric journal",
    )?;
    let resumed_metrics = read_exact_artifact(&evidence.resumed_metrics, "resumed metric journal")?;
    let uninterrupted_digest = metric_log_digests_from_bytes(&uninterrupted_metrics, None)?;
    let resumed_digest = metric_log_digests_from_bytes(&resumed_metrics, None)?;
    ensure!(
        uninterrupted_digest.last_global_step == Some(uninterrupted.global_step as u64)
            && resumed_digest.last_global_step == Some(resumed.global_step as u64),
        "exact-resume metric journal does not end at its final checkpoint"
    );
    ensure!(
        uninterrupted_digest.records > 0 && resumed_digest.records > 0,
        "exact-resume metric journals are empty"
    );
    ensure!(
        uninterrupted_digest.semantic_progress_sha256 == resumed_digest.semantic_progress_sha256,
        "interrupted and uninterrupted metric journals differ semantically"
    );
    Ok(())
}

/// Evaluate a promotion exclusively from immutable benchmark and resource
/// artifacts.  `comparison_runs` is the fixed sleep ablation matrix; it is
/// used to derive (rather than assert) which capacity- and compute-matched
/// baseline is strongest.
pub fn evaluate_verified_promotion(
    selected_run: &VerifiedBenchmarkRun,
    comparison_runs: &[VerifiedBenchmarkRun],
    resources: &VerifiedResourceComparison,
    policy: &AcceptancePolicy,
    policy_sha256: &str,
) -> Result<PromotionReport> {
    resources.verify_contents()?;
    let strongest_baseline_id = verified_resource_benchmark_context(selected_run, comparison_runs)?;

    let comparison = &resources.comparison;
    ensure!(
        comparison
            .benchmark_run_sha256
            .eq_ignore_ascii_case(&selected_run.sha256),
        "resource evidence addresses a different benchmark run"
    );
    ensure!(
        comparison.baseline_id == selected_run.run.metadata.baseline.id
            && comparison.candidate_id == selected_run.run.metadata.candidate.id,
        "resource evidence target identities do not match the selected benchmark run"
    );
    ensure!(
        comparison.strongest_baseline_id == strongest_baseline_id,
        "resource evidence names `{}` as strongest, but verified runs derive `{}`",
        comparison.strongest_baseline_id,
        strongest_baseline_id
    );
    ensure!(
        comparison
            .exact_resume
            .uninterrupted_final_state
            .sha256
            .eq_ignore_ascii_case(
                &selected_run
                    .run
                    .metadata
                    .candidate
                    .checkpoint_manifest_sha256,
            ),
        "exact-resume evidence does not terminate at the candidate checkpoint"
    );
    crate::resource_worker::validate_execution_receipt(
        selected_run,
        comparison_runs,
        policy,
        policy_sha256,
        comparison,
        resources.path(),
    )?;
    let (suite, results) = selected_run.run.acceptance_inputs();
    let context = VerifiedPromotionContext {
        benchmark_run_sha256: &selected_run.sha256,
        strongest_baseline_id: &strongest_baseline_id,
        baseline_id: &selected_run.run.metadata.baseline.id,
        candidate_id: &selected_run.run.metadata.candidate.id,
        capacity_matched: selected_run.run.capacity_matched(),
        active_capacity_matched: selected_run.run.routed_active_parameters_matched(),
        fixed_gpu_hour_measured: selected_run.run.training_compute_matched(),
        baseline_stored_parameters: selected_run.run.metadata.baseline.parameters,
        candidate_stored_parameters: selected_run.run.metadata.candidate.parameters,
        baseline_stored_bytes: selected_run.run.metadata.baseline.stored_bytes,
        candidate_stored_bytes: selected_run.run.metadata.candidate.stored_bytes,
        candidate_routed_active_parameters: selected_run
            .run
            .metadata
            .candidate
            .routed_active_parameters,
        content_addressed_evidence: true,
        executed_resource_evidence: true,
        exact_resume_verified: resources.exact_resume_verified,
    };
    evaluate_with_verified_context(&suite, &results, comparison, policy, &context)
}

pub(crate) fn verified_resource_benchmark_context(
    selected_run: &VerifiedBenchmarkRun,
    comparison_runs: &[VerifiedBenchmarkRun],
) -> Result<String> {
    selected_run.verify_contents()?;
    selected_run.run.validate()?;
    validate_run_catalog_coverage(&selected_run.run, false)?;
    let selected_occurrences = comparison_runs
        .iter()
        .filter(|run| run.sha256 == selected_run.sha256)
        .count();
    ensure!(
        selected_occurrences == 1,
        "the selected benchmark run must occur exactly once in the comparison set"
    );
    validate_comparison_set(comparison_runs)?;
    derive_strongest_baseline(comparison_runs)
}

fn validate_run_catalog_coverage(run: &BenchmarkRun, include_later_sweeps: bool) -> Result<()> {
    let mut covered = BTreeMap::new();
    for case in &run.cases {
        let key = (case.catalog_id.as_str(), case.visibility);
        ensure!(
            covered.insert(key, &case.family).is_none(),
            "benchmark run repeats {:?} catalog entry `{}`",
            case.visibility,
            case.catalog_id
        );
    }
    for entry in required_catalog() {
        if !entry.stage.required_now() && !include_later_sweeps {
            continue;
        }
        for visibility in [SuiteVisibility::Public, SuiteVisibility::Sealed] {
            let family = covered.get(&(entry.id, visibility)).with_context(|| {
                format!(
                    "benchmark run is missing {:?} catalog entry `{}`",
                    visibility, entry.id
                )
            })?;
            ensure!(
                **family == entry.family,
                "benchmark run catalog `{}` has {:?} family {:?}, expected {:?}",
                entry.id,
                visibility,
                family,
                entry.family
            );
        }
    }
    Ok(())
}

fn validate_comparison_set(runs: &[VerifiedBenchmarkRun]) -> Result<()> {
    ensure!(
        runs.len() == AblationId::ALL.len(),
        "promotion comparison set must contain all {} fixed ablations",
        AblationId::ALL.len()
    );
    let reference = runs.first().context("promotion comparison set is empty")?;
    let mut ablations = BTreeSet::new();
    let mut baseline_ids = BTreeSet::new();
    for run in runs {
        run.verify_contents()?;
        run.run.validate()?;
        validate_run_catalog_coverage(&run.run, false)?;
        let ablation = run
            .run
            .metadata
            .ablation
            .context("every promotion comparison run must name its ablation")?;
        ensure!(
            ablations.insert(ablation),
            "comparison set repeats ablation `{}`",
            ablation.as_str()
        );
        ensure!(
            baseline_ids.insert(run.run.metadata.baseline.id.as_str()),
            "comparison set repeats baseline `{}`",
            run.run.metadata.baseline.id
        );
        ensure!(
            run.run.capacity_matched()
                && run.run.routed_active_parameters_matched()
                && run.run.training_compute_matched(),
            "baseline `{}` is not capacity-, routed-active-, and training-compute-matched",
            run.run.metadata.baseline.id
        );
        validate_comparison_compatibility(&reference.run, &run.run)?;
    }
    ensure!(
        AblationId::ALL.into_iter().collect::<BTreeSet<_>>() == ablations,
        "promotion comparison set does not cover the fixed ablation catalog"
    );
    Ok(())
}

fn validate_comparison_compatibility(reference: &BenchmarkRun, run: &BenchmarkRun) -> Result<()> {
    let left = &reference.metadata;
    let right = &run.metadata;
    ensure!(
        left.evaluator_id == right.evaluator_id
            && left.evaluator_version == right.evaluator_version
            && left.paired_seeds == right.paired_seeds
            && left.order_seed == right.order_seed
            && left.fixed_case_order == right.fixed_case_order
            && same_f64(
                left.gpu_hours_per_evaluation,
                right.gpu_hours_per_evaluation
            )
            && same_f64(
                left.gpu_hour_budget_per_target,
                right.gpu_hour_budget_per_target
            )
            && left.suite_manifest_sha256 == right.suite_manifest_sha256,
        "comparison baseline `{}` was not evaluated with the fixed evaluator, order, suites, seeds, and compute budget",
        right.baseline.id
    );
    ensure!(
        same_target_identity(&left.candidate, &right.candidate),
        "comparison run for baseline `{}` evaluates a different candidate",
        right.baseline.id
    );
    ensure!(
        reference.cases.len() == run.cases.len(),
        "comparison run case counts differ"
    );
    for (expected, actual) in reference.cases.iter().zip(&run.cases) {
        ensure!(
            expected.suite_id == actual.suite_id
                && expected.case_id == actual.case_id
                && expected.catalog_id == actual.catalog_id
                && expected.visibility == actual.visibility
                && expected.family == actual.family
                && expected.metric == actual.metric
                && expected.direction == actual.direction
                && expected.stable_anchor == actual.stable_anchor,
            "comparison run benchmark definitions differ at `{}`",
            expected.case_id
        );
        for (expected_pair, actual_pair) in expected.pairs.iter().zip(&actual.pairs) {
            ensure!(
                expected_pair.seed == actual_pair.seed
                    && expected_pair.example_order_seed == actual_pair.example_order_seed
                    && expected_pair.candidate.score.to_bits()
                        == actual_pair.candidate.score.to_bits()
                    && expected_pair.candidate.examples == actual_pair.candidate.examples,
                "candidate measurement is not deterministic for `{}` seed {}",
                expected.case_id,
                expected_pair.seed
            );
        }
    }
    Ok(())
}

fn same_target_identity(left: &BenchmarkTarget, right: &BenchmarkTarget) -> bool {
    left.id == right.id
        && left.checkpoint_manifest_sha256 == right.checkpoint_manifest_sha256
        && left.training_evidence_sha256 == right.training_evidence_sha256
        && same_f64(left.training_gpu_hours, right.training_gpu_hours)
        && left.parameters == right.parameters
        && left.routed_active_parameters == right.routed_active_parameters
        && left.stored_bytes == right.stored_bytes
        && left.representation_identity() == right.representation_identity()
}

/// Borda aggregation over every case/seed observation avoids mixing metric
/// scales.  Direction-adjusted scores determine ranks; baseline id provides a
/// deterministic tie-break for byte-identical measurements.
fn derive_strongest_baseline(runs: &[VerifiedBenchmarkRun]) -> Result<String> {
    let mut points = BTreeMap::<&str, u128>::new();
    for run in runs {
        points.insert(run.run.metadata.baseline.id.as_str(), 0);
    }
    let reference = &runs[0].run;
    for case_index in 0..reference.cases.len() {
        for pair_index in 0..reference.cases[case_index].pairs.len() {
            let mut ranked = runs
                .iter()
                .map(|run| {
                    let case = &run.run.cases[case_index];
                    (
                        case.direction
                            .acceptance_score(case.pairs[pair_index].baseline.score),
                        run.run.metadata.baseline.id.as_str(),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| {
                right.0.total_cmp(&left.0).then_with(|| left.1.cmp(right.1))
            });
            let count = ranked.len() as u128;
            for (rank, (_, id)) in ranked.into_iter().enumerate() {
                *points
                    .get_mut(id)
                    .expect("all baseline ids were registered") += count - rank as u128;
            }
        }
    }
    points
        .into_iter()
        .max_by(|(left_id, left_points), (right_id, right_points)| {
            left_points
                .cmp(right_points)
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(id, _)| id.to_owned())
        .context("promotion comparison set is empty")
}

#[derive(Clone, Debug)]
pub struct BenchmarkRunner {
    pub config: BenchmarkRunConfig,
}

impl BenchmarkRunner {
    pub fn run<E: BenchmarkEvaluator>(
        &self,
        suites: &[VerifiedBenchmarkSuite],
        baseline: &BenchmarkTarget,
        candidate: &BenchmarkTarget,
        evaluator: &mut E,
    ) -> Result<BenchmarkRun> {
        self.config.validate()?;
        ensure!(!suites.is_empty(), "no benchmark suites were supplied");
        let baseline_model = baseline.verify()?;
        let candidate_model = candidate.verify()?;

        // Suite ids are sorted, while each suite's manifest order is retained.
        // Thus CLI argument order cannot silently alter an experiment.
        let mut suite_order = (0..suites.len()).collect::<Vec<_>>();
        suite_order.sort_by(|left, right| {
            suites[*left]
                .manifest
                .suite_id
                .cmp(&suites[*right].manifest.suite_id)
        });
        ensure!(
            suite_order.windows(2).all(|pair| {
                suites[pair[0]].manifest.suite_id != suites[pair[1]].manifest.suite_id
            }),
            "benchmark suite ids must be globally unique"
        );

        let mut global_ids = BTreeSet::new();
        let mut schedule = Vec::new();
        let mut manifest_hashes = BTreeMap::new();
        for suite_index in suite_order {
            let suite = &suites[suite_index];
            suite.manifest.validate()?;
            suite.verify_contents()?;
            manifest_hashes.insert(
                suite.manifest.suite_id.clone(),
                suite.manifest_sha256.clone(),
            );
            for case_index in 0..suite.manifest.cases.len() {
                let case = &suite.manifest.cases[case_index];
                ensure!(
                    global_ids.insert(case.id.as_str()),
                    "benchmark case id `{}` occurs in multiple suites",
                    case.id
                );
                schedule.push((suite, case_index));
            }
        }

        let fixed_case_order = schedule
            .iter()
            .map(|(suite, case_index)| suite.manifest.cases[*case_index].id.clone())
            .collect::<Vec<_>>();

        let mut raw_cases = Vec::with_capacity(schedule.len());
        let mut pair_ordinal = 0usize;
        for (suite, case_index) in schedule {
            let case = &suite.manifest.cases[case_index];
            let artifact = suite.artifact(&case.id).with_context(|| {
                format!("verified artifact for benchmark `{}` disappeared", case.id)
            })?;
            let mut pairs = Vec::with_capacity(self.config.paired_seeds.len());
            for &seed in &self.config.paired_seeds {
                let example_order_seed = derive_order_seed(
                    self.config.order_seed,
                    &suite.manifest.suite_id,
                    &case.id,
                    seed,
                );
                let baseline_request = EvaluationRequest {
                    suite_id: &suite.manifest.suite_id,
                    visibility: suite.manifest.visibility,
                    case,
                    artifact,
                    target: baseline,
                    model: &baseline_model,
                    role: TargetRole::Baseline,
                    model_seed: seed,
                    example_order_seed,
                    pair_ordinal,
                    max_gpu_hours: self.config.gpu_hours_per_evaluation,
                };
                let baseline_measurement =
                    evaluator.evaluate(&baseline_request).with_context(|| {
                        format!("baseline evaluation failed for `{}` seed {seed}", case.id)
                    })?;
                baseline_measurement.validate(&case.id, self.config.gpu_hours_per_evaluation)?;

                let candidate_request = EvaluationRequest {
                    target: candidate,
                    model: &candidate_model,
                    role: TargetRole::Candidate,
                    ..baseline_request
                };
                let candidate_measurement =
                    evaluator.evaluate(&candidate_request).with_context(|| {
                        format!("candidate evaluation failed for `{}` seed {seed}", case.id)
                    })?;
                candidate_measurement.validate(&case.id, self.config.gpu_hours_per_evaluation)?;

                pairs.push(RawPairedResult {
                    seed,
                    example_order_seed,
                    baseline: baseline_measurement,
                    candidate: candidate_measurement,
                });
                pair_ordinal += 1;
            }
            raw_cases.push(RawCaseResult {
                suite_id: suite.manifest.suite_id.clone(),
                case_id: case.id.clone(),
                catalog_id: case.catalog_id().to_owned(),
                visibility: suite.manifest.visibility,
                family: case.family.clone(),
                metric: case.metric.clone(),
                direction: case.direction,
                stable_anchor: case.stable_anchor,
                pairs,
            });
        }

        let evaluation_count = raw_cases.len() * self.config.paired_seeds.len();
        let run = BenchmarkRun {
            metadata: BenchmarkRunMetadata {
                evaluator_id: self.config.evaluator_id.clone(),
                evaluator_version: self.config.evaluator_version.clone(),
                paired_seeds: self.config.paired_seeds.clone(),
                order_seed: self.config.order_seed,
                fixed_case_order,
                gpu_hours_per_evaluation: self.config.gpu_hours_per_evaluation,
                gpu_hour_budget_per_target: self.config.gpu_hours_per_evaluation
                    * evaluation_count as f64,
                baseline: baseline.clone(),
                candidate: candidate.clone(),
                suite_manifest_sha256: manifest_hashes,
                ablation: self.config.ablation,
            },
            cases: raw_cases,
        };
        run.validate()?;
        Ok(run)
    }
}

fn derive_order_seed(base: u64, suite_id: &str, case_id: &str, model_seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(base.to_le_bytes());
    hasher.update((suite_id.len() as u64).to_le_bytes());
    hasher.update(suite_id.as_bytes());
    hasher.update((case_id.len() as u64).to_le_bytes());
    hasher.update(case_id.as_bytes());
    hasher.update(model_seed.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

fn same_f64(left: f64, right: f64) -> bool {
    (left - right).abs() <= f64::EPSILON * left.abs().max(right.abs()).max(1.0) * 8.0
}

fn validate_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "SHA-256 must contain exactly 64 hexadecimal characters"
    );
    Ok(())
}

fn validate_prefixed_sha256(value: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .context("SHA-256 identity must use sha256:<64 lowercase hex>")?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "SHA-256 identity must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEvaluator {
        calls: Vec<(String, u64, u64, TargetRole)>,
    }

    impl BenchmarkEvaluator for MockEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            self.calls.push((
                request.case.id.clone(),
                request.model_seed,
                request.example_order_seed,
                request.role,
            ));
            let candidate = request.role == TargetRole::Candidate;
            let score = match request.case.direction {
                MetricDirection::Maximize => 0.5 + if candidate { 0.1 } else { 0.0 },
                MetricDirection::Minimize => 1.0 - if candidate { 0.1 } else { 0.0 },
            };
            Ok(EvaluationMeasurement {
                score,
                gpu_hours: request.max_gpu_hours / 2.0,
                examples: 8,
                metrics: BTreeMap::new(),
            })
        }
    }

    fn write_suite(
        root: &Path,
        suite_id: &str,
        visibility: SuiteVisibility,
        case_id: &str,
        direction: MetricDirection,
    ) -> VerifiedBenchmarkSuite {
        let artifact_path = root.join(format!("{case_id}.jsonl"));
        fs::write(&artifact_path, b"{\"input\":\"fixture\"}\n").unwrap();
        let artifact_hash = sha256_hex(&fs::read(&artifact_path).unwrap());
        let manifest = BenchmarkSuiteManifest {
            version: BENCHMARK_MANIFEST_VERSION,
            suite_id: suite_id.into(),
            visibility,
            cases: vec![BenchmarkSpec {
                id: case_id.into(),
                catalog_id: None,
                family: BenchmarkFamily::Reasoning,
                metric: "score".into(),
                direction,
                stable_anchor: false,
                artifact: BenchmarkArtifact {
                    path: artifact_path.file_name().unwrap().into(),
                    sha256: artifact_hash,
                },
                evaluator: BTreeMap::new(),
            }],
        };
        let manifest_path = root.join(format!("{suite_id}.json"));
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&manifest_path, &bytes).unwrap();
        VerifiedBenchmarkSuite::load(&manifest_path, &sha256_hex(&bytes)).unwrap()
    }

    fn target(root: &Path, id: &str, parameters: u64) -> BenchmarkTarget {
        use safetensors::{Dtype, tensor::TensorView};

        let output = root.join(id);
        let staging = output.join("generations").join("staging");
        fs::create_dir_all(&staging).unwrap();
        let salt = id.bytes().fold(0_u64, |sum, byte| sum + u64::from(byte)) as f32;
        let raw_weights = (0..parameters)
            .map(|index| ((index as f32 + salt) * 0.03125).sin())
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![1, parameters as usize], &raw_weights).unwrap();
        let weights = safetensors::tensor::serialize([("weight", view)], None).unwrap();
        let weights_sha256 = sha256_hex(&weights);
        fs::write(staging.join(CHECKPOINT_WEIGHTS_FILE), &weights).unwrap();
        let accounting = TrainingAccounting {
            version: TRAINING_ACCOUNTING_VERSION,
            training_gpu_hours: 8.0,
            parameters,
            routed_active_parameters: 80,
            weights_bytes: weights.len() as u64,
            weights_sha256: weights_sha256.clone(),
        };
        let accounting_bytes = serde_json::to_vec(&accounting).unwrap();
        let accounting_sha256 = sha256_hex(&accounting_bytes);
        fs::write(staging.join(TRAINING_ACCOUNTING_FILE), &accounting_bytes).unwrap();
        let manifest = CheckpointManifest {
            version: 1,
            training_state_version: 2,
            global_step: 20,
            _phase: 0,
            phase_id: "test".into(),
            files: vec![
                CheckpointManifestFile {
                    path: TRAINING_ACCOUNTING_FILE.into(),
                    bytes: accounting_bytes.len() as u64,
                    sha256: accounting_sha256.clone(),
                },
                CheckpointManifestFile {
                    path: CHECKPOINT_WEIGHTS_FILE.into(),
                    bytes: weights.len() as u64,
                    sha256: weights_sha256.clone(),
                },
            ],
        };
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let checkpoint_manifest_sha256 = sha256_hex(&manifest_bytes);
        let generation = output
            .join("generations")
            .join(format!("sha256-{checkpoint_manifest_sha256}"));
        fs::rename(&staging, &generation).unwrap();
        let path = generation.join("generation-manifest.json");
        fs::write(&path, &manifest_bytes).unwrap();
        let evidence = TrainingEvidence {
            version: 1,
            checkpoint_manifest_sha256: checkpoint_manifest_sha256.clone(),
            accounting_sha256,
            training_gpu_hours: 8.0,
            parameters,
            routed_active_parameters: 80,
            stored_bytes: weights.len() as u64,
            weights_sha256,
        };
        let evidence_bytes = serde_json::to_vec(&evidence).unwrap();
        let evidence_sha256 = sha256_hex(&evidence_bytes);
        let evidence_path = output
            .join("training-evidence")
            .join(format!("sha256-{evidence_sha256}.json"));
        fs::create_dir_all(evidence_path.parent().unwrap()).unwrap();
        fs::write(&evidence_path, &evidence_bytes).unwrap();
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest_sha256,
            training_evidence: evidence_path,
            training_evidence_sha256: evidence_sha256,
            training_gpu_hours: 8.0,
            checkpoint_manifest: path,
            parameters,
            routed_active_parameters: 80,
            stored_bytes: weights.len() as u64,
            representation: ModelRepresentationTarget::FullPrecision,
        }
    }

    fn attach_hquant_candidate(root: &Path, mut target: BenchmarkTarget) -> BenchmarkTarget {
        use crate::qat_candidate::{QatArchiveMetrics, QatCandidateManifest, open_qat_candidate};
        use crate::quantization::{
            BONSAI_GROUP_SIZE, QuantizationRecipe, QuantizedArchive, UltraQuantFormat,
            export_safetensors_archive,
        };

        let source = target
            .checkpoint_manifest
            .parent()
            .unwrap()
            .join(CHECKPOINT_WEIGHTS_FILE);
        let candidate_key = format!("{}-hquant", target.id);
        let candidate_root = root.join(&candidate_key);
        fs::create_dir(&candidate_root).unwrap();
        let candidate_weights = candidate_root.join("weights.safetensors");
        fs::copy(&source, &candidate_weights).unwrap();
        let recipe = QuantizationRecipe {
            format: UltraQuantFormat::BinaryG128,
            group_size: BONSAI_GROUP_SIZE,
            fake_quant_start_step: 0,
            ternary_warmup_steps: 0,
            distillation_weight: 0.0,
            quantize_embeddings: true,
            quantize_lm_head: true,
        };
        let archive_path = candidate_root.join("hquant");
        let archive_manifest =
            export_safetensors_archive(&candidate_weights, &archive_path, &recipe).unwrap();
        let quantized_elements = archive_manifest
            .matrices
            .iter()
            .map(|matrix| matrix.elements)
            .sum::<u64>();
        let packed_bytes = archive_manifest
            .matrices
            .iter()
            .map(|matrix| matrix.packed_bytes)
            .sum::<u64>();
        let floating_elements = archive_manifest
            .floating_tensors
            .iter()
            .map(|tensor| tensor.elements)
            .sum::<u64>();
        let floating_bytes = archive_manifest
            .floating_tensors
            .iter()
            .map(|tensor| tensor.bytes)
            .sum::<u64>();
        let weighted_mean_squared_error = archive_manifest
            .matrices
            .iter()
            .map(|matrix| matrix.mean_squared_error * matrix.elements as f64)
            .sum::<f64>()
            / quantized_elements as f64;
        let maximum_absolute_error = archive_manifest
            .matrices
            .iter()
            .map(|matrix| f64::from(matrix.maximum_absolute_error))
            .fold(0.0_f64, f64::max);
        let metrics = QatArchiveMetrics {
            quantized_tensors: archive_manifest.matrices.len() as u64,
            quantized_elements,
            packed_bytes,
            floating_tensors: archive_manifest.floating_tensors.len() as u64,
            floating_elements,
            floating_bytes,
            archive_weight_elements: quantized_elements + floating_elements,
            archive_weight_bytes: packed_bytes + floating_bytes,
            average_bits_per_weight: archive_manifest.true_average_bits_per_weight().unwrap(),
            weighted_mean_squared_error,
            maximum_absolute_error,
        };
        let archive = QuantizedArchive::open(&archive_path).unwrap();
        let archive_manifest_path = archive_path.join("manifest.json");
        let manifest = QatCandidateManifest {
            version: 1,
            candidate_key,
            weights_file: "weights.safetensors".into(),
            weights_bytes: fs::metadata(&candidate_weights).unwrap().len(),
            weights_sha256: format!(
                "sha256:{}",
                sha256_hex(&fs::read(&candidate_weights).unwrap())
            ),
            archive_directory: "hquant".into(),
            archive_manifest: "hquant/manifest.json".into(),
            archive_manifest_sha256: format!(
                "sha256:{}",
                sha256_hex(&fs::read(&archive_manifest_path).unwrap())
            ),
            archive_content_sha256: archive.content_hash().unwrap(),
            recipe,
            metrics: metrics.clone(),
        };
        let candidate_manifest = candidate_root.join("candidate.json");
        let candidate_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&candidate_manifest, &candidate_bytes).unwrap();
        let publication = open_qat_candidate(&candidate_root).unwrap();
        target.stored_bytes = metrics.archive_weight_bytes;
        target.representation = ModelRepresentationTarget::Hquant {
            candidate_manifest,
            candidate_manifest_sha256: publication.candidate_manifest_sha256,
        };
        target
    }

    fn write_catalog_suite(
        root: &Path,
        suite_id: &str,
        visibility: SuiteVisibility,
    ) -> VerifiedBenchmarkSuite {
        let prefix = match visibility {
            SuiteVisibility::Public => "public",
            SuiteVisibility::Sealed => "sealed",
        };
        let stable_anchors = AcceptancePolicy::default().stable_anchor_catalog_ids;
        let cases = required_catalog()
            .into_iter()
            .filter(|entry| entry.stage.required_now())
            .map(|entry| {
                let case_id = format!("{prefix}-{}", entry.id);
                let artifact_path = root.join(format!("{case_id}.jsonl"));
                fs::write(&artifact_path, format!("{{\"id\":\"{case_id}\"}}\n")).unwrap();
                BenchmarkSpec {
                    id: case_id,
                    catalog_id: Some(entry.id.into()),
                    family: entry.family,
                    metric: "score".into(),
                    direction: MetricDirection::Maximize,
                    stable_anchor: stable_anchors.contains(entry.id),
                    artifact: BenchmarkArtifact {
                        path: artifact_path.file_name().unwrap().into(),
                        sha256: sha256_hex(&fs::read(&artifact_path).unwrap()),
                    },
                    evaluator: BTreeMap::new(),
                }
            })
            .collect();
        let manifest = BenchmarkSuiteManifest {
            version: BENCHMARK_MANIFEST_VERSION,
            suite_id: suite_id.into(),
            visibility,
            cases,
        };
        let path = root.join(format!("{suite_id}.json"));
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&path, &bytes).unwrap();
        VerifiedBenchmarkSuite::load(path, &sha256_hex(&bytes)).unwrap()
    }

    struct RankingEvaluator;

    impl BenchmarkEvaluator for RankingEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            let score = match request.role {
                TargetRole::Candidate => 0.9,
                TargetRole::Baseline => {
                    let rank = request
                        .target
                        .id
                        .strip_prefix("baseline-")
                        .context("unexpected baseline fixture id")?
                        .parse::<u64>()?;
                    0.4 + rank as f64 / 100.0
                }
            };
            Ok(EvaluationMeasurement {
                score,
                gpu_hours: request.max_gpu_hours,
                examples: 128,
                metrics: BTreeMap::new(),
            })
        }
    }

    fn write_run(root: &Path, index: usize, run: &BenchmarkRun) -> VerifiedBenchmarkRun {
        let path = root.join(format!("run-{index:02}.json"));
        let bytes = serde_json::to_vec_pretty(run).unwrap();
        fs::write(&path, &bytes).unwrap();
        VerifiedBenchmarkRun::load(path, &sha256_hex(&bytes)).unwrap()
    }

    fn resource_comparison(run: &VerifiedBenchmarkRun) -> ResourceComparison {
        use crate::metrics::{
            MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, MetricWriter,
            ThroughputMetrics,
        };

        let final_manifest = &run.run.metadata.candidate.checkpoint_manifest;
        let final_generation = final_manifest.parent().unwrap();
        let evidence_vault = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let approved_root = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .strip_prefix(evidence_vault)
            .unwrap()
            .to_path_buf();
        let exact_root = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("exact-resume");
        let resumed_generation = exact_root
            .join("resumed")
            .join("generations")
            .join(final_generation.file_name().unwrap());
        fs::create_dir_all(&resumed_generation).unwrap();
        for entry in fs::read_dir(final_generation).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), resumed_generation.join(entry.file_name())).unwrap();
        }
        let resumed_manifest = resumed_generation.join("generation-manifest.json");

        let final_bytes = fs::read(final_manifest).unwrap();
        let mut interrupted: CheckpointManifest = serde_json::from_slice(&final_bytes).unwrap();
        interrupted.global_step = 10;
        let interrupted_bytes = serde_json::to_vec(&interrupted).unwrap();
        let interrupted_sha256 = sha256_hex(&interrupted_bytes);
        let interrupted_generation = exact_root
            .join("interrupted")
            .join("generations")
            .join(format!("sha256-{interrupted_sha256}"));
        fs::create_dir_all(&interrupted_generation).unwrap();
        for file in &interrupted.files {
            fs::copy(
                final_generation.join(&file.path),
                interrupted_generation.join(&file.path),
            )
            .unwrap();
        }
        let interrupted_manifest = interrupted_generation.join("generation-manifest.json");
        fs::write(&interrupted_manifest, &interrupted_bytes).unwrap();

        let write_metrics = |directory: &Path, run_id: &str, elapsed: f64, timestamp: u64| {
            fs::create_dir_all(directory).unwrap();
            let path = directory.join("metrics.jsonl");
            let mut writer = MetricWriter::create(&path, run_id).unwrap();
            writer
                .append_at(
                    MetricContext {
                        global_step: 20,
                        phase: MetricPhase {
                            index: 0,
                            name: "test".into(),
                            kind: MetricPhaseKind::Pretrain,
                        },
                        checkpoint_hash: None,
                    },
                    MetricEvent::Throughput(ThroughputMetrics {
                        optimizer_steps: 20,
                        compute_tokens: 200,
                        supervised_tokens: 200,
                        examples: 20,
                        elapsed_seconds: elapsed,
                        tokens_per_second: 200.0 / elapsed,
                        examples_per_second: 20.0 / elapsed,
                        input_wait_seconds: 0.0,
                        host_to_device_seconds: 0.0,
                        gpu_busy_seconds: elapsed,
                    }),
                    timestamp,
                )
                .unwrap();
            writer.sync_all().unwrap();
            drop(writer);
            path
        };
        let uninterrupted_metrics =
            write_metrics(&exact_root.join("uninterrupted"), "uninterrupted", 2.0, 100);
        let resumed_metrics = write_metrics(&exact_root.join("resumed"), "resumed", 3.0, 200);

        let relative = |path: &Path| path.strip_prefix(evidence_vault).unwrap().to_path_buf();
        let mut comparison = ResourceComparison {
            version: crate::acceptance::RESOURCE_COMPARISON_VERSION,
            baseline_id: run.run.metadata.baseline.id.clone(),
            candidate_id: run.run.metadata.candidate.id.clone(),
            benchmark_run_sha256: run.sha256.clone(),
            strongest_baseline_id: run.run.metadata.baseline.id.clone(),
            measurement_evaluator_id: "hermes-resource-evaluator".into(),
            measurement_evaluator_version: format!("sha256:{}", "c".repeat(64)),
            wake_trials: (0..3)
                .map(|trial| crate::acceptance::PairedWakeTrial {
                    trial,
                    baseline: crate::acceptance::WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                    candidate: crate::acceptance::WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 1_000.0 / 96.0,
                        request_latency_ms: vec![10.4],
                    },
                })
                .collect(),
            candidate_capacity: (0..=4)
                .map(
                    |completed_sleep_cycles| crate::acceptance::CapacityObservation {
                        completed_sleep_cycles,
                        routed_active_parameters: run
                            .run
                            .metadata
                            .candidate
                            .routed_active_parameters,
                        stored_parameters: run.run.metadata.candidate.parameters,
                        stored_bytes: run.run.metadata.candidate.stored_bytes,
                    },
                )
                .collect(),
            grouped_mm_parity: crate::acceptance::KernelParityEvidence {
                fixture_sha256: "a".repeat(64),
                samples: (0..1024)
                    .map(|_| crate::acceptance::KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0 + 1e-5,
                    })
                    .collect(),
            },
            pytorch_parity: crate::acceptance::KernelParityEvidence {
                fixture_sha256: "b".repeat(64),
                samples: (0..1024)
                    .map(|_| crate::acceptance::KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0 + 1e-5,
                    })
                    .collect(),
            },
            exact_resume: crate::acceptance::ExactResumeEvidence {
                interrupted_checkpoint: crate::acceptance::ExactResumeArtifact {
                    path: relative(&interrupted_manifest),
                    sha256: interrupted_sha256,
                },
                uninterrupted_final_state: crate::acceptance::ExactResumeArtifact {
                    path: relative(final_manifest),
                    sha256: run
                        .run
                        .metadata
                        .candidate
                        .checkpoint_manifest_sha256
                        .clone(),
                },
                resumed_final_state: crate::acceptance::ExactResumeArtifact {
                    path: relative(&resumed_manifest),
                    sha256: run
                        .run
                        .metadata
                        .candidate
                        .checkpoint_manifest_sha256
                        .clone(),
                },
                uninterrupted_metrics: crate::acceptance::ExactResumeArtifact {
                    sha256: sha256_hex(&fs::read(&uninterrupted_metrics).unwrap()),
                    path: relative(&uninterrupted_metrics),
                },
                resumed_metrics: crate::acceptance::ExactResumeArtifact {
                    sha256: sha256_hex(&fs::read(&resumed_metrics).unwrap()),
                    path: relative(&resumed_metrics),
                },
                interruption_step: 10,
                resumed_from_step: 10,
            },
            execution: crate::acceptance::ResourceExecutionReceipt {
                protocol_version: crate::acceptance::RESOURCE_EXECUTION_PROTOCOL_VERSION,
                evaluator_sha256: format!("sha256:{}", "c".repeat(64)),
                request_sha256: format!("sha256:{}", "d".repeat(64)),
                observations_sha256: format!("sha256:{}", "e".repeat(64)),
                baseline_target_sha256: format!("sha256:{}", "1".repeat(64)),
                candidate_target_sha256: format!("sha256:{}", "2".repeat(64)),
                policy_sha256: "f".repeat(64),
                evaluator_arguments: Vec::new(),
                approved_artifact_roots: vec![approved_root],
            },
        };
        comparison.execution.observations_sha256 = comparison.observations_sha256().unwrap();
        comparison
    }

    fn write_resources(root: &Path, comparison: &ResourceComparison) -> VerifiedResourceComparison {
        let path = root.join("resources.json");
        let bytes = serde_json::to_vec_pretty(comparison).unwrap();
        fs::write(&path, &bytes).unwrap();
        VerifiedResourceComparison::load(path, &sha256_hex(&bytes)).unwrap()
    }

    #[test]
    fn verified_runner_is_paired_ordered_and_acceptance_ready() {
        let temporary = tempfile::tempdir().unwrap();
        // Supply reverse lexical order to ensure suite order is canonical.
        let sealed = write_suite(
            temporary.path(),
            "z-sealed",
            SuiteVisibility::Sealed,
            "sealed-loss",
            MetricDirection::Minimize,
        );
        let public = write_suite(
            temporary.path(),
            "a-public",
            SuiteVisibility::Public,
            "public-score",
            MetricDirection::Maximize,
        );
        let baseline = target(temporary.path(), "baseline", 100);
        let candidate = target(temporary.path(), "candidate", 100);
        let runner = BenchmarkRunner {
            config: BenchmarkRunConfig {
                evaluator_id: "mock".into(),
                evaluator_version: "1".into(),
                paired_seeds: vec![11, 22, 33],
                order_seed: 7,
                gpu_hours_per_evaluation: 0.25,
                ablation: Some(AblationId::WakeOnly),
            },
        };
        let mut evaluator = MockEvaluator { calls: Vec::new() };
        let run = runner
            .run(&[sealed, public], &baseline, &candidate, &mut evaluator)
            .unwrap();

        assert_eq!(
            run.metadata.fixed_case_order,
            ["public-score", "sealed-loss"]
        );
        assert!(run.capacity_matched());
        assert_eq!(run.metadata.gpu_hour_budget_per_target, 1.5);
        assert_eq!(evaluator.calls.len(), 12);
        for calls in evaluator.calls.chunks_exact(2) {
            assert_eq!(calls[0].0, calls[1].0);
            assert_eq!(calls[0].1, calls[1].1);
            assert_eq!(calls[0].2, calls[1].2);
            assert_eq!(calls[0].3, TargetRole::Baseline);
            assert_eq!(calls[1].3, TargetRole::Candidate);
        }

        let (suite, results) = run.acceptance_inputs();
        suite.validate().unwrap();
        assert_eq!(results.len(), 2);
        for result in results {
            assert_eq!(result.pairs.len(), 3);
            for pair in result.pairs {
                assert!((pair.candidate - pair.baseline - 0.1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn manifest_and_artifact_hashes_are_mandatory() {
        let temporary = tempfile::tempdir().unwrap();
        let suite = write_suite(
            temporary.path(),
            "public",
            SuiteVisibility::Public,
            "case",
            MetricDirection::Maximize,
        );
        assert!(VerifiedBenchmarkSuite::load(&suite.manifest_path, &"0".repeat(64),).is_err());

        fs::write(suite.artifact("case").unwrap().path.clone(), b"mutated").unwrap();
        let manifest_bytes = fs::read(&suite.manifest_path).unwrap();
        assert!(
            VerifiedBenchmarkSuite::load(&suite.manifest_path, &sha256_hex(&manifest_bytes),)
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn suite_manifest_and_case_artifacts_reject_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let suite = write_suite(
            temporary.path(),
            "public",
            SuiteVisibility::Public,
            "case",
            MetricDirection::Maximize,
        );

        let linked_manifest = temporary.path().join("linked-manifest.json");
        symlink(&suite.manifest_path, &linked_manifest).unwrap();
        let error = VerifiedBenchmarkSuite::load(&linked_manifest, &suite.manifest_sha256)
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-symlink"), "{error}");

        let linked_artifact = temporary.path().join("linked-artifact.jsonl");
        symlink(
            suite.artifact("case").unwrap().path.clone(),
            &linked_artifact,
        )
        .unwrap();
        let mut manifest = suite.manifest.clone();
        manifest.cases[0].artifact.path = linked_artifact.file_name().unwrap().into();
        let manifest_path = temporary.path().join("artifact-link-manifest.json");
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&manifest_path, &bytes).unwrap();
        let error = VerifiedBenchmarkSuite::load(&manifest_path, &sha256_hex(&bytes))
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-symlink"), "{error}");

        let original_artifact = suite.artifact("case").unwrap().path.clone();
        let moved_artifact = temporary.path().join("moved-artifact.jsonl");
        fs::rename(&original_artifact, &moved_artifact).unwrap();
        symlink(&moved_artifact, &original_artifact).unwrap();
        let error = suite.verify_contents().unwrap_err().to_string();
        assert!(error.contains("non-symlink"), "{error}");

        let real_directory = temporary.path().join("real-suite");
        fs::create_dir(&real_directory).unwrap();
        let nested = write_suite(
            &real_directory,
            "nested",
            SuiteVisibility::Public,
            "nested-case",
            MetricDirection::Maximize,
        );
        let linked_directory = temporary.path().join("linked-suite");
        symlink(&real_directory, &linked_directory).unwrap();
        let linked_path = linked_directory.join("nested.json");
        let error = VerifiedBenchmarkSuite::load(&linked_path, &nested.manifest_sha256)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("parent") && error.contains("symlink"),
            "{error}"
        );
    }

    #[test]
    fn target_resource_claims_are_bound_to_training_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let mut candidate = target(temporary.path(), "candidate", 100);
        candidate.verify().unwrap();

        candidate.training_gpu_hours = 7.5;
        let error = candidate.verify().unwrap_err().to_string();
        assert!(error.contains("resource claims"), "{error}");

        candidate.training_gpu_hours = 8.0;
        fs::write(&candidate.training_evidence, b"{}").unwrap();
        let error = candidate.verify().unwrap_err().to_string();
        assert!(error.contains("hash mismatch"), "{error}");
    }

    #[test]
    fn hquant_target_authenticates_candidate_source_archive_and_storage() {
        let temporary = tempfile::tempdir().unwrap();
        let full_precision = target(temporary.path(), "candidate", 100);
        let full_precision_bytes = full_precision.stored_bytes;
        let VerifiedModelRepresentation::FullPrecision {
            weights_sha256: full_precision_sha256,
            ..
        } = full_precision.verify().unwrap()
        else {
            unreachable!()
        };
        let candidate = attach_hquant_candidate(temporary.path(), full_precision.clone());
        assert!(candidate.stored_bytes < full_precision_bytes);
        let verified = candidate.verify().unwrap();
        let VerifiedModelRepresentation::Hquant {
            candidate_manifest,
            archive,
            source_weights_sha256,
            stored_bytes,
            ..
        } = verified
        else {
            panic!("expected verified HQUANT representation")
        };
        assert_eq!(candidate_manifest.file_name().unwrap(), "candidate.json");
        assert_eq!(archive.file_name().unwrap(), "hquant");
        assert_eq!(
            source_weights_sha256,
            format!("sha256:{full_precision_sha256}")
        );
        assert_eq!(stored_bytes, candidate.stored_bytes);
    }

    #[test]
    fn hquant_target_rejects_substitution_hash_archive_and_storage_tampering() {
        let temporary = tempfile::tempdir().unwrap();

        let original =
            attach_hquant_candidate(temporary.path(), target(temporary.path(), "original", 100));
        let substitute = attach_hquant_candidate(
            temporary.path(),
            target(temporary.path(), "substitute", 100),
        );
        let mut substituted = original.clone();
        substituted.representation = substitute.representation;
        substituted.stored_bytes = substitute.stored_bytes;
        let error = substituted.verify().unwrap_err().to_string();
        assert!(error.contains("different source weights"), "{error}");

        let mut wrong_hash = attach_hquant_candidate(
            temporary.path(),
            target(temporary.path(), "wrong-hash", 100),
        );
        let ModelRepresentationTarget::Hquant {
            candidate_manifest_sha256,
            ..
        } = &mut wrong_hash.representation
        else {
            unreachable!()
        };
        *candidate_manifest_sha256 = format!("sha256:{}", "0".repeat(64));
        let error = wrong_hash.verify().unwrap_err().to_string();
        assert!(error.contains("manifest identity mismatch"), "{error}");

        let mut wrong_storage = attach_hquant_candidate(
            temporary.path(),
            target(temporary.path(), "wrong-storage", 100),
        );
        wrong_storage.stored_bytes += 1;
        let error = wrong_storage.verify().unwrap_err().to_string();
        assert!(error.contains("validated archive members"), "{error}");

        let corrupt = attach_hquant_candidate(
            temporary.path(),
            target(temporary.path(), "corrupt-archive", 100),
        );
        let VerifiedModelRepresentation::Hquant { archive, .. } = corrupt.verify().unwrap() else {
            unreachable!()
        };
        let opened = crate::quantization::QuantizedArchive::open(&archive).unwrap();
        let member = opened
            .manifest()
            .matrices
            .first()
            .map(|matrix| archive.join(&matrix.file))
            .unwrap();
        let mut bytes = fs::read(&member).unwrap();
        bytes[0] ^= 1;
        fs::write(member, bytes).unwrap();
        let error = corrupt.verify().unwrap_err().to_string();
        assert!(error.contains("invalid sealed HQUANT candidate"), "{error}");
    }

    struct RepresentationEvaluator {
        hquant_candidate_calls: usize,
    }

    impl BenchmarkEvaluator for RepresentationEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            match request.role {
                TargetRole::Baseline => ensure!(
                    matches!(
                        request.model,
                        VerifiedModelRepresentation::FullPrecision { .. }
                    ),
                    "baseline did not receive FP weights"
                ),
                TargetRole::Candidate => {
                    let VerifiedModelRepresentation::Hquant {
                        archive,
                        archive_manifest,
                        ..
                    } = request.model
                    else {
                        anyhow::bail!("candidate did not receive HQUANT archive")
                    };
                    ensure!(archive.is_dir(), "candidate HQUANT archive is unavailable");
                    ensure!(
                        archive_manifest.is_file(),
                        "candidate HQUANT manifest is unavailable"
                    );
                    self.hquant_candidate_calls += 1;
                }
            }
            Ok(EvaluationMeasurement {
                score: 0.5,
                gpu_hours: request.max_gpu_hours,
                examples: 1,
                metrics: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn runner_propagates_verified_hquant_archive_to_evaluator() {
        let temporary = tempfile::tempdir().unwrap();
        let suite = write_suite(
            temporary.path(),
            "public-hquant",
            SuiteVisibility::Public,
            "quality",
            MetricDirection::Maximize,
        );
        let baseline = target(temporary.path(), "baseline-hquant", 100);
        let candidate = attach_hquant_candidate(
            temporary.path(),
            target(temporary.path(), "candidate-hquant", 100),
        );
        let runner = BenchmarkRunner {
            config: BenchmarkRunConfig {
                evaluator_id: "hquant-backend".into(),
                evaluator_version: "1".into(),
                paired_seeds: vec![1, 2, 3],
                order_seed: 9,
                gpu_hours_per_evaluation: 0.25,
                ablation: None,
            },
        };
        let mut evaluator = RepresentationEvaluator {
            hquant_candidate_calls: 0,
        };
        let run = runner
            .run(&[suite], &baseline, &candidate, &mut evaluator)
            .unwrap();
        assert_eq!(evaluator.hquant_candidate_calls, 3);
        assert!(matches!(
            run.metadata.candidate.representation,
            ModelRepresentationTarget::Hquant { .. }
        ));
        assert!(!same_target_identity(&baseline, &candidate));
    }

    #[test]
    fn self_consistent_outer_evidence_cannot_override_sealed_accounting() {
        let temporary = tempfile::tempdir().unwrap();
        let mut candidate = target(temporary.path(), "candidate", 100);
        let mut evidence: TrainingEvidence =
            serde_json::from_slice(&fs::read(&candidate.training_evidence).unwrap()).unwrap();
        evidence.parameters = 90;
        candidate.parameters = 90;
        let bytes = serde_json::to_vec(&evidence).unwrap();
        let digest = sha256_hex(&bytes);
        let path = candidate
            .training_evidence
            .parent()
            .unwrap()
            .join(format!("sha256-{digest}.json"));
        fs::write(&path, bytes).unwrap();
        candidate.training_evidence = path;
        candidate.training_evidence_sha256 = digest;
        let error = candidate.verify().unwrap_err().to_string();
        assert!(error.contains("resource claims"), "{error}");
    }

    #[test]
    fn target_verification_hashes_actual_weights_not_only_the_manifest() {
        let temporary = tempfile::tempdir().unwrap();
        let candidate = target(temporary.path(), "candidate", 100);
        let weights = candidate
            .checkpoint_manifest
            .parent()
            .unwrap()
            .join(CHECKPOINT_WEIGHTS_FILE);
        let len = fs::metadata(&weights).unwrap().len() as usize;
        fs::write(&weights, vec![0_u8; len]).unwrap();
        let error = candidate.verify().unwrap_err().to_string();
        assert!(error.contains("checkpoint manifest"), "{error}");
    }

    #[test]
    fn exact_resume_gate_recomputes_semantic_progress_from_metric_artifacts() {
        use crate::metrics::{
            MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, MetricWriter,
            ThroughputMetrics,
        };

        let temporary = tempfile::tempdir().unwrap();
        let public = write_catalog_suite(temporary.path(), "public", SuiteVisibility::Public);
        let sealed = write_catalog_suite(temporary.path(), "sealed", SuiteVisibility::Sealed);
        let baseline = target(temporary.path(), "baseline-0", 100);
        let candidate = target(temporary.path(), "candidate", 100);
        let run = BenchmarkRunner {
            config: BenchmarkRunConfig {
                evaluator_id: "fixed".into(),
                evaluator_version: "sha256:evaluator".into(),
                paired_seeds: vec![1, 2, 3],
                order_seed: 7,
                gpu_hours_per_evaluation: 0.25,
                ablation: Some(AblationId::WakeOnly),
            },
        }
        .run(
            &[public, sealed],
            &baseline,
            &candidate,
            &mut RankingEvaluator,
        )
        .unwrap();
        let run = write_run(temporary.path(), 99, &run);
        let mut comparison = resource_comparison(&run);

        let path = temporary
            .path()
            .join(&comparison.exact_resume.resumed_metrics.path);
        let mut writer = MetricWriter::create(&path, "resumed-tampered").unwrap();
        writer
            .append_at(
                MetricContext {
                    global_step: 20,
                    phase: MetricPhase {
                        index: 0,
                        name: "test".into(),
                        kind: MetricPhaseKind::Pretrain,
                    },
                    checkpoint_hash: None,
                },
                MetricEvent::Throughput(ThroughputMetrics {
                    optimizer_steps: 20,
                    compute_tokens: 201,
                    supervised_tokens: 201,
                    examples: 20,
                    elapsed_seconds: 3.0,
                    tokens_per_second: 67.0,
                    examples_per_second: 20.0 / 3.0,
                    input_wait_seconds: 0.0,
                    host_to_device_seconds: 0.0,
                    gpu_busy_seconds: 3.0,
                }),
                300,
            )
            .unwrap();
        writer.sync_all().unwrap();
        drop(writer);
        comparison.exact_resume.resumed_metrics.sha256 = sha256_hex(&fs::read(&path).unwrap());
        comparison.execution.observations_sha256 = comparison.observations_sha256().unwrap();
        let resource_path = temporary.path().join("forged-resources.json");
        let bytes = serde_json::to_vec(&comparison).unwrap();
        fs::write(&resource_path, &bytes).unwrap();
        let error = VerifiedResourceComparison::load(&resource_path, &sha256_hex(&bytes))
            .unwrap_err()
            .to_string();
        assert!(error.contains("differ semantically"), "{error}");
    }

    #[test]
    fn target_paths_resolve_against_their_manifest() {
        let mut target = BenchmarkTarget {
            id: "candidate".into(),
            checkpoint_manifest: "artifacts/checkpoint.json".into(),
            checkpoint_manifest_sha256: "1".repeat(64),
            training_evidence: "evidence/training.json".into(),
            training_evidence_sha256: "2".repeat(64),
            training_gpu_hours: 1.0,
            parameters: 10,
            routed_active_parameters: 8,
            stored_bytes: 100,
            representation: ModelRepresentationTarget::Hquant {
                candidate_manifest: "qat/candidate.json".into(),
                candidate_manifest_sha256: format!("sha256:{}", "3".repeat(64)),
            },
        };
        target.resolve_paths(Path::new("/runs/target.json"));
        assert_eq!(
            target.checkpoint_manifest,
            Path::new("/runs/artifacts/checkpoint.json")
        );
        assert_eq!(
            target.training_evidence,
            Path::new("/runs/evidence/training.json")
        );
        let ModelRepresentationTarget::Hquant {
            candidate_manifest, ..
        } = target.representation
        else {
            panic!("expected HQUANT target")
        };
        assert_eq!(candidate_manifest, Path::new("/runs/qat/candidate.json"));
    }

    #[test]
    fn target_schema_requires_an_explicit_model_representation() {
        let temporary = tempfile::tempdir().unwrap();
        let target = target(temporary.path(), "schema-target", 100);
        let mut value = serde_json::to_value(&target).unwrap();
        value.as_object_mut().unwrap().remove("representation");
        assert!(serde_json::from_value::<BenchmarkTarget>(value).is_err());

        let mut hquant = target.clone();
        hquant.representation = ModelRepresentationTarget::Hquant {
            candidate_manifest: temporary.path().join("candidate.json"),
            candidate_manifest_sha256: format!("sha256:{}", "a".repeat(64)),
        };
        assert_ne!(
            target.representation_identity(),
            hquant.representation_identity()
        );
        assert!(!same_target_identity(&target, &hquant));
    }

    #[test]
    fn runner_rejects_fewer_than_three_or_reordered_seeds() {
        let config = BenchmarkRunConfig {
            evaluator_id: "mock".into(),
            evaluator_version: "1".into(),
            paired_seeds: vec![2, 1, 3],
            order_seed: 0,
            gpu_hours_per_evaluation: 1.0,
            ablation: None,
        };
        assert!(config.validate().is_err());

        let config = BenchmarkRunConfig {
            paired_seeds: vec![1, 2],
            ..config
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn catalog_and_ablation_ids_are_complete_and_stable() {
        let catalog = required_catalog();
        let ids = catalog
            .iter()
            .map(|entry| entry.id)
            .collect::<BTreeSet<_>>();
        for id in [
            "pretraining_causal",
            "summarization",
            "retrieval_representation_ranking",
            "retrieval_planning",
            "reasoning_qa",
            "pairwise_preference",
            "verifiable_rl",
            "synthetic_retention_smoke",
            "clinc_continual_learning",
            "banking77_continual_learning",
            "mk_niah",
            "qasper",
            "squad_no_context",
            "arc_dreaming",
            "manchu_continual_learning",
            "kalamang_continual_learning",
            "babilong",
        ] {
            assert!(ids.contains(id));
        }
        assert_eq!(AblationId::ALL.len(), 11);
        assert_eq!(
            AblationId::CmsWithoutKnowledgeSeeding.as_str(),
            "cms_without_knowledge_seeding"
        );
    }

    #[test]
    fn promotion_is_derived_from_content_addressed_complete_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let public =
            write_catalog_suite(temporary.path(), "public-catalog", SuiteVisibility::Public);
        let sealed =
            write_catalog_suite(temporary.path(), "sealed-catalog", SuiteVisibility::Sealed);
        validate_catalog_coverage(&[public.clone(), sealed.clone()], false).unwrap();

        let candidate = target(temporary.path(), "candidate", 100);
        let mut runs = Vec::new();
        for (index, ablation) in AblationId::ALL.into_iter().enumerate() {
            let baseline = target(temporary.path(), &format!("baseline-{index:02}"), 100);
            let runner = BenchmarkRunner {
                config: BenchmarkRunConfig {
                    evaluator_id: "fixed-evaluator".into(),
                    evaluator_version: "sha256:evaluator".into(),
                    paired_seeds: vec![11, 22, 33],
                    order_seed: 7,
                    gpu_hours_per_evaluation: 0.25,
                    ablation: Some(ablation),
                },
            };
            let run = runner
                .run(
                    &[public.clone(), sealed.clone()],
                    &baseline,
                    &candidate,
                    &mut RankingEvaluator,
                )
                .unwrap();
            runs.push(write_run(temporary.path(), index, &run));
        }

        // RankingEvaluator monotonically increases baseline strength.
        let selected = runs.last().unwrap();
        let policy = AcceptancePolicy::default();
        let policy_sha256 = "f".repeat(64);
        let mut comparison = resource_comparison(selected);
        comparison.execution = crate::resource_worker::derive_execution_receipt(
            selected,
            &runs,
            &policy,
            &policy_sha256,
            Vec::new(),
            comparison.execution.approved_artifact_roots.clone(),
            &comparison,
        )
        .unwrap();
        let resources = write_resources(temporary.path(), &comparison);
        assert!(
            [
                &resources.comparison.exact_resume.interrupted_checkpoint,
                &resources.comparison.exact_resume.uninterrupted_final_state,
                &resources.comparison.exact_resume.resumed_final_state,
                &resources.comparison.exact_resume.uninterrupted_metrics,
                &resources.comparison.exact_resume.resumed_metrics,
            ]
            .iter()
            .all(|artifact| artifact.path.is_relative()),
            "reopening content-addressed evidence must retain portable signed paths"
        );
        let report =
            evaluate_verified_promotion(selected, &runs, &resources, &policy, &policy_sha256)
                .unwrap();
        assert!(report.accepted);
        assert!(report.resource_gates["content_addressed_evidence"]);
        assert!(report.resource_gates["strongest_matched_baseline"]);
        assert_eq!(report.cases.len(), required_catalog().len() - 3);
        assert_eq!(report.sealed.case_count, required_catalog().len() - 3);
        let run_evidence = serde_json::to_string(&selected.run).unwrap();
        assert!(run_evidence.contains("sealed-arc_dreaming"));
        assert!(run_evidence.contains("sealed-pretraining_causal"));
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("sealed-arc_dreaming"));
        assert!(!serialized.contains("sealed-pretraining_causal"));

        let mut in_memory_resources = resources.clone();
        in_memory_resources.comparison.wake_trials[0]
            .candidate
            .elapsed_seconds = 0.001;
        let error = in_memory_resources
            .verify_contents()
            .unwrap_err()
            .to_string();
        assert!(error.contains("in-memory resource evidence"), "{error}");

        let mut in_memory_run = selected.clone();
        in_memory_run.run.metadata.candidate.id = "substituted-candidate".into();
        let error = in_memory_run.verify_contents().unwrap_err().to_string();
        assert!(error.contains("in-memory benchmark run"), "{error}");

        let resource_bytes = fs::read(&resources.path).unwrap();
        fs::write(&resources.path, b"{}").unwrap();
        assert!(
            evaluate_verified_promotion(selected, &runs, &resources, &policy, &policy_sha256)
                .is_err()
        );
        fs::write(&resources.path, resource_bytes).unwrap();

        fs::write(&selected.path, b"{}").unwrap();
        assert!(
            evaluate_verified_promotion(selected, &runs, &resources, &policy, &policy_sha256)
                .is_err()
        );
    }

    #[test]
    fn coverage_requires_each_public_and_sealed_catalog_case() {
        let temporary = tempfile::tempdir().unwrap();
        let public =
            write_catalog_suite(temporary.path(), "public-catalog", SuiteVisibility::Public);
        let sealed =
            write_catalog_suite(temporary.path(), "sealed-catalog", SuiteVisibility::Sealed);
        assert!(validate_catalog_coverage(std::slice::from_ref(&public), false).is_err());

        let mut incomplete = sealed;
        incomplete.manifest.cases.pop();
        assert!(validate_catalog_coverage(&[public, incomplete], false).is_err());
    }

    #[test]
    fn benchmark_suite_manifest_v1_is_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let mut suite = write_suite(
            temporary.path(),
            "legacy-suite",
            SuiteVisibility::Public,
            "legacy-case",
            MetricDirection::Maximize,
        )
        .manifest;
        suite.version = 1;
        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("unsupported benchmark manifest version 1"));
    }
}
