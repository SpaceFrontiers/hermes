//! Deterministic execution of public and sealed model acceptance benchmarks.
//!
//! A suite manifest and every data artifact are content addressed.  The runner
//! never downloads data: callers materialize public or sealed suites locally,
//! then provide the expected manifest digest out of band.  Baseline and
//! candidate evaluations receive the same model seed, example-order seed, and
//! GPU-hour allowance.  Raw measurements can be converted directly into the
//! higher-is-better paired inputs consumed by [`crate::acceptance`].

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::acceptance::{
    AcceptancePolicy, AcceptanceSuite, BenchmarkCase, BenchmarkFamily, CaseResult, PairedRun,
    PromotionReport, ResourceComparison, SuiteVisibility, VerifiedPromotionContext,
    evaluate_with_verified_context,
};

pub const BENCHMARK_MANIFEST_VERSION: u32 = 1;
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
    #[serde(default)]
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
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read benchmark manifest {}", path.display()))?;
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
            let artifact_bytes = fs::read(&artifact_path).with_context(|| {
                format!(
                    "failed to read artifact for benchmark `{}` at {}",
                    case.id,
                    artifact_path.display()
                )
            })?;
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
        let manifest_bytes = fs::read(&self.manifest_path).with_context(|| {
            format!(
                "failed to re-read benchmark manifest {}",
                self.manifest_path.display()
            )
        })?;
        let actual_manifest = sha256_hex(&manifest_bytes);
        ensure!(
            actual_manifest == self.manifest_sha256,
            "benchmark manifest changed after verification: {}",
            self.manifest_path.display()
        );
        for (case_id, artifact) in &self.artifacts {
            let bytes = fs::read(&artifact.path).with_context(|| {
                format!(
                    "failed to re-read artifact for benchmark `{case_id}` at {}",
                    artifact.path.display()
                )
            })?;
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

/// Immutable model identity passed to an evaluator.  The path addresses a
/// checkpoint manifest (which may in turn address sharded tensors).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkTarget {
    pub id: String,
    pub checkpoint_manifest: PathBuf,
    pub checkpoint_manifest_sha256: String,
    /// Content identity of the immutable trainer metrics/accounting manifest.
    pub training_evidence_sha256: String,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub stored_bytes: u64,
}

impl BenchmarkTarget {
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
        Ok(())
    }

    pub fn verify(&self) -> Result<()> {
        self.validate_identity()?;
        let bytes = fs::read(&self.checkpoint_manifest).with_context(|| {
            format!(
                "failed to read checkpoint manifest for `{}` at {}",
                self.id,
                self.checkpoint_manifest.display()
            )
        })?;
        let actual = sha256_hex(&bytes);
        ensure!(
            actual.eq_ignore_ascii_case(&self.checkpoint_manifest_sha256),
            "checkpoint manifest hash mismatch for `{}`: expected {}, got {}",
            self.id,
            self.checkpoint_manifest_sha256,
            actual
        );
        Ok(())
    }
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

/// Fully deterministic evaluator input.  The evaluator must consume examples
/// in `example_order_seed` order and stop at `max_gpu_hours`.
pub struct EvaluationRequest<'a> {
    pub suite_id: &'a str,
    pub visibility: SuiteVisibility,
    pub case: &'a BenchmarkSpec,
    pub artifact: &'a VerifiedArtifact,
    pub target: &'a BenchmarkTarget,
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
/// Artifact materialization and network access deliberately remain outside
/// this interface.
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
            version: 1,
            cases: self
                .cases
                .iter()
                .map(|case| BenchmarkCase {
                    id: case.case_id.clone(),
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
    pub run: BenchmarkRun,
    pub path: PathBuf,
    pub sha256: String,
}

impl VerifiedBenchmarkRun {
    pub fn load(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        validate_sha256(expected_sha256).context("invalid expected benchmark run hash")?;
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read benchmark run {}", path.display()))?;
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
        let bytes = fs::read(&self.path)
            .with_context(|| format!("failed to re-read benchmark run {}", self.path.display()))?;
        ensure!(
            sha256_hex(&bytes) == self.sha256,
            "benchmark run changed after verification: {}",
            self.path.display()
        );
        Ok(())
    }
}

/// Content-addressed performance, memory, kernel, and resume measurements.
#[derive(Clone, Debug)]
pub struct VerifiedResourceComparison {
    pub comparison: ResourceComparison,
    pub path: PathBuf,
    pub sha256: String,
}

impl VerifiedResourceComparison {
    pub fn load(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        validate_sha256(expected_sha256).context("invalid expected resource evidence hash")?;
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read resource evidence {}", path.display()))?;
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
        Ok(Self {
            comparison,
            path: path.to_path_buf(),
            sha256: actual,
        })
    }

    pub fn verify_contents(&self) -> Result<()> {
        let bytes = fs::read(&self.path).with_context(|| {
            format!(
                "failed to re-read resource evidence {}",
                self.path.display()
            )
        })?;
        ensure!(
            sha256_hex(&bytes) == self.sha256,
            "resource evidence changed after verification: {}",
            self.path.display()
        );
        Ok(())
    }
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
) -> Result<PromotionReport> {
    selected_run.verify_contents()?;
    resources.verify_contents()?;
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
    let strongest_baseline_id = derive_strongest_baseline(comparison_runs)?;

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
        comparison.baseline_stored_parameters == selected_run.run.metadata.baseline.parameters
            && comparison.candidate_stored_parameters
                == selected_run.run.metadata.candidate.parameters,
        "resource evidence stored-parameter counts do not match benchmark targets"
    );
    ensure!(
        comparison.baseline_stored_bytes == selected_run.run.metadata.baseline.stored_bytes
            && comparison.candidate_stored_bytes
                == selected_run.run.metadata.candidate.stored_bytes,
        "resource evidence stored-byte counts do not match benchmark targets"
    );
    ensure!(
        comparison.baseline_routed_active_parameters
            == selected_run.run.metadata.baseline.routed_active_parameters
            && comparison.candidate_routed_active_parameters
                == selected_run.run.metadata.candidate.routed_active_parameters,
        "resource evidence active-parameter counts do not match benchmark targets"
    );
    ensure!(
        comparison.exact_resume.uninterrupted_final_state_sha256
            == selected_run
                .run
                .metadata
                .candidate
                .checkpoint_manifest_sha256,
        "exact-resume evidence does not terminate at the candidate checkpoint"
    );
    ensure!(
        same_f64(
            comparison.baseline_training_gpu_hours,
            selected_run.run.metadata.baseline.training_gpu_hours,
        ) && same_f64(
            comparison.candidate_training_gpu_hours,
            selected_run.run.metadata.candidate.training_gpu_hours,
        ),
        "resource evidence GPU-hour measurements do not match target training evidence"
    );

    let (suite, results) = selected_run.run.acceptance_inputs();
    let context = VerifiedPromotionContext {
        benchmark_run_sha256: &selected_run.sha256,
        strongest_baseline_id: &strongest_baseline_id,
        baseline_id: &selected_run.run.metadata.baseline.id,
        candidate_id: &selected_run.run.metadata.candidate.id,
        capacity_matched: selected_run.run.capacity_matched(),
        active_capacity_matched: selected_run.run.routed_active_parameters_matched(),
        fixed_gpu_hour_measured: selected_run.run.training_compute_matched(),
        content_addressed_evidence: true,
    };
    evaluate_with_verified_context(&suite, &results, comparison, policy, &context)
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
        baseline.verify()?;
        candidate.verify()?;

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
        let path = root.join(format!("{id}-checkpoint.json"));
        fs::write(&path, format!("{{\"id\":\"{id}\"}}")).unwrap();
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest_sha256: sha256_hex(&fs::read(&path).unwrap()),
            training_evidence_sha256: "f".repeat(64),
            training_gpu_hours: 8.0,
            checkpoint_manifest: path,
            parameters,
            routed_active_parameters: 80,
            stored_bytes: 4_096,
        }
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
                    stable_anchor: false,
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
        ResourceComparison {
            baseline_id: run.run.metadata.baseline.id.clone(),
            candidate_id: run.run.metadata.candidate.id.clone(),
            benchmark_run_sha256: run.sha256.clone(),
            strongest_baseline_id: run.run.metadata.baseline.id.clone(),
            baseline_stored_parameters: run.run.metadata.baseline.parameters,
            candidate_stored_parameters: run.run.metadata.candidate.parameters,
            maximum_candidate_stored_parameters: run.run.metadata.baseline.parameters,
            baseline_stored_bytes: 4_096,
            candidate_stored_bytes: 4_096,
            maximum_candidate_stored_bytes: 4_096,
            baseline_routed_active_parameters: run.run.metadata.baseline.routed_active_parameters,
            candidate_routed_active_parameters: run.run.metadata.candidate.routed_active_parameters,
            baseline_training_gpu_hours: run.run.metadata.baseline.training_gpu_hours,
            candidate_training_gpu_hours: run.run.metadata.candidate.training_gpu_hours,
            baseline_wake_tokens_per_second: 100.0,
            candidate_wake_tokens_per_second: 96.0,
            baseline_wake_p95_ms: 10.0,
            candidate_wake_p95_ms: 10.4,
            active_parameters_before: run.run.metadata.candidate.routed_active_parameters,
            active_parameters_after: run.run.metadata.candidate.routed_active_parameters,
            grouped_mm_parity: crate::acceptance::KernelParityEvidence {
                fixture_sha256: "a".repeat(64),
                samples: 1024,
                max_absolute_error: 1e-5,
                max_relative_error: 1e-4,
                absolute_tolerance: 2e-5,
                relative_tolerance: 2e-4,
            },
            pytorch_parity: crate::acceptance::KernelParityEvidence {
                fixture_sha256: "b".repeat(64),
                samples: 1024,
                max_absolute_error: 1e-5,
                max_relative_error: 1e-4,
                absolute_tolerance: 2e-5,
                relative_tolerance: 2e-4,
            },
            exact_resume: crate::acceptance::ExactResumeEvidence {
                interrupted_checkpoint_sha256: "c".repeat(64),
                uninterrupted_final_state_sha256: run
                    .run
                    .metadata
                    .candidate
                    .checkpoint_manifest_sha256
                    .clone(),
                resumed_final_state_sha256: run
                    .run
                    .metadata
                    .candidate
                    .checkpoint_manifest_sha256
                    .clone(),
                uninterrupted_metrics_sha256: "e".repeat(64),
                resumed_metrics_sha256: "e".repeat(64),
                interruption_step: 10,
                resumed_from_step: 10,
            },
        }
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
        let resources = write_resources(temporary.path(), &resource_comparison(selected));
        let report =
            evaluate_verified_promotion(selected, &runs, &resources, &AcceptancePolicy::default())
                .unwrap();
        assert!(report.accepted);
        assert!(report.resource_gates["content_addressed_evidence"]);
        assert!(report.resource_gates["strongest_matched_baseline"]);
        assert_eq!(report.cases.len(), required_catalog().len() - 3);
        assert_eq!(report.sealed.case_count, required_catalog().len() - 3);
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("sealed-arc_dreaming"));
        assert!(!serialized.contains("sealed-pretraining_causal"));

        let resource_bytes = fs::read(&resources.path).unwrap();
        fs::write(&resources.path, b"{}").unwrap();
        assert!(
            evaluate_verified_promotion(selected, &runs, &resources, &AcceptancePolicy::default(),)
                .is_err()
        );
        fs::write(&resources.path, resource_bytes).unwrap();

        fs::write(&selected.path, b"{}").unwrap();
        assert!(
            evaluate_verified_promotion(selected, &runs, &resources, &AcceptancePolicy::default(),)
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
        assert!(validate_catalog_coverage(&[public.clone()], false).is_err());

        let mut incomplete = sealed;
        incomplete.manifest.cases.pop();
        assert!(validate_catalog_coverage(&[public, incomplete], false).is_err());
    }
}
