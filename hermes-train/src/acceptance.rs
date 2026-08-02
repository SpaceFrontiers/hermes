//! Reproducible acceptance gates for immutable training candidates.
//!
//! Public and sealed suites share the same metric contract. Content-addressed,
//! access-controlled benchmark evidence retains sealed case ids and scores for
//! audit, but the promotion report exposes only their aggregate gate. Suite
//! examples themselves are never copied into either artifact.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteVisibility {
    Public,
    Sealed,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkFamily {
    Pretraining,
    Summarization,
    Retrieval,
    RetrievalPlanning,
    Reasoning,
    Preference,
    VerifiableRl,
    ContinualRetention,
    LongContext,
    FactualIncorporation,
    Dreaming,
    Throughput,
    Resume,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkCase {
    pub id: String,
    /// Stable catalog identity used to bind anchor policy independently of the
    /// suite-local public or sealed case id.
    pub catalog_id: String,
    pub family: BenchmarkFamily,
    pub visibility: SuiteVisibility,
    /// Higher-is-better metric written by the evaluator.
    pub metric: String,
    pub stable_anchor: bool,
}

pub const ACCEPTANCE_SUITE_VERSION: u32 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceSuite {
    pub version: u32,
    pub cases: Vec<BenchmarkCase>,
}

impl AcceptanceSuite {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == ACCEPTANCE_SUITE_VERSION,
            "unsupported acceptance suite version {}",
            self.version
        );
        ensure!(!self.cases.is_empty(), "acceptance suite has no cases");
        let mut ids = BTreeSet::new();
        let mut catalog_entries = BTreeSet::new();
        for case in &self.cases {
            ensure!(!case.id.trim().is_empty(), "acceptance case id is empty");
            ensure!(
                ids.insert(case.id.as_str()),
                "duplicate acceptance case `{}`",
                case.id
            );
            ensure!(
                !case.catalog_id.trim().is_empty(),
                "acceptance case `{}` catalog_id is empty",
                case.id
            );
            ensure!(
                catalog_entries.insert((case.catalog_id.as_str(), case.visibility)),
                "acceptance suite repeats {:?} catalog entry `{}`",
                case.visibility,
                case.catalog_id
            );
            ensure!(
                !case.metric.trim().is_empty(),
                "acceptance case `{}` metric is empty",
                case.id
            );
        }
        ensure!(
            self.cases
                .iter()
                .any(|case| case.visibility == SuiteVisibility::Sealed),
            "acceptance suite must contain a sealed case"
        );
        ensure!(
            self.cases
                .iter()
                .any(|case| case.visibility == SuiteVisibility::Public),
            "acceptance suite must contain a public case"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedRun {
    pub seed: u64,
    pub baseline: f64,
    pub candidate: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub case_id: String,
    pub pairs: Vec<PairedRun>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceComparison {
    pub version: u32,
    pub baseline_id: String,
    pub candidate_id: String,
    /// SHA-256 of the exact [`crate::benchmark::BenchmarkRun`] JSON artifact.
    pub benchmark_run_sha256: String,
    /// This is checked against the baseline identity derived from all verified,
    /// capacity- and compute-matched comparison runs.  The value is never
    /// sufficient by itself to authorize promotion.
    pub strongest_baseline_id: String,
    /// Identity of the fixed harness that emitted all raw measurements below.
    pub measurement_evaluator_id: String,
    pub measurement_evaluator_version: String,
    /// Matched raw observations. Throughput and p95 latency are recomputed by
    /// promotion; evidence cannot submit either aggregate directly.
    pub wake_trials: Vec<PairedWakeTrial>,
    /// Cycle zero is the initial candidate and must be followed by every
    /// completed sleep cycle without gaps. Capacity envelopes are derived from
    /// this complete series.
    pub candidate_capacity: Vec<CapacityObservation>,
    pub grouped_mm_parity: KernelParityEvidence,
    pub pytorch_parity: KernelParityEvidence,
    pub exact_resume: ExactResumeEvidence,
    /// Host-derived receipt for the exact pinned worker request and its raw
    /// observations. This field is mandatory: handwritten aggregate JSON is
    /// never valid promotion evidence.
    pub execution: ResourceExecutionReceipt,
}

pub const RESOURCE_COMPARISON_VERSION: u32 = 2;
pub const ACCEPTANCE_POLICY_VERSION: u32 = 2;
pub const RESOURCE_EXECUTION_PROTOCOL_VERSION: u32 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceExecutionReceipt {
    pub protocol_version: u32,
    /// Exact executable identity verified by the host, never copied from the
    /// worker response.
    pub evaluator_sha256: String,
    /// Canonical SHA-256 of the strict request sent to the worker.
    pub request_sha256: String,
    /// Canonical SHA-256 of the raw observations retained below.
    pub observations_sha256: String,
    /// Canonical identities of the exact targets embedded in the selected
    /// benchmark run.
    pub baseline_target_sha256: String,
    pub candidate_target_sha256: String,
    /// Raw SHA-256 of the content-addressed acceptance-policy artifact.
    pub policy_sha256: String,
    /// Exact portable argument vector used with the pinned evaluator binary.
    pub evaluator_arguments: Vec<String>,
    /// Safe relative directories beneath the resource-evidence vault in which
    /// the worker was permitted to place exact-resume artifacts.
    pub approved_artifact_roots: Vec<PathBuf>,
}

impl ResourceExecutionReceipt {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.protocol_version == RESOURCE_EXECUTION_PROTOCOL_VERSION,
            "unsupported resource execution protocol version {}",
            self.protocol_version
        );
        for (name, value) in [
            ("resource evaluator", self.evaluator_sha256.as_str()),
            ("resource request", self.request_sha256.as_str()),
            ("resource observations", self.observations_sha256.as_str()),
            ("baseline target", self.baseline_target_sha256.as_str()),
            ("candidate target", self.candidate_target_sha256.as_str()),
        ] {
            validate_prefixed_sha256(value).map_err(|error| anyhow::anyhow!("{name}: {error}"))?;
        }
        validate_sha256(&self.policy_sha256)
            .map_err(|error| anyhow::anyhow!("resource policy: {error}"))?;
        ensure!(
            self.evaluator_arguments.len() <= 64
                && self
                    .evaluator_arguments
                    .iter()
                    .all(|argument| argument.len() <= 4096 && !argument.contains('\0')),
            "resource evaluator arguments exceed protocol limits"
        );
        ensure!(
            !self.approved_artifact_roots.is_empty(),
            "resource execution has no approved artifact root"
        );
        for root in &self.approved_artifact_roots {
            validate_safe_relative_path(root, "resource execution artifact root")?;
        }
        let unique = self.approved_artifact_roots.iter().collect::<BTreeSet<_>>();
        ensure!(
            unique.len() == self.approved_artifact_roots.len(),
            "resource execution repeats an approved artifact root"
        );
        ensure!(
            self.approved_artifact_roots
                .iter()
                .enumerate()
                .all(|(index, root)| {
                    self.approved_artifact_roots[index + 1..]
                        .iter()
                        .all(|other| !root.starts_with(other) && !other.starts_with(root))
                }),
            "resource execution artifact roots must not overlap"
        );
        Ok(())
    }
}

fn validate_safe_relative_path(path: &Path, name: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{name} is empty");
    ensure!(!path.is_absolute(), "{name} must be relative");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{name} must not contain prefixes, `.` or `..`"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WakeMeasurement {
    pub tokens: u64,
    pub elapsed_seconds: f64,
    pub request_latency_ms: Vec<f64>,
}

impl WakeMeasurement {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(self.tokens > 0, "{name} wake measurement has no tokens");
        ensure!(
            self.elapsed_seconds.is_finite() && self.elapsed_seconds > 0.0,
            "{name} wake measurement has invalid elapsed seconds"
        );
        ensure!(
            !self.request_latency_ms.is_empty(),
            "{name} wake measurement has no request latencies"
        );
        ensure!(
            self.request_latency_ms
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "{name} wake measurement has an invalid request latency"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedWakeTrial {
    /// Zero-based, contiguous trial ordinal.
    pub trial: u64,
    pub baseline: WakeMeasurement,
    pub candidate: WakeMeasurement,
}

impl PairedWakeTrial {
    fn validate(&self) -> Result<()> {
        self.baseline.validate("baseline")?;
        self.candidate.validate("candidate")?;
        ensure!(
            self.baseline.tokens == self.candidate.tokens,
            "wake trial {} did not use the same token workload",
            self.trial
        );
        ensure!(
            self.baseline.request_latency_ms.len() == self.candidate.request_latency_ms.len(),
            "wake trial {} did not use the same request workload",
            self.trial
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityObservation {
    /// Zero denotes the initial wake model; positive values denote the state
    /// after that many completed sleep cycles.
    pub completed_sleep_cycles: u64,
    pub routed_active_parameters: u64,
    pub stored_parameters: u64,
    pub stored_bytes: u64,
}

impl CapacityObservation {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.routed_active_parameters > 0
                && self.stored_parameters > 0
                && self.stored_bytes > 0,
            "capacity observation {} contains an empty measurement",
            self.completed_sleep_cycles
        );
        ensure!(
            self.routed_active_parameters <= self.stored_parameters,
            "capacity observation {} routes more parameters than it stores",
            self.completed_sleep_cycles
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParitySample {
    pub reference: f64,
    pub candidate: f64,
}

/// Raw outputs produced from a fixed reference fixture. Promotion recomputes
/// both error bounds; tolerances never come from this evidence.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParityEvidence {
    pub fixture_sha256: String,
    pub samples: Vec<KernelParitySample>,
}

impl KernelParityEvidence {
    fn validate(&self, name: &str) -> Result<()> {
        validate_sha256(&self.fixture_sha256)
            .map_err(|error| anyhow::anyhow!("{name} fixture: {error}"))?;
        ensure!(
            !self.samples.is_empty(),
            "{name} parity evaluated no values"
        );
        ensure!(
            self.samples
                .iter()
                .all(|sample| { sample.reference.is_finite() && sample.candidate.is_finite() }),
            "{name} parity contains a non-finite sample"
        );
        Ok(())
    }

    fn maximum_errors(&self) -> (f64, f64) {
        self.samples.iter().fold(
            (0.0_f64, 0.0_f64),
            |(maximum_absolute, maximum_relative), sample| {
                let absolute = (sample.candidate - sample.reference).abs();
                let relative = if sample.reference == 0.0 {
                    if absolute == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    absolute / sample.reference.abs()
                };
                (
                    maximum_absolute.max(absolute),
                    maximum_relative.max(relative),
                )
            },
        )
    }
}

/// One immutable file used by the exact-resume verifier. Paths resolve against
/// the resource-comparison artifact; the digest is verified from a stable file
/// handle before any equality decision is made.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactResumeArtifact {
    pub path: PathBuf,
    pub sha256: String,
}

impl ExactResumeArtifact {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "{name} artifact path is empty"
        );
        validate_sha256(&self.sha256).map_err(|error| anyhow::anyhow!("{name}: {error}"))
    }
}

/// Immutable artifacts proving that an interrupted run and an uninterrupted
/// reference converged to byte-identical state and semantic progress from the
/// same checkpoint. Promotion verifies every referenced file; digest-shaped
/// strings without those files are never promotable.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactResumeEvidence {
    pub interrupted_checkpoint: ExactResumeArtifact,
    pub uninterrupted_final_state: ExactResumeArtifact,
    pub resumed_final_state: ExactResumeArtifact,
    pub uninterrupted_metrics: ExactResumeArtifact,
    pub resumed_metrics: ExactResumeArtifact,
    pub interruption_step: u64,
    pub resumed_from_step: u64,
}

impl ExactResumeEvidence {
    fn validate(&self) -> Result<()> {
        for (name, artifact) in [
            ("interrupted checkpoint", &self.interrupted_checkpoint),
            ("uninterrupted final state", &self.uninterrupted_final_state),
            ("resumed final state", &self.resumed_final_state),
            ("uninterrupted metrics", &self.uninterrupted_metrics),
            ("resumed metrics", &self.resumed_metrics),
        ] {
            artifact.validate(name)?;
        }
        ensure!(
            self.interruption_step > 0,
            "resume evidence must interrupt after at least one step"
        );
        Ok(())
    }
}

impl ResourceComparison {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == RESOURCE_COMPARISON_VERSION,
            "unsupported resource-comparison version {}",
            self.version
        );
        ensure!(!self.baseline_id.trim().is_empty(), "baseline_id is empty");
        ensure!(
            !self.candidate_id.trim().is_empty(),
            "candidate_id is empty"
        );
        ensure!(
            self.baseline_id != self.candidate_id,
            "baseline and candidate identities must differ"
        );
        ensure!(
            !self.strongest_baseline_id.trim().is_empty(),
            "strongest_baseline_id is empty"
        );
        validate_sha256(&self.benchmark_run_sha256)
            .map_err(|error| anyhow::anyhow!("benchmark run: {error}"))?;
        ensure!(
            !self.measurement_evaluator_id.trim().is_empty(),
            "resource measurement evaluator id is empty"
        );
        validate_prefixed_sha256(&self.measurement_evaluator_version)
            .map_err(|error| anyhow::anyhow!("resource measurement evaluator: {error}"))?;
        ensure!(
            !self.wake_trials.is_empty(),
            "resource evidence has no wake trials"
        );
        for (index, trial) in self.wake_trials.iter().enumerate() {
            ensure!(
                trial.trial == index as u64,
                "wake trial ordinals must be zero-based and contiguous; expected {index}, got {}",
                trial.trial
            );
            trial.validate()?;
        }
        ensure!(
            self.candidate_capacity.len() >= 2,
            "wake capacity evidence must contain cycle zero and at least one completed sleep cycle"
        );
        for (index, observation) in self.candidate_capacity.iter().enumerate() {
            ensure!(
                observation.completed_sleep_cycles == index as u64,
                "capacity observations must cover cycle zero and every completed cycle; expected {index}, got {}",
                observation.completed_sleep_cycles
            );
            observation.validate()?;
        }
        self.grouped_mm_parity.validate("grouped-mm")?;
        self.pytorch_parity.validate("PyTorch")?;
        self.exact_resume.validate()?;
        self.execution.validate()?;
        ensure!(
            self.execution.evaluator_sha256 == self.measurement_evaluator_version,
            "resource execution worker differs from the measurement evaluator"
        );
        ensure!(
            self.execution.observations_sha256 == self.observations_sha256()?,
            "resource execution receipt does not authenticate its raw observations"
        );
        Ok(())
    }

    pub fn observations_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        struct ExactResumeIdentity<'a> {
            interrupted_checkpoint_sha256: &'a str,
            uninterrupted_final_state_sha256: &'a str,
            resumed_final_state_sha256: &'a str,
            uninterrupted_metrics_sha256: &'a str,
            resumed_metrics_sha256: &'a str,
            interruption_step: u64,
            resumed_from_step: u64,
        }

        #[derive(Serialize)]
        struct RawObservations<'a> {
            wake_trials: &'a [PairedWakeTrial],
            candidate_capacity: &'a [CapacityObservation],
            grouped_mm_samples: &'a [KernelParitySample],
            pytorch_samples: &'a [KernelParitySample],
            exact_resume: ExactResumeIdentity<'a>,
        }

        let exact = &self.exact_resume;
        let bytes = serde_json::to_vec(&RawObservations {
            wake_trials: &self.wake_trials,
            candidate_capacity: &self.candidate_capacity,
            grouped_mm_samples: &self.grouped_mm_parity.samples,
            pytorch_samples: &self.pytorch_parity.samples,
            exact_resume: ExactResumeIdentity {
                interrupted_checkpoint_sha256: &exact.interrupted_checkpoint.sha256,
                uninterrupted_final_state_sha256: &exact.uninterrupted_final_state.sha256,
                resumed_final_state_sha256: &exact.resumed_final_state.sha256,
                uninterrupted_metrics_sha256: &exact.uninterrupted_metrics.sha256,
                resumed_metrics_sha256: &exact.resumed_metrics.sha256,
                interruption_step: exact.interruption_step,
                resumed_from_step: exact.resumed_from_step,
            },
        })?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }

    pub(crate) fn derived_measurements(&self) -> Result<DerivedResourceMeasurements> {
        self.validate()?;
        let baseline_tokens = self.wake_trials.iter().try_fold(0_u128, |total, trial| {
            total
                .checked_add(u128::from(trial.baseline.tokens))
                .ok_or_else(|| anyhow::anyhow!("baseline wake token total overflowed"))
        })?;
        let candidate_tokens = self.wake_trials.iter().try_fold(0_u128, |total, trial| {
            total
                .checked_add(u128::from(trial.candidate.tokens))
                .ok_or_else(|| anyhow::anyhow!("candidate wake token total overflowed"))
        })?;
        let baseline_elapsed = self
            .wake_trials
            .iter()
            .map(|trial| trial.baseline.elapsed_seconds)
            .sum::<f64>();
        let candidate_elapsed = self
            .wake_trials
            .iter()
            .map(|trial| trial.candidate.elapsed_seconds)
            .sum::<f64>();
        ensure!(
            baseline_elapsed.is_finite() && candidate_elapsed.is_finite(),
            "wake elapsed-time aggregate overflowed"
        );
        let mut baseline_latencies = self
            .wake_trials
            .iter()
            .flat_map(|trial| trial.baseline.request_latency_ms.iter().copied())
            .collect::<Vec<_>>();
        let mut candidate_latencies = self
            .wake_trials
            .iter()
            .flat_map(|trial| trial.candidate.request_latency_ms.iter().copied())
            .collect::<Vec<_>>();
        let initial = self
            .candidate_capacity
            .first()
            .expect("resource validation requires capacity observations");
        let minimum_routed_active_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.routed_active_parameters)
            .min()
            .expect("resource validation requires capacity observations");
        let maximum_routed_active_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.routed_active_parameters)
            .max()
            .expect("resource validation requires capacity observations");
        let maximum_stored_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_parameters)
            .max()
            .expect("resource validation requires capacity observations");
        let minimum_stored_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_parameters)
            .min()
            .expect("resource validation requires capacity observations");
        let maximum_stored_bytes = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_bytes)
            .max()
            .expect("resource validation requires capacity observations");
        let minimum_stored_bytes = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_bytes)
            .min()
            .expect("resource validation requires capacity observations");
        Ok(DerivedResourceMeasurements {
            wake_trials: self.wake_trials.len(),
            wake_latency_samples: baseline_latencies.len(),
            baseline_wake_tokens_per_second: baseline_tokens as f64 / baseline_elapsed,
            candidate_wake_tokens_per_second: candidate_tokens as f64 / candidate_elapsed,
            baseline_wake_p95_ms: nearest_rank_p95(&mut baseline_latencies),
            candidate_wake_p95_ms: nearest_rank_p95(&mut candidate_latencies),
            initial_routed_active_parameters: initial.routed_active_parameters,
            initial_stored_parameters: initial.stored_parameters,
            initial_stored_bytes: initial.stored_bytes,
            completed_sleep_cycles_observed: self.candidate_capacity.len() - 1,
            minimum_routed_active_parameters,
            maximum_routed_active_parameters,
            minimum_stored_parameters,
            maximum_stored_parameters,
            minimum_stored_bytes,
            maximum_stored_bytes,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DerivedResourceMeasurements {
    pub wake_trials: usize,
    pub wake_latency_samples: usize,
    pub baseline_wake_tokens_per_second: f64,
    pub candidate_wake_tokens_per_second: f64,
    pub baseline_wake_p95_ms: f64,
    pub candidate_wake_p95_ms: f64,
    pub initial_routed_active_parameters: u64,
    pub initial_stored_parameters: u64,
    pub initial_stored_bytes: u64,
    pub completed_sleep_cycles_observed: usize,
    pub minimum_routed_active_parameters: u64,
    pub maximum_routed_active_parameters: u64,
    pub minimum_stored_parameters: u64,
    pub maximum_stored_parameters: u64,
    pub minimum_stored_bytes: u64,
    pub maximum_stored_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParityPolicy {
    pub fixture_sha256: String,
    pub minimum_samples: usize,
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
}

impl KernelParityPolicy {
    fn validate(&self, name: &str) -> Result<()> {
        validate_sha256(&self.fixture_sha256)
            .map_err(|error| anyhow::anyhow!("{name} fixture: {error}"))?;
        ensure!(
            self.minimum_samples > 0,
            "{name} policy requires no parity samples"
        );
        ensure!(
            self.maximum_absolute_error.is_finite()
                && self.maximum_absolute_error >= 0.0
                && self.maximum_relative_error.is_finite()
                && self.maximum_relative_error >= 0.0,
            "{name} policy contains an invalid error ceiling"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptancePolicy {
    pub version: u32,
    pub minimum_paired_seeds: usize,
    pub maximum_anchor_regression: f64,
    pub stable_anchor_catalog_ids: BTreeSet<String>,
    pub resource_evaluator_id: String,
    pub resource_evaluator_version: String,
    pub minimum_wake_trials: usize,
    pub minimum_wake_latency_samples: usize,
    pub minimum_wake_throughput_ratio: f64,
    pub maximum_wake_latency_ratio: f64,
    pub grouped_mm_parity: KernelParityPolicy,
    pub pytorch_parity: KernelParityPolicy,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self {
            version: ACCEPTANCE_POLICY_VERSION,
            minimum_paired_seeds: 3,
            maximum_anchor_regression: 0.01,
            stable_anchor_catalog_ids: [
                "pretraining_causal".to_owned(),
                "synthetic_retention_smoke".to_owned(),
                "clinc_continual_learning".to_owned(),
                "banking77_continual_learning".to_owned(),
            ]
            .into_iter()
            .collect(),
            resource_evaluator_id: "hermes-resource-evaluator".to_owned(),
            resource_evaluator_version: format!("sha256:{}", "c".repeat(64)),
            minimum_wake_trials: 3,
            minimum_wake_latency_samples: 3,
            minimum_wake_throughput_ratio: 0.95,
            maximum_wake_latency_ratio: 1.05,
            grouped_mm_parity: KernelParityPolicy {
                fixture_sha256: "a".repeat(64),
                minimum_samples: 1024,
                maximum_absolute_error: 2e-5,
                maximum_relative_error: 2e-4,
            },
            pytorch_parity: KernelParityPolicy {
                fixture_sha256: "b".repeat(64),
                minimum_samples: 1024,
                maximum_absolute_error: 2e-5,
                maximum_relative_error: 2e-4,
            },
        }
    }
}

impl AcceptancePolicy {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == ACCEPTANCE_POLICY_VERSION,
            "unsupported acceptance-policy version {}",
            self.version
        );
        ensure!(
            self.minimum_paired_seeds >= 3,
            "promotion requires at least three paired seeds"
        );
        ensure!(
            self.maximum_anchor_regression.is_finite() && self.maximum_anchor_regression >= 0.0,
            "maximum anchor regression must be finite and non-negative"
        );
        ensure!(
            !self.stable_anchor_catalog_ids.is_empty()
                && self
                    .stable_anchor_catalog_ids
                    .iter()
                    .all(|id| !id.trim().is_empty()),
            "acceptance policy must prescribe at least one non-empty stable-anchor catalog id"
        );
        ensure!(
            !self.resource_evaluator_id.trim().is_empty(),
            "resource evaluator id is empty"
        );
        validate_prefixed_sha256(&self.resource_evaluator_version)
            .map_err(|error| anyhow::anyhow!("resource evaluator: {error}"))?;
        ensure!(
            self.minimum_wake_trials > 0 && self.minimum_wake_latency_samples > 0,
            "wake measurement minima must be positive"
        );
        ensure!(
            self.minimum_wake_throughput_ratio.is_finite()
                && (0.0..=1.0).contains(&self.minimum_wake_throughput_ratio),
            "minimum wake throughput ratio must be in [0, 1]"
        );
        ensure!(
            self.maximum_wake_latency_ratio.is_finite() && self.maximum_wake_latency_ratio >= 1.0,
            "maximum wake latency ratio must be at least one"
        );
        self.grouped_mm_parity.validate("grouped-mm")?;
        self.pytorch_parity.validate("PyTorch")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseGate {
    pub case_id: String,
    pub mean_delta: f64,
    pub lower_confidence_bound: f64,
    pub allowed_anchor_regression: Option<f64>,
    pub passed: bool,
}

/// Sealed examples, ids, scores, and per-case deltas are deliberately absent
/// from the promotion report. Only this aggregate crosses the promotion
/// boundary; the verified benchmark-run evidence remains auditable.
#[derive(Clone, Debug, Serialize)]
pub struct SealedGate {
    pub case_count: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromotionReport {
    pub accepted: bool,
    /// Public cases only.  Sealed ids, metrics, and deltas are never retained
    /// in the promotion report.
    pub cases: Vec<CaseGate>,
    pub sealed: SealedGate,
    pub resource_gates: BTreeMap<String, bool>,
}

#[derive(Clone, Debug)]
pub(crate) struct VerifiedPromotionContext<'a> {
    pub benchmark_run_sha256: &'a str,
    pub strongest_baseline_id: &'a str,
    pub baseline_id: &'a str,
    pub candidate_id: &'a str,
    pub capacity_matched: bool,
    pub active_capacity_matched: bool,
    pub fixed_gpu_hour_measured: bool,
    pub baseline_stored_parameters: u64,
    pub candidate_stored_parameters: u64,
    pub baseline_stored_bytes: u64,
    pub candidate_stored_bytes: u64,
    pub candidate_routed_active_parameters: u64,
    pub content_addressed_evidence: bool,
    pub executed_resource_evidence: bool,
    pub exact_resume_verified: bool,
}

/// Evaluate raw inputs for diagnostics.
///
/// Raw caller-supplied JSON is intentionally never promotable because it has
/// not crossed the content-addressed benchmark boundary.  Production
/// promotion must call [`crate::benchmark::evaluate_verified_promotion`].
pub fn evaluate_candidate(
    suite: &AcceptanceSuite,
    results: &[CaseResult],
    resources: &ResourceComparison,
    policy: &AcceptancePolicy,
) -> Result<PromotionReport> {
    let measurements = resources.derived_measurements()?;
    let context = VerifiedPromotionContext {
        benchmark_run_sha256: &resources.benchmark_run_sha256,
        strongest_baseline_id: &resources.strongest_baseline_id,
        baseline_id: &resources.baseline_id,
        candidate_id: &resources.candidate_id,
        capacity_matched: false,
        active_capacity_matched: false,
        fixed_gpu_hour_measured: false,
        baseline_stored_parameters: measurements.initial_stored_parameters,
        candidate_stored_parameters: measurements.initial_stored_parameters,
        baseline_stored_bytes: measurements.initial_stored_bytes,
        candidate_stored_bytes: measurements.initial_stored_bytes,
        candidate_routed_active_parameters: measurements.initial_routed_active_parameters,
        content_addressed_evidence: false,
        executed_resource_evidence: false,
        exact_resume_verified: false,
    };
    evaluate_with_verified_context(suite, results, resources, policy, &context)
}

pub(crate) fn evaluate_with_verified_context(
    suite: &AcceptanceSuite,
    results: &[CaseResult],
    resources: &ResourceComparison,
    policy: &AcceptancePolicy,
    context: &VerifiedPromotionContext<'_>,
) -> Result<PromotionReport> {
    suite.validate()?;
    policy.validate()?;
    let measurements = resources.derived_measurements()?;

    ensure!(
        context.benchmark_run_sha256 == resources.benchmark_run_sha256,
        "resource evidence addresses benchmark run {}, not {}",
        resources.benchmark_run_sha256,
        context.benchmark_run_sha256
    );
    ensure!(
        context.baseline_id == resources.baseline_id
            && context.candidate_id == resources.candidate_id,
        "resource evidence target identities do not match the benchmark run"
    );
    ensure!(
        resources.measurement_evaluator_id == policy.resource_evaluator_id
            && resources.measurement_evaluator_version == policy.resource_evaluator_version,
        "resource evidence was not produced by the policy-pinned evaluator"
    );
    ensure!(
        resources
            .grouped_mm_parity
            .fixture_sha256
            .eq_ignore_ascii_case(&policy.grouped_mm_parity.fixture_sha256),
        "grouped-mm parity evidence uses a fixture not pinned by policy"
    );
    ensure!(
        resources
            .pytorch_parity
            .fixture_sha256
            .eq_ignore_ascii_case(&policy.pytorch_parity.fixture_sha256),
        "PyTorch parity evidence uses a fixture not pinned by policy"
    );
    ensure!(
        measurements.wake_trials >= policy.minimum_wake_trials,
        "resource evidence has {} wake trials; policy requires {}",
        measurements.wake_trials,
        policy.minimum_wake_trials
    );
    ensure!(
        measurements.wake_latency_samples >= policy.minimum_wake_latency_samples,
        "resource evidence has {} paired wake latency samples; policy requires {}",
        measurements.wake_latency_samples,
        policy.minimum_wake_latency_samples
    );
    ensure!(
        resources.grouped_mm_parity.samples.len() >= policy.grouped_mm_parity.minimum_samples,
        "grouped-mm parity evidence has {} samples; policy requires {}",
        resources.grouped_mm_parity.samples.len(),
        policy.grouped_mm_parity.minimum_samples
    );
    ensure!(
        resources.pytorch_parity.samples.len() >= policy.pytorch_parity.minimum_samples,
        "PyTorch parity evidence has {} samples; policy requires {}",
        resources.pytorch_parity.samples.len(),
        policy.pytorch_parity.minimum_samples
    );

    let suite_catalog_ids = suite
        .cases
        .iter()
        .map(|case| case.catalog_id.as_str())
        .collect::<BTreeSet<_>>();
    for catalog_id in &policy.stable_anchor_catalog_ids {
        ensure!(
            suite_catalog_ids.contains(catalog_id.as_str()),
            "policy-prescribed stable-anchor catalog `{catalog_id}` is absent from the suite"
        );
    }

    let mut result_by_id = BTreeMap::new();
    for result in results {
        ensure!(
            result_by_id
                .insert(result.case_id.as_str(), result)
                .is_none(),
            "duplicate acceptance result `{}`",
            result.case_id
        );
    }
    ensure!(
        result_by_id.len() == suite.cases.len(),
        "acceptance results do not exactly match the suite"
    );
    let mut cases = Vec::with_capacity(suite.cases.len());
    let mut sealed_case_count = 0usize;
    let mut sealed_passed = true;
    let mut expected_seeds: Option<Vec<u64>> = None;
    for case in &suite.cases {
        let policy_anchor = policy
            .stable_anchor_catalog_ids
            .contains(case.catalog_id.as_str());
        ensure!(
            case.stable_anchor == policy_anchor,
            "acceptance case `{}` stable_anchor={} conflicts with policy for catalog `{}`",
            case.id,
            case.stable_anchor,
            case.catalog_id
        );
        let result = result_by_id
            .get(case.id.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing acceptance result `{}`", case.id))?;
        ensure!(
            result.pairs.len() >= policy.minimum_paired_seeds,
            "acceptance case `{}` has {} paired seeds; need {}",
            case.id,
            result.pairs.len(),
            policy.minimum_paired_seeds
        );
        let unique_seeds = result
            .pairs
            .iter()
            .map(|pair| pair.seed)
            .collect::<BTreeSet<_>>();
        ensure!(
            unique_seeds.len() == result.pairs.len(),
            "acceptance case `{}` repeats a seed",
            case.id
        );
        let seeds = result
            .pairs
            .iter()
            .map(|pair| pair.seed)
            .collect::<Vec<_>>();
        ensure!(
            seeds.windows(2).all(|pair| pair[0] < pair[1]),
            "acceptance case `{}` seeds are not strictly increasing",
            case.id
        );
        if let Some(expected) = &expected_seeds {
            ensure!(
                &seeds == expected,
                "acceptance case `{}` does not use the common paired seeds",
                case.id
            );
        } else {
            expected_seeds = Some(seeds);
        }
        ensure!(
            result
                .pairs
                .iter()
                .all(|pair| pair.baseline.is_finite() && pair.candidate.is_finite()),
            "acceptance case `{}` contains non-finite metrics",
            case.id
        );
        let deltas = result
            .pairs
            .iter()
            .map(|pair| pair.candidate - pair.baseline)
            .collect::<Vec<_>>();
        let (mean_delta, lower_confidence_bound) = paired_lower_95(&deltas);
        let allowed_anchor_regression = case.stable_anchor.then(|| {
            policy
                .maximum_anchor_regression
                .max(sample_standard_deviation(
                    &result
                        .pairs
                        .iter()
                        .map(|pair| pair.baseline)
                        .collect::<Vec<_>>(),
                ))
        });
        let passed = if case.stable_anchor {
            mean_delta >= -allowed_anchor_regression.unwrap()
        } else {
            lower_confidence_bound > 0.0
        };
        if case.visibility == SuiteVisibility::Sealed {
            sealed_case_count += 1;
            sealed_passed &= passed;
        } else {
            cases.push(CaseGate {
                case_id: case.id.clone(),
                mean_delta,
                lower_confidence_bound,
                allowed_anchor_regression,
                passed,
            });
        }
    }

    let throughput_ratio = measurements.candidate_wake_tokens_per_second
        / measurements.baseline_wake_tokens_per_second;
    let latency_ratio = measurements.candidate_wake_p95_ms / measurements.baseline_wake_p95_ms;
    let grouped_mm_errors = resources.grouped_mm_parity.maximum_errors();
    let pytorch_errors = resources.pytorch_parity.maximum_errors();
    let resource_gates = BTreeMap::from([
        (
            "strongest_matched_baseline".into(),
            context.baseline_id == context.strongest_baseline_id,
        ),
        ("capacity_matched".into(), context.capacity_matched),
        (
            "active_capacity_matched".into(),
            context.active_capacity_matched,
        ),
        (
            "fixed_gpu_hour_measured".into(),
            context.fixed_gpu_hour_measured,
        ),
        (
            "content_addressed_evidence".into(),
            context.content_addressed_evidence,
        ),
        (
            "executed_resource_evidence".into(),
            context.executed_resource_evidence,
        ),
        ("exact_resume".into(), context.exact_resume_verified),
        (
            "constant_active_parameters".into(),
            measurements.completed_sleep_cycles_observed > 0
                && measurements.minimum_routed_active_parameters
                    == measurements.initial_routed_active_parameters
                && measurements.maximum_routed_active_parameters
                    == measurements.initial_routed_active_parameters
                && measurements.initial_routed_active_parameters
                    == context.candidate_routed_active_parameters,
        ),
        (
            "bounded_stored_parameters".into(),
            measurements.initial_stored_parameters == context.candidate_stored_parameters
                && measurements.minimum_stored_parameters == measurements.initial_stored_parameters
                && measurements.maximum_stored_parameters == measurements.initial_stored_parameters
                && context.candidate_stored_parameters <= context.baseline_stored_parameters,
        ),
        (
            "bounded_stored_bytes".into(),
            measurements.initial_stored_bytes == context.candidate_stored_bytes
                && measurements.minimum_stored_bytes == measurements.initial_stored_bytes
                && measurements.maximum_stored_bytes == measurements.initial_stored_bytes
                && context.candidate_stored_bytes <= context.baseline_stored_bytes,
        ),
        (
            "grouped_mm_parity".into(),
            grouped_mm_errors.0 <= policy.grouped_mm_parity.maximum_absolute_error
                && grouped_mm_errors.1 <= policy.grouped_mm_parity.maximum_relative_error,
        ),
        (
            "pytorch_parity".into(),
            pytorch_errors.0 <= policy.pytorch_parity.maximum_absolute_error
                && pytorch_errors.1 <= policy.pytorch_parity.maximum_relative_error,
        ),
        (
            "wake_throughput".into(),
            throughput_ratio >= policy.minimum_wake_throughput_ratio,
        ),
        (
            "wake_latency".into(),
            latency_ratio <= policy.maximum_wake_latency_ratio,
        ),
    ]);
    let accepted = cases.iter().all(|case| case.passed)
        && sealed_passed
        && resource_gates.values().all(|gate| *gate);
    Ok(PromotionReport {
        accepted,
        cases,
        sealed: SealedGate {
            case_count: sealed_case_count,
            passed: sealed_passed,
        },
        resource_gates,
    })
}

fn validate_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "SHA-256 must contain exactly 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn validate_prefixed_sha256(value: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .ok_or_else(|| anyhow::anyhow!("identity must use the `sha256:<hex>` form"))?;
    validate_sha256(digest)
}

fn nearest_rank_p95(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let rank = (95 * values.len()).div_ceil(100);
    values[rank.saturating_sub(1)]
}

fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

fn paired_lower_95(deltas: &[f64]) -> (f64, f64) {
    let count = deltas.len();
    let mean = deltas.iter().sum::<f64>() / count as f64;
    if count == 1 {
        return (mean, f64::NEG_INFINITY);
    }
    let variance = deltas
        .iter()
        .map(|delta| (delta - mean).powi(2))
        .sum::<f64>()
        / (count - 1) as f64;
    let standard_error = (variance / count as f64).sqrt();
    (mean, mean - t_critical_95(count - 1) * standard_error)
}

/// Two-sided 95% Student-t critical values; the lower bound therefore uses
/// the conservative half-width expected by paired experiment reports.
fn t_critical_95(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    TABLE.get(df.saturating_sub(1)).copied().unwrap_or(1.96)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite() -> AcceptanceSuite {
        AcceptanceSuite {
            version: ACCEPTANCE_SUITE_VERSION,
            cases: vec![
                BenchmarkCase {
                    id: "quality".into(),
                    catalog_id: "quality".into(),
                    family: BenchmarkFamily::Reasoning,
                    visibility: SuiteVisibility::Public,
                    metric: "accuracy".into(),
                    stable_anchor: false,
                },
                BenchmarkCase {
                    id: "retention".into(),
                    catalog_id: "retention".into(),
                    family: BenchmarkFamily::ContinualRetention,
                    visibility: SuiteVisibility::Sealed,
                    metric: "accuracy".into(),
                    stable_anchor: true,
                },
            ],
        }
    }

    fn policy() -> AcceptancePolicy {
        AcceptancePolicy {
            stable_anchor_catalog_ids: ["retention".to_owned()].into_iter().collect(),
            ..AcceptancePolicy::default()
        }
    }

    fn resources() -> ResourceComparison {
        let parity_samples = || {
            (0..1024)
                .map(|_| KernelParitySample {
                    reference: 1.0,
                    candidate: 1.0 + 1e-5,
                })
                .collect()
        };
        let mut resources = ResourceComparison {
            version: RESOURCE_COMPARISON_VERSION,
            baseline_id: "static-moe".into(),
            candidate_id: "sleep-candidate".into(),
            benchmark_run_sha256: "1".repeat(64),
            strongest_baseline_id: "static-moe".into(),
            measurement_evaluator_id: "hermes-resource-evaluator".into(),
            measurement_evaluator_version: format!("sha256:{}", "c".repeat(64)),
            wake_trials: (0..3)
                .map(|trial| PairedWakeTrial {
                    trial,
                    baseline: WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                    candidate: WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 1_000.0 / 96.0,
                        request_latency_ms: vec![10.4],
                    },
                })
                .collect(),
            candidate_capacity: (0..=4)
                .map(|completed_sleep_cycles| CapacityObservation {
                    completed_sleep_cycles,
                    routed_active_parameters: 80,
                    stored_parameters: 100,
                    stored_bytes: 1_000,
                })
                .collect(),
            grouped_mm_parity: KernelParityEvidence {
                fixture_sha256: "a".repeat(64),
                samples: parity_samples(),
            },
            pytorch_parity: KernelParityEvidence {
                fixture_sha256: "b".repeat(64),
                samples: parity_samples(),
            },
            exact_resume: ExactResumeEvidence {
                interrupted_checkpoint: ExactResumeArtifact {
                    path: "interrupted/generation-manifest.json".into(),
                    sha256: "4".repeat(64),
                },
                uninterrupted_final_state: ExactResumeArtifact {
                    path: "uninterrupted/generation-manifest.json".into(),
                    sha256: "5".repeat(64),
                },
                resumed_final_state: ExactResumeArtifact {
                    path: "resumed/generation-manifest.json".into(),
                    sha256: "5".repeat(64),
                },
                uninterrupted_metrics: ExactResumeArtifact {
                    path: "uninterrupted/metrics.jsonl".into(),
                    sha256: "6".repeat(64),
                },
                resumed_metrics: ExactResumeArtifact {
                    path: "resumed/metrics.jsonl".into(),
                    sha256: "7".repeat(64),
                },
                interruption_step: 17,
                resumed_from_step: 17,
            },
            execution: ResourceExecutionReceipt {
                protocol_version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
                evaluator_sha256: format!("sha256:{}", "c".repeat(64)),
                request_sha256: format!("sha256:{}", "8".repeat(64)),
                observations_sha256: format!("sha256:{}", "9".repeat(64)),
                baseline_target_sha256: format!("sha256:{}", "a".repeat(64)),
                candidate_target_sha256: format!("sha256:{}", "b".repeat(64)),
                policy_sha256: "d".repeat(64),
                evaluator_arguments: Vec::new(),
                approved_artifact_roots: vec!["artifacts".into()],
            },
        };
        resources.execution.observations_sha256 = resources.observations_sha256().unwrap();
        resources
    }

    fn verified_context<'a>(resources: &'a ResourceComparison) -> VerifiedPromotionContext<'a> {
        VerifiedPromotionContext {
            benchmark_run_sha256: &resources.benchmark_run_sha256,
            strongest_baseline_id: &resources.strongest_baseline_id,
            baseline_id: &resources.baseline_id,
            candidate_id: &resources.candidate_id,
            capacity_matched: true,
            active_capacity_matched: true,
            fixed_gpu_hour_measured: true,
            baseline_stored_parameters: 100,
            candidate_stored_parameters: 100,
            baseline_stored_bytes: 1_000,
            candidate_stored_bytes: 1_000,
            candidate_routed_active_parameters: 80,
            content_addressed_evidence: true,
            exact_resume_verified: true,
            executed_resource_evidence: true,
        }
    }

    fn reseal(resources: &mut ResourceComparison) {
        resources.execution.observations_sha256 = resources.observations_sha256().unwrap();
    }

    #[test]
    fn promotion_requires_confident_gain_and_stable_anchors() {
        let results = vec![
            CaseResult {
                case_id: "quality".into(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.4,
                        candidate: 0.5,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.4,
                        candidate: 0.51,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.4,
                        candidate: 0.49,
                    },
                ],
            },
            CaseResult {
                case_id: "retention".into(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.8,
                        candidate: 0.795,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.8,
                        candidate: 0.796,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.8,
                        candidate: 0.794,
                    },
                ],
            },
        ];
        let resources = resources();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(report.accepted);
        assert!(report.cases.iter().all(|case| case.passed));
        assert_eq!(report.sealed.case_count, 1);
        assert!(report.sealed.passed);
        let report_json = serde_json::to_string(&report).unwrap();
        assert!(!report_json.contains("retention"));
    }

    #[test]
    fn active_compute_growth_rejects_candidate() {
        let mut resources = resources();
        resources
            .candidate_capacity
            .last_mut()
            .unwrap()
            .routed_active_parameters += 1;
        reseal(&mut resources);
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.accepted);
        assert!(!report.resource_gates["constant_active_parameters"]);
    }

    #[test]
    fn first_activation_compute_growth_rejects_candidate() {
        let mut resources = resources();
        resources.candidate_capacity[0].routed_active_parameters -= 1;
        reseal(&mut resources);
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.accepted);
        assert!(!report.resource_gates["constant_active_parameters"]);
    }

    #[test]
    fn topology_or_storage_shrink_across_cycles_rejects_candidate() {
        let mut resources = resources();
        let last = resources.candidate_capacity.last_mut().unwrap();
        last.stored_parameters -= 1;
        last.stored_bytes -= 1;
        reseal(&mut resources);
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.resource_gates["bounded_stored_parameters"]);
        assert!(!report.resource_gates["bounded_stored_bytes"]);
        assert!(!report.accepted);
    }

    #[test]
    fn wake_capacity_envelope_evidence_is_mandatory() {
        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .remove("candidate_capacity");
        assert!(
            serde_json::from_value::<ResourceComparison>(encoded).is_err(),
            "promotion evidence without capacity observations must fail closed"
        );

        let mut incomplete = resources();
        incomplete.candidate_capacity.truncate(1);
        assert!(incomplete.validate().is_err());
    }

    #[test]
    fn duplicate_seeds_fail_loudly() {
        let results = vec![CaseResult {
            case_id: "quality".into(),
            pairs: vec![
                PairedRun {
                    seed: 1,
                    baseline: 0.0,
                    candidate: 1.0,
                },
                PairedRun {
                    seed: 1,
                    baseline: 0.0,
                    candidate: 1.0,
                },
                PairedRun {
                    seed: 2,
                    baseline: 0.0,
                    candidate: 1.0,
                },
            ],
        }];
        assert!(evaluate_candidate(&suite(), &results, &resources(), &policy()).is_err());
    }

    #[test]
    fn raw_json_inputs_can_never_promote() {
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let report = evaluate_candidate(&suite(), &results, &resources(), &policy()).unwrap();
        assert!(!report.accepted);
        assert!(!report.resource_gates["content_addressed_evidence"]);
    }

    #[test]
    fn parity_memory_and_resume_are_measured_gates() {
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let mut resources = resources();
        resources.grouped_mm_parity.samples[0].candidate = 2.0;
        resources
            .candidate_capacity
            .last_mut()
            .unwrap()
            .stored_bytes += 1;
        reseal(&mut resources);
        let mut context = verified_context(&resources);
        context.exact_resume_verified = false;
        let report =
            evaluate_with_verified_context(&suite(), &results, &resources, &policy(), &context)
                .unwrap();
        assert!(!report.resource_gates["grouped_mm_parity"]);
        assert!(!report.resource_gates["bounded_stored_bytes"]);
        assert!(!report.resource_gates["exact_resume"]);
        assert!(!report.accepted);
    }

    #[test]
    fn exact_resume_requires_addressed_artifacts() {
        let mut exact = resources().exact_resume;
        exact.validate().unwrap();
        exact.resumed_metrics.path = PathBuf::new();
        assert!(exact.validate().is_err());
    }

    #[test]
    fn execution_receipt_and_safe_relative_roots_are_mandatory() {
        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded.as_object_mut().unwrap().remove("execution");
        assert!(serde_json::from_value::<ResourceComparison>(encoded).is_err());

        for root in ["/absolute", "../escape", "./dot"] {
            let mut resources = resources();
            resources.execution.approved_artifact_roots = vec![root.into()];
            assert!(resources.validate().is_err(), "accepted unsafe root {root}");
        }
    }

    #[test]
    fn suite_authors_cannot_expand_the_policy_anchor_set() {
        let mut forged_suite = suite();
        forged_suite.cases[0].stable_anchor = true;
        let results = forged_suite
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 1.0,
                        candidate: 0.999,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let resources = resources();
        let error = evaluate_with_verified_context(
            &forged_suite,
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("conflicts with policy"), "{error}");
    }

    #[test]
    fn resource_evidence_cannot_choose_parity_tolerances_or_submit_aggregates() {
        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded["grouped_mm_parity"]["absolute_tolerance"] = serde_json::json!(1e100);
        assert!(serde_json::from_value::<ResourceComparison>(encoded).is_err());

        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded["candidate_wake_tokens_per_second"] = serde_json::json!(1e100);
        assert!(serde_json::from_value::<ResourceComparison>(encoded).is_err());
    }

    #[test]
    fn wake_performance_is_recomputed_from_raw_trials() {
        let mut resources = resources();
        for trial in &mut resources.wake_trials {
            trial.candidate.elapsed_seconds = 20.0;
            trial.candidate.request_latency_ms = vec![20.0];
        }
        reseal(&mut resources);
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.resource_gates["wake_throughput"]);
        assert!(!report.resource_gates["wake_latency"]);
    }

    #[test]
    fn v1_acceptance_artifacts_fail_closed() {
        let mut old_suite = suite();
        old_suite.version = 1;
        assert!(old_suite.validate().is_err());

        let mut old_resources = resources();
        old_resources.version = 1;
        assert!(old_resources.validate().is_err());

        let mut old_policy = policy();
        old_policy.version = 1;
        assert!(old_policy.validate().is_err());
    }
}
