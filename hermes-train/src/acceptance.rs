//! Reproducible acceptance gates for immutable training candidates.
//!
//! Public and sealed suites share the same metric contract.  Sealed results
//! intentionally expose only the aggregate gate to the training workflow so
//! candidate generation cannot adapt to hidden examples.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

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
    pub family: BenchmarkFamily,
    pub visibility: SuiteVisibility,
    /// Higher-is-better metric written by the evaluator.
    pub metric: String,
    #[serde(default)]
    pub stable_anchor: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceSuite {
    pub version: u32,
    pub cases: Vec<BenchmarkCase>,
}

impl AcceptanceSuite {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported acceptance suite version {}",
            self.version
        );
        ensure!(!self.cases.is_empty(), "acceptance suite has no cases");
        let mut ids = BTreeSet::new();
        for case in &self.cases {
            ensure!(!case.id.trim().is_empty(), "acceptance case id is empty");
            ensure!(
                ids.insert(case.id.as_str()),
                "duplicate acceptance case `{}`",
                case.id
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
    pub baseline_id: String,
    pub candidate_id: String,
    /// SHA-256 of the exact [`crate::benchmark::BenchmarkRun`] JSON artifact.
    pub benchmark_run_sha256: String,
    /// This is checked against the baseline identity derived from all verified,
    /// capacity- and compute-matched comparison runs.  The value is never
    /// sufficient by itself to authorize promotion.
    pub strongest_baseline_id: String,
    pub baseline_stored_parameters: u64,
    pub candidate_stored_parameters: u64,
    pub maximum_candidate_stored_parameters: u64,
    pub baseline_stored_bytes: u64,
    pub candidate_stored_bytes: u64,
    pub maximum_candidate_stored_bytes: u64,
    pub baseline_routed_active_parameters: u64,
    pub candidate_routed_active_parameters: u64,
    pub baseline_training_gpu_hours: f64,
    pub candidate_training_gpu_hours: f64,
    pub baseline_wake_tokens_per_second: f64,
    pub candidate_wake_tokens_per_second: f64,
    pub baseline_wake_p95_ms: f64,
    pub candidate_wake_p95_ms: f64,
    pub active_parameters_before: u64,
    pub active_parameters_after: u64,
    pub grouped_mm_parity: KernelParityEvidence,
    pub pytorch_parity: KernelParityEvidence,
    pub exact_resume: ExactResumeEvidence,
}

/// Numerical comparison produced by a fixed reference fixture.  Requiring
/// both bounds is deliberately conservative: an implementation cannot hide a
/// large relative error behind a permissive absolute tolerance (or vice
/// versa).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParityEvidence {
    pub fixture_sha256: String,
    pub samples: u64,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl KernelParityEvidence {
    fn validate(&self, name: &str) -> Result<()> {
        validate_sha256(&self.fixture_sha256)
            .map_err(|error| anyhow::anyhow!("{name} fixture: {error}"))?;
        ensure!(self.samples > 0, "{name} parity evaluated no values");
        ensure!(
            [
                self.max_absolute_error,
                self.max_relative_error,
                self.absolute_tolerance,
                self.relative_tolerance,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0),
            "{name} parity contains an invalid error or tolerance"
        );
        Ok(())
    }

    fn passed(&self) -> bool {
        self.max_absolute_error <= self.absolute_tolerance
            && self.max_relative_error <= self.relative_tolerance
    }
}

/// Evidence that an interrupted run and an uninterrupted reference converged
/// to byte-identical state and metric streams from the same checkpoint.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactResumeEvidence {
    pub interrupted_checkpoint_sha256: String,
    pub uninterrupted_final_state_sha256: String,
    pub resumed_final_state_sha256: String,
    pub uninterrupted_metrics_sha256: String,
    pub resumed_metrics_sha256: String,
    pub interruption_step: u64,
    pub resumed_from_step: u64,
}

impl ExactResumeEvidence {
    fn validate(&self) -> Result<()> {
        for (name, digest) in [
            (
                "interrupted checkpoint",
                &self.interrupted_checkpoint_sha256,
            ),
            (
                "uninterrupted final state",
                &self.uninterrupted_final_state_sha256,
            ),
            ("resumed final state", &self.resumed_final_state_sha256),
            ("uninterrupted metrics", &self.uninterrupted_metrics_sha256),
            ("resumed metrics", &self.resumed_metrics_sha256),
        ] {
            validate_sha256(digest).map_err(|error| anyhow::anyhow!("{name}: {error}"))?;
        }
        ensure!(
            self.interruption_step > 0,
            "resume evidence must interrupt after at least one step"
        );
        Ok(())
    }

    fn passed(&self) -> bool {
        self.interruption_step == self.resumed_from_step
            && self.uninterrupted_final_state_sha256 == self.resumed_final_state_sha256
            && self.uninterrupted_metrics_sha256 == self.resumed_metrics_sha256
    }
}

impl ResourceComparison {
    pub fn validate(&self) -> Result<()> {
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
            self.baseline_stored_parameters > 0
                && self.candidate_stored_parameters > 0
                && self.maximum_candidate_stored_parameters > 0,
            "stored parameter measurements must be positive"
        );
        ensure!(
            self.baseline_stored_bytes > 0
                && self.candidate_stored_bytes > 0
                && self.maximum_candidate_stored_bytes > 0,
            "stored-byte measurements must be positive"
        );
        ensure!(
            self.baseline_routed_active_parameters > 0
                && self.candidate_routed_active_parameters > 0
                && self.active_parameters_before > 0
                && self.active_parameters_after > 0,
            "active parameter measurements must be positive"
        );
        ensure!(
            [
                self.baseline_training_gpu_hours,
                self.candidate_training_gpu_hours,
                self.baseline_wake_tokens_per_second,
                self.candidate_wake_tokens_per_second,
                self.baseline_wake_p95_ms,
                self.candidate_wake_p95_ms,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0),
            "resource comparison contains an invalid positive measurement"
        );
        self.grouped_mm_parity.validate("grouped-mm")?;
        self.pytorch_parity.validate("PyTorch")?;
        self.exact_resume.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptancePolicy {
    pub minimum_paired_seeds: usize,
    pub maximum_anchor_regression: f64,
    pub minimum_wake_throughput_ratio: f64,
    pub maximum_wake_latency_ratio: f64,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self {
            minimum_paired_seeds: 3,
            maximum_anchor_regression: 0.01,
            minimum_wake_throughput_ratio: 0.95,
            maximum_wake_latency_ratio: 1.05,
        }
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

/// Sealed examples and per-case deltas are deliberately absent.  Only this
/// aggregate is allowed to cross the evaluator boundary.
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
    pub content_addressed_evidence: bool,
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
    let capacity_matched =
        resources.baseline_stored_parameters == resources.candidate_stored_parameters;
    let active_capacity_matched =
        resources.baseline_routed_active_parameters == resources.candidate_routed_active_parameters;
    let fixed_gpu_hour_measured = same_f64(
        resources.baseline_training_gpu_hours,
        resources.candidate_training_gpu_hours,
    );
    let context = VerifiedPromotionContext {
        benchmark_run_sha256: &resources.benchmark_run_sha256,
        strongest_baseline_id: &resources.strongest_baseline_id,
        baseline_id: &resources.baseline_id,
        candidate_id: &resources.candidate_id,
        capacity_matched,
        active_capacity_matched,
        fixed_gpu_hour_measured,
        content_addressed_evidence: false,
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
    resources.validate()?;
    ensure!(
        policy.minimum_paired_seeds >= 3,
        "promotion requires at least three paired seeds"
    );
    ensure!(
        policy.maximum_anchor_regression.is_finite() && policy.maximum_anchor_regression >= 0.0,
        "maximum anchor regression must be finite and non-negative"
    );
    ensure!(
        policy.minimum_wake_throughput_ratio.is_finite()
            && (0.0..=1.0).contains(&policy.minimum_wake_throughput_ratio),
        "minimum wake throughput ratio must be in [0, 1]"
    );
    ensure!(
        policy.maximum_wake_latency_ratio.is_finite() && policy.maximum_wake_latency_ratio >= 1.0,
        "maximum wake latency ratio must be at least one"
    );

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

    let throughput_ratio =
        resources.candidate_wake_tokens_per_second / resources.baseline_wake_tokens_per_second;
    let latency_ratio = resources.candidate_wake_p95_ms / resources.baseline_wake_p95_ms;
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
        ("exact_resume".into(), resources.exact_resume.passed()),
        (
            "constant_active_parameters".into(),
            resources.active_parameters_before == resources.active_parameters_after
                && resources.active_parameters_before
                    == resources.candidate_routed_active_parameters,
        ),
        (
            "bounded_stored_parameters".into(),
            resources.candidate_stored_parameters <= resources.maximum_candidate_stored_parameters
                && resources.maximum_candidate_stored_parameters
                    <= resources.baseline_stored_parameters,
        ),
        (
            "bounded_stored_bytes".into(),
            resources.candidate_stored_bytes <= resources.maximum_candidate_stored_bytes
                && resources.maximum_candidate_stored_bytes <= resources.baseline_stored_bytes,
        ),
        (
            "grouped_mm_parity".into(),
            resources.grouped_mm_parity.passed(),
        ),
        ("pytorch_parity".into(), resources.pytorch_parity.passed()),
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
            version: 1,
            cases: vec![
                BenchmarkCase {
                    id: "quality".into(),
                    family: BenchmarkFamily::Reasoning,
                    visibility: SuiteVisibility::Public,
                    metric: "accuracy".into(),
                    stable_anchor: false,
                },
                BenchmarkCase {
                    id: "retention".into(),
                    family: BenchmarkFamily::ContinualRetention,
                    visibility: SuiteVisibility::Sealed,
                    metric: "accuracy".into(),
                    stable_anchor: true,
                },
            ],
        }
    }

    fn resources() -> ResourceComparison {
        ResourceComparison {
            baseline_id: "static-moe".into(),
            candidate_id: "sleep-candidate".into(),
            benchmark_run_sha256: "1".repeat(64),
            strongest_baseline_id: "static-moe".into(),
            baseline_stored_parameters: 100,
            candidate_stored_parameters: 100,
            maximum_candidate_stored_parameters: 100,
            baseline_stored_bytes: 1_000,
            candidate_stored_bytes: 1_000,
            maximum_candidate_stored_bytes: 1_000,
            baseline_routed_active_parameters: 80,
            candidate_routed_active_parameters: 80,
            baseline_training_gpu_hours: 2.0,
            candidate_training_gpu_hours: 2.0,
            baseline_wake_tokens_per_second: 100.0,
            candidate_wake_tokens_per_second: 96.0,
            baseline_wake_p95_ms: 10.0,
            candidate_wake_p95_ms: 10.4,
            active_parameters_before: 80,
            active_parameters_after: 80,
            grouped_mm_parity: KernelParityEvidence {
                fixture_sha256: "2".repeat(64),
                samples: 1024,
                max_absolute_error: 1e-5,
                max_relative_error: 1e-4,
                absolute_tolerance: 2e-5,
                relative_tolerance: 2e-4,
            },
            pytorch_parity: KernelParityEvidence {
                fixture_sha256: "3".repeat(64),
                samples: 1024,
                max_absolute_error: 1e-5,
                max_relative_error: 1e-4,
                absolute_tolerance: 2e-5,
                relative_tolerance: 2e-4,
            },
            exact_resume: ExactResumeEvidence {
                interrupted_checkpoint_sha256: "4".repeat(64),
                uninterrupted_final_state_sha256: "5".repeat(64),
                resumed_final_state_sha256: "5".repeat(64),
                uninterrupted_metrics_sha256: "6".repeat(64),
                resumed_metrics_sha256: "6".repeat(64),
                interruption_step: 17,
                resumed_from_step: 17,
            },
        }
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
            content_addressed_evidence: true,
        }
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
            &AcceptancePolicy::default(),
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
        resources.active_parameters_after += 1;
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
            &AcceptancePolicy::default(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.accepted);
        assert!(!report.resource_gates["constant_active_parameters"]);
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
        assert!(
            evaluate_candidate(
                &suite(),
                &results,
                &resources(),
                &AcceptancePolicy::default()
            )
            .is_err()
        );
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
        let report = evaluate_candidate(
            &suite(),
            &results,
            &resources(),
            &AcceptancePolicy::default(),
        )
        .unwrap();
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
        resources.grouped_mm_parity.max_absolute_error = 1.0;
        resources.candidate_stored_bytes += 1;
        resources.exact_resume.resumed_metrics_sha256 = "7".repeat(64);
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &AcceptancePolicy::default(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.resource_gates["grouped_mm_parity"]);
        assert!(!report.resource_gates["bounded_stored_bytes"]);
        assert!(!report.resource_gates["exact_resume"]);
        assert!(!report.accepted);
    }
}
