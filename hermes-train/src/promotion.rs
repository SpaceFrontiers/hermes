//! Trainer-owned, content-addressed WorkflowV2 promotion gate.
//!
//! Promotion is deliberately not a generic worker phase. The executor below
//! independently reloads the complete benchmark/ablation evidence, binds it to
//! the exact runtime checkpoint, evaluates the acceptance policy, and writes a
//! deterministic immutable decision. A rejected decision is retained for
//! audit but can never produce a [`PhaseProduct::Release`].

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::acceptance::{AcceptancePolicy, PromotionReport};
use crate::benchmark::{
    AblationId, BenchmarkTarget, ModelRepresentationIdentity, VerifiedBenchmarkRun,
    VerifiedResourceComparison, evaluate_verified_promotion,
};
use crate::runtime::{
    ImmutableArtifact, ImmutableModelCheckpoint, PhaseExecutionRequest, PhaseExecutionResult,
    PhaseExecutor, PhaseProduct, PhaseProgressSink,
};
use crate::workflow::{PhaseKind, PromotionConfig, PromotionEvidenceRef};

pub const PROMOTION_DECISION_VERSION: u32 = 2;

/// Complete auditable authorization record. The acceptance report is nested
/// beside the exact checkpoint and every immutable input identity so the
/// report cannot later be replayed for another model.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedPromotionDecision {
    pub version: u32,
    pub phase_index: usize,
    pub phase_name: String,
    pub candidate: ImmutableModelCheckpoint,
    pub benchmark_candidate_id: String,
    pub benchmark_candidate_representation: ModelRepresentationIdentity,
    pub candidate_training_evidence_sha256: String,
    pub config_sha256: String,
    pub selected_run_sha256: String,
    pub comparison_run_sha256: Vec<String>,
    pub resource_evidence_sha256: String,
    pub policy_sha256: String,
    pub report: PromotionReport,
}

/// Built-in promotion executor registered by the stock CLI and embedded host
/// in place of the external worker for [`PhaseKind::Promotion`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NativePromotionExecutor;

impl<C> PhaseExecutor<C> for NativePromotionExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        _context: &mut C,
        _progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        ensure!(
            request.phase.kind == PhaseKind::Promotion,
            "native promotion executor received a `{}` phase",
            request.phase.kind.name()
        );
        ensure!(
            request.resume_state.is_none(),
            "native promotion rejects opaque resume state from an external phase worker"
        );
        let candidate = request
            .input_checkpoint
            .as_ref()
            .context("native promotion requires the current immutable checkpoint")?;
        let config = request
            .phase
            .promotion
            .as_ref()
            .context("native promotion requires typed promotion settings")?;
        config.validate(&request.phase.name)?;

        let (decision, artifact) = evaluate_and_publish(request, candidate, config)?;
        ensure!(
            decision.report.accepted,
            "candidate failed native promotion gates; immutable rejection decision published at {}",
            artifact.uri()
        );
        Ok(PhaseExecutionResult::Complete(PhaseProduct::Release {
            candidate: candidate.clone(),
            receipt: artifact,
        }))
    }
}

fn evaluate_and_publish(
    request: &PhaseExecutionRequest,
    candidate: &ImmutableModelCheckpoint,
    config: &PromotionConfig,
) -> Result<(VerifiedPromotionDecision, ImmutableArtifact)> {
    // PhaseV2 validation normally precedes execution. Validate all files here
    // again at the trust boundary so direct executor users receive the same
    // fail-closed behavior.
    verify_evidence_ref(&config.selected_run, "selected benchmark run")?;
    for (index, run) in config.comparison_runs.iter().enumerate() {
        verify_evidence_ref(run, &format!("comparison benchmark run {index}"))?;
    }
    verify_evidence_ref(&config.resources, "resource evidence")?;
    verify_evidence_ref(&config.policy, "acceptance policy")?;

    let selected =
        VerifiedBenchmarkRun::load(&config.selected_run.path, config.selected_run.raw_sha256())?;
    let mut comparisons = Vec::with_capacity(AblationId::ALL.len());
    comparisons.push(selected.clone());
    for run in &config.comparison_runs {
        comparisons.push(VerifiedBenchmarkRun::load(&run.path, run.raw_sha256())?);
    }

    // Benchmark runs pin both checkpoint manifests and trainer-produced
    // training evidence. Re-verify those files at promotion time rather than
    // relying only on the earlier benchmark invocation.
    for run in &comparisons {
        verify_target(&run.run.metadata.baseline, &run.path)?;
        verify_target(&run.run.metadata.candidate, &run.path)?;
    }

    let benchmark_candidate = &selected.run.metadata.candidate;
    let runtime_digest = candidate
        .sha256()
        .strip_prefix("sha256:")
        .expect("immutable checkpoint validation enforces canonical digest");
    ensure!(
        benchmark_candidate
            .checkpoint_manifest_sha256
            .eq_ignore_ascii_case(runtime_digest),
        "selected benchmark candidate checkpoint {} does not match current workflow checkpoint {}",
        benchmark_candidate.checkpoint_manifest_sha256,
        candidate.sha256()
    );

    let resources =
        VerifiedResourceComparison::load(&config.resources.path, config.resources.raw_sha256())?;
    let policy: AcceptancePolicy = read_addressed_json(&config.policy, "acceptance policy")?;
    let report = evaluate_verified_promotion(
        &selected,
        &comparisons,
        &resources,
        &policy,
        config.policy.raw_sha256(),
    )?;
    let config_sha256 = canonical_sha256(config)?;
    let decision = VerifiedPromotionDecision {
        version: PROMOTION_DECISION_VERSION,
        phase_index: request.phase_index,
        phase_name: request.phase.name.clone(),
        candidate: candidate.clone(),
        benchmark_candidate_id: benchmark_candidate.id.clone(),
        benchmark_candidate_representation: benchmark_candidate.representation_identity(),
        candidate_training_evidence_sha256: format!(
            "sha256:{}",
            benchmark_candidate
                .training_evidence_sha256
                .to_ascii_lowercase()
        ),
        config_sha256,
        selected_run_sha256: config.selected_run.sha256.clone(),
        comparison_run_sha256: config
            .comparison_runs
            .iter()
            .map(|run| run.sha256.clone())
            .collect(),
        resource_evidence_sha256: config.resources.sha256.clone(),
        policy_sha256: config.policy.sha256.clone(),
        report,
    };
    let bytes = decision_bytes(&decision)?;
    let decision_sha256 = format!("sha256:{:x}", Sha256::digest(&bytes));
    let directory = prepare_artifact_directory(&config.artifact_directory)?;
    let path = directory.join(format!(
        "sha256-{}.json",
        decision_sha256
            .strip_prefix("sha256:")
            .expect("locally constructed digest")
    ));
    publish_idempotent(&path, &bytes)?;
    let uri = path
        .to_str()
        .context("promotion decision path is not valid UTF-8")?
        .to_owned();
    let artifact = ImmutableArtifact::new(uri, decision_sha256)?;
    Ok((decision, artifact))
}

fn verify_target(target: &BenchmarkTarget, run_source: &Path) -> Result<()> {
    let mut resolved = target.clone();
    resolved.resolve_paths(run_source);
    resolved.verify().map(|_| ()).with_context(|| {
        format!(
            "failed to verify benchmark target `{}` referenced by {}",
            resolved.id,
            run_source.display()
        )
    })
}

fn verify_evidence_ref(reference: &PromotionEvidenceRef, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(&reference.path)
        .with_context(|| format!("{label} {} is unavailable", reference.path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        reference.path.display()
    );
    Ok(())
}

fn read_addressed_json<T: serde::de::DeserializeOwned>(
    reference: &PromotionEvidenceRef,
    label: &str,
) -> Result<T> {
    let bytes = fs::read(&reference.path)
        .with_context(|| format!("failed to read {label} {}", reference.path.display()))?;
    let actual = format!("sha256:{:x}", Sha256::digest(&bytes));
    ensure!(
        actual == reference.sha256,
        "{label} hash mismatch for {}: expected {}, got {}",
        reference.path.display(),
        reference.sha256,
        actual
    );
    serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid {label} JSON in {}", reference.path.display()))
}

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(value)?)
    ))
}

fn decision_bytes(decision: &VerifiedPromotionDecision) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(decision)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Resolve an output directory through its nearest existing ancestor, then
/// create each missing normal component. The final directory itself may never
/// be a symlink.
fn prepare_artifact_directory(path: &Path) -> Result<PathBuf> {
    ensure!(
        !path.as_os_str().is_empty(),
        "promotion artifact directory is empty"
    );
    if let Ok(metadata) = fs::symlink_metadata(path) {
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "promotion artifact directory {} must be a non-symlink directory",
            path.display()
        );
        if let Some(parent) = path.parent() {
            let parent_metadata = fs::symlink_metadata(parent).with_context(|| {
                format!("inspect promotion artifact parent {}", parent.display())
            })?;
            ensure!(
                parent_metadata.file_type().is_dir() && !parent_metadata.file_type().is_symlink(),
                "promotion artifact parent {} must be a non-symlink directory",
                parent.display()
            );
        }
        return path
            .canonicalize()
            .with_context(|| format!("canonicalize promotion directory {}", path.display()));
    }

    let mut missing = Vec::new();
    let mut ancestor = path;
    loop {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) => {
                ensure!(
                    metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
                    "promotion artifact ancestor {} must be a non-symlink directory",
                    ancestor.display()
                );
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let component = ancestor
                    .file_name()
                    .context("promotion artifact directory has no existing ancestor")?;
                missing.push(component.to_owned());
                ancestor = ancestor
                    .parent()
                    .context("promotion artifact directory has no existing ancestor")?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "inspect promotion artifact directory {}",
                        ancestor.display()
                    )
                });
            }
        }
    }
    let mut directory = ancestor
        .canonicalize()
        .with_context(|| format!("canonicalize promotion ancestor {}", ancestor.display()))?;
    for component in missing.into_iter().rev() {
        ensure!(
            Path::new(&component)
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
            "promotion artifact directory contains an unsafe path component"
        );
        directory.push(component);
        match fs::create_dir(&directory) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "create promotion artifact directory {}",
                        directory.display()
                    )
                });
            }
        }
        let metadata = fs::symlink_metadata(&directory)?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "promotion artifact directory {} became a symlink",
            directory.display()
        );
    }
    Ok(directory)
}

fn publish_idempotent(path: &Path, expected: &[u8]) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "promotion decision {} must be a regular non-symlink file",
                path.display()
            );
            ensure!(
                fs::read(path)? == expected,
                "existing promotion decision {} does not contain the derived bytes",
                path.display()
            );
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect promotion decision {}", path.display()));
        }
    }

    let parent = path
        .parent()
        .context("promotion decision path has no parent")?;
    let file_name = path
        .file_name()
        .context("promotion decision path has no file name")?;
    for attempt in 0..100_u32 {
        let temporary = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            attempt
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).context("create promotion decision temporary file");
            }
        };
        let publication = (|| -> Result<()> {
            file.write_all(expected)?;
            file.sync_all()?;
            match fs::hard_link(&temporary, path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let metadata = fs::symlink_metadata(path)?;
                    ensure!(
                        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                        "concurrent promotion decision is not a regular file"
                    );
                    ensure!(
                        fs::read(path)? == expected,
                        "concurrent promotion decision contains different bytes"
                    );
                }
                Err(error) => return Err(error).context("publish promotion decision"),
            }
            fs::remove_file(&temporary)?;
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if publication.is_err() {
            drop(file);
            let _ = fs::remove_file(&temporary);
        }
        return publication;
    }
    bail!(
        "failed to allocate a promotion decision temporary file beside {}",
        path.display()
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::acceptance::{
        CapacityObservation, ExactResumeArtifact, ExactResumeEvidence, KernelParityEvidence,
        KernelParitySample, PairedWakeTrial, RESOURCE_COMPARISON_VERSION, ResourceComparison,
        SuiteVisibility, WakeMeasurement,
    };
    use crate::benchmark::{
        BENCHMARK_MANIFEST_VERSION, BenchmarkArtifact, BenchmarkEvaluator, BenchmarkRun,
        BenchmarkRunConfig, BenchmarkRunner, BenchmarkSpec, BenchmarkSuiteManifest,
        EvaluationMeasurement, EvaluationRequest, MetricDirection, TargetRole, TrainingEvidence,
        VerifiedBenchmarkSuite, required_catalog,
    };
    use crate::runtime::{
        ExecutorRegistry, RuntimeBoundary, RuntimeCheckpoint, RuntimeStatus, WorkflowRunState,
        run_next_phase,
    };
    use crate::workflow::{ResolvedWorkflow, WorkflowV2};

    fn raw_sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn write_json<T: Serialize>(path: &Path, value: &T) -> String {
        let bytes = serde_json::to_vec_pretty(value).unwrap();
        fs::write(path, &bytes).unwrap();
        raw_sha256(&bytes)
    }

    fn target(root: &Path, id: &str) -> BenchmarkTarget {
        use safetensors::{Dtype, tensor::TensorView};

        let output = root.join(id);
        let staging = output.join("generations").join("staging");
        fs::create_dir_all(&staging).unwrap();
        let raw_weights = vec![0_u8; 100 * 4];
        let view = TensorView::new(Dtype::F32, vec![100], &raw_weights).unwrap();
        let weights = safetensors::tensor::serialize([("weight", view)], None).unwrap();
        let weights_sha256 = raw_sha256(&weights);
        fs::write(staging.join("weights.safetensors"), &weights).unwrap();
        let accounting = crate::benchmark::TrainingAccounting {
            version: crate::benchmark::TRAINING_ACCOUNTING_VERSION,
            training_gpu_hours: 8.0,
            parameters: 100,
            routed_active_parameters: 80,
            weights_bytes: weights.len() as u64,
            weights_sha256: weights_sha256.clone(),
        };
        let accounting_bytes = serde_json::to_vec(&accounting).unwrap();
        let accounting_sha256 = raw_sha256(&accounting_bytes);
        fs::write(
            staging.join(crate::benchmark::TRAINING_ACCOUNTING_FILE),
            &accounting_bytes,
        )
        .unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "training_state_version": 2,
            "global_step": 20,
            "phase": 0,
            "phase_id": "test",
            "files": [
                {"path": crate::benchmark::TRAINING_ACCOUNTING_FILE, "bytes": accounting_bytes.len(), "sha256": accounting_sha256},
                {"path": "weights.safetensors", "bytes": weights.len(), "sha256": weights_sha256}
            ]
        });
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let checkpoint_manifest_sha256 = raw_sha256(&manifest_bytes);
        let generation = output
            .join("generations")
            .join(format!("sha256-{checkpoint_manifest_sha256}"));
        fs::rename(&staging, &generation).unwrap();
        let checkpoint_manifest = generation.join("generation-manifest.json");
        fs::write(&checkpoint_manifest, &manifest_bytes).unwrap();
        let evidence = TrainingEvidence {
            version: 1,
            checkpoint_manifest_sha256: checkpoint_manifest_sha256.clone(),
            accounting_sha256,
            training_gpu_hours: 8.0,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: weights.len() as u64,
            weights_sha256,
        };
        let evidence_bytes = serde_json::to_vec(&evidence).unwrap();
        let training_evidence_sha256 = raw_sha256(&evidence_bytes);
        let training_evidence = output
            .join("training-evidence")
            .join(format!("sha256-{training_evidence_sha256}.json"));
        fs::create_dir_all(training_evidence.parent().unwrap()).unwrap();
        fs::write(&training_evidence, evidence_bytes).unwrap();
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest,
            checkpoint_manifest_sha256,
            training_evidence,
            training_evidence_sha256,
            training_gpu_hours: 8.0,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: weights.len() as u64,
            representation: crate::benchmark::ModelRepresentationTarget::FullPrecision,
        }
    }

    fn catalog_suite(
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
                let artifact = root.join(format!("{case_id}.jsonl"));
                fs::write(&artifact, format!("{{\"id\":\"{case_id}\"}}\n")).unwrap();
                BenchmarkSpec {
                    id: case_id,
                    catalog_id: Some(entry.id.into()),
                    family: entry.family,
                    metric: "score".into(),
                    direction: MetricDirection::Maximize,
                    stable_anchor: stable_anchors.contains(entry.id),
                    artifact: BenchmarkArtifact {
                        path: artifact.file_name().unwrap().into(),
                        sha256: raw_sha256(&fs::read(&artifact).unwrap()),
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
        let digest = write_json(&path, &manifest);
        VerifiedBenchmarkSuite::load(path, &digest).unwrap()
    }

    struct PassingEvaluator;

    impl BenchmarkEvaluator for PassingEvaluator {
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
                examples: 32,
                metrics: BTreeMap::new(),
            })
        }
    }

    struct Fixture {
        workflow: ResolvedWorkflow,
        checkpoint: ImmutableModelCheckpoint,
        artifact_directory: PathBuf,
        candidate_training_evidence: PathBuf,
        resources_path: PathBuf,
    }

    fn resource_comparison(selected: &BenchmarkRun, selected_sha256: &str) -> ResourceComparison {
        use crate::metrics::{
            MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, MetricWriter,
            ThroughputMetrics,
        };

        let baseline = &selected.metadata.baseline;
        let candidate = &selected.metadata.candidate;
        let final_manifest = &candidate.checkpoint_manifest;
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
        let mut interrupted: serde_json::Value =
            serde_json::from_slice(&fs::read(final_manifest).unwrap()).unwrap();
        interrupted["global_step"] = serde_json::json!(10);
        let interrupted_bytes = serde_json::to_vec(&interrupted).unwrap();
        let interrupted_sha256 = raw_sha256(&interrupted_bytes);
        let interrupted_generation = exact_root
            .join("interrupted")
            .join("generations")
            .join(format!("sha256-{interrupted_sha256}"));
        fs::create_dir_all(&interrupted_generation).unwrap();
        for entry in fs::read_dir(final_generation).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name() != "generation-manifest.json" {
                fs::copy(entry.path(), interrupted_generation.join(entry.file_name())).unwrap();
            }
        }
        let interrupted_manifest = interrupted_generation.join("generation-manifest.json");
        fs::write(&interrupted_manifest, interrupted_bytes).unwrap();
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
            version: RESOURCE_COMPARISON_VERSION,
            baseline_id: baseline.id.clone(),
            candidate_id: candidate.id.clone(),
            benchmark_run_sha256: selected_sha256.into(),
            strongest_baseline_id: baseline.id.clone(),
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
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                })
                .collect(),
            candidate_capacity: (0..=3)
                .map(|completed_sleep_cycles| CapacityObservation {
                    completed_sleep_cycles,
                    routed_active_parameters: candidate.routed_active_parameters,
                    stored_parameters: candidate.parameters,
                    stored_bytes: candidate.stored_bytes,
                })
                .collect(),
            grouped_mm_parity: KernelParityEvidence {
                fixture_sha256: "a".repeat(64),
                samples: (0..1024)
                    .map(|_| KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0,
                    })
                    .collect(),
            },
            pytorch_parity: KernelParityEvidence {
                fixture_sha256: "b".repeat(64),
                samples: (0..1024)
                    .map(|_| KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0,
                    })
                    .collect(),
            },
            exact_resume: ExactResumeEvidence {
                interrupted_checkpoint: ExactResumeArtifact {
                    path: relative(&interrupted_manifest),
                    sha256: interrupted_sha256,
                },
                uninterrupted_final_state: ExactResumeArtifact {
                    path: relative(final_manifest),
                    sha256: candidate.checkpoint_manifest_sha256.clone(),
                },
                resumed_final_state: ExactResumeArtifact {
                    path: relative(&resumed_manifest),
                    sha256: candidate.checkpoint_manifest_sha256.clone(),
                },
                uninterrupted_metrics: ExactResumeArtifact {
                    sha256: raw_sha256(&fs::read(&uninterrupted_metrics).unwrap()),
                    path: relative(&uninterrupted_metrics),
                },
                resumed_metrics: ExactResumeArtifact {
                    sha256: raw_sha256(&fs::read(&resumed_metrics).unwrap()),
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

    fn fixture() -> Fixture {
        let temporary = tempfile::tempdir().unwrap().keep();
        let public = catalog_suite(&temporary, "public", SuiteVisibility::Public);
        let sealed = catalog_suite(&temporary, "sealed", SuiteVisibility::Sealed);
        let candidate = target(&temporary, "candidate");
        let candidate_training_evidence = candidate.training_evidence.clone();
        let mut run_references = Vec::new();
        let mut selected_run = None;
        let mut selected_digest = None;
        for (index, ablation) in AblationId::ALL.into_iter().enumerate() {
            let baseline = target(&temporary, &format!("baseline-{index}"));
            let run = BenchmarkRunner {
                config: BenchmarkRunConfig {
                    evaluator_id: "promotion-test".into(),
                    evaluator_version:
                        "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                            .into(),
                    paired_seeds: vec![11, 22, 33],
                    order_seed: 7,
                    gpu_hours_per_evaluation: 0.01,
                    ablation: Some(ablation),
                },
            }
            .run(
                &[public.clone(), sealed.clone()],
                &baseline,
                &candidate,
                &mut PassingEvaluator,
            )
            .unwrap();
            let path = temporary.join(format!("run-{index}.json"));
            let digest = write_json(&path, &run);
            let reference = PromotionEvidenceRef {
                path,
                sha256: format!("sha256:{digest}"),
            };
            if index + 1 == AblationId::ALL.len() {
                selected_run = Some(run);
                selected_digest = Some(digest);
            }
            run_references.push(reference);
        }
        let selected_run = selected_run.unwrap();
        let selected_digest = selected_digest.unwrap();
        let selected_reference = run_references.pop().unwrap();
        let policy_path = temporary.join("policy.json");
        let policy = AcceptancePolicy::default();
        let policy_digest = write_json(&policy_path, &policy);
        let selected_verified =
            VerifiedBenchmarkRun::load(&selected_reference.path, selected_reference.raw_sha256())
                .unwrap();
        let mut verified_runs = vec![selected_verified.clone()];
        verified_runs.extend(run_references.iter().map(|reference| {
            VerifiedBenchmarkRun::load(&reference.path, reference.raw_sha256()).unwrap()
        }));
        let resources_path = temporary.join("resources.json");
        let mut resources = resource_comparison(&selected_run, &selected_digest);
        resources.execution = crate::resource_worker::derive_execution_receipt(
            &selected_verified,
            &verified_runs,
            &policy,
            &policy_digest,
            Vec::new(),
            resources.execution.approved_artifact_roots.clone(),
            &resources,
        )
        .unwrap();
        let resources_digest = write_json(&resources_path, &resources);
        let artifact_directory = temporary.join("decisions");
        let config = PromotionConfig {
            selected_run: selected_reference,
            comparison_runs: run_references,
            resources: PromotionEvidenceRef {
                path: resources_path.clone(),
                sha256: format!("sha256:{resources_digest}"),
            },
            policy: PromotionEvidenceRef {
                path: policy_path,
                sha256: format!("sha256:{policy_digest}"),
            },
            artifact_directory: artifact_directory.clone(),
        };
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "name": "native-promotion-test",
            "phases": [{
                "name": "promotion",
                "type": "promotion",
                "promotion": config
            }]
        }))
        .unwrap();
        let workflow = workflow.resolve(&temporary.join("workflow.json")).unwrap();
        let checkpoint = ImmutableModelCheckpoint::new(
            "checkpoint://candidate",
            format!(
                "sha256:{}",
                selected_run.metadata.candidate.checkpoint_manifest_sha256
            ),
        )
        .unwrap();
        Fixture {
            workflow,
            checkpoint,
            artifact_directory,
            candidate_training_evidence,
            resources_path,
        }
    }

    #[derive(Default)]
    struct FailPreparedOnce {
        fail: bool,
    }

    impl RuntimeCheckpoint for FailPreparedOnce {
        fn persist(&mut self, boundary: &RuntimeBoundary, _state: &WorkflowRunState) -> Result<()> {
            if self.fail && matches!(boundary, RuntimeBoundary::PhasePrepared { .. }) {
                self.fail = false;
                bail!("simulated failure after immutable decision publication")
            }
            Ok(())
        }
    }

    #[test]
    fn runtime_retries_the_same_verified_receipt_without_external_promotion() {
        let fixture = fixture();
        let mut state =
            WorkflowRunState::new(&fixture.workflow, Some(fixture.checkpoint.clone())).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Promotion, NativePromotionExecutor)
            .unwrap();
        let mut persistence = FailPreparedOnce { fail: true };
        let error = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("simulated failure"), "{error}");
        assert_eq!(
            fs::read_dir(&fixture.artifact_directory).unwrap().count(),
            1
        );

        let status = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap();
        assert!(matches!(
            status,
            RuntimeStatus::PhaseCommitted {
                workflow_complete: true,
                ..
            }
        ));
        assert_eq!(
            fs::read_dir(&fixture.artifact_directory).unwrap().count(),
            1
        );
        let receipt = match state.completed_phases()[0].product() {
            PhaseProduct::Release { candidate, receipt } => {
                assert_eq!(candidate, &fixture.checkpoint);
                receipt
            }
            other => panic!("unexpected product: {other:?}"),
        };
        let bytes = fs::read(receipt.uri()).unwrap();
        assert_eq!(receipt.sha256(), format!("sha256:{}", raw_sha256(&bytes)));
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["version"], PROMOTION_DECISION_VERSION);
        assert_eq!(json["report"]["accepted"], true);
        assert_eq!(json["candidate"]["sha256"], fixture.checkpoint.sha256());
        assert_eq!(
            json["benchmark_candidate_representation"]["type"],
            "full_precision"
        );

        let mut changed = fixture.workflow.clone();
        changed.phases[0]
            .promotion
            .as_mut()
            .unwrap()
            .artifact_directory
            .push("changed");
        let error = state.validate(&changed).unwrap_err().to_string();
        assert!(error.contains("does not belong"), "{error}");
    }

    #[test]
    fn promotion_rechecks_training_evidence_and_rejects_tampering() {
        let fixture = fixture();
        fs::write(&fixture.candidate_training_evidence, b"{}\n").unwrap();
        let mut state = WorkflowRunState::new(&fixture.workflow, Some(fixture.checkpoint)).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Promotion, NativePromotionExecutor)
            .unwrap();
        let error = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut FailPreparedOnce::default(),
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("training evidence hash mismatch"), "{error}");
        assert!(!fixture.artifact_directory.exists());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_directory_rejects_a_symlink_ancestor() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let target = temporary.path().join("target");
        fs::create_dir(&target).unwrap();
        let link = temporary.path().join("linked");
        symlink(&target, &link).unwrap();
        let error = prepare_artifact_directory(&link.join("decision"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-symlink directory"), "{error}");
        assert!(!target.join("decision").exists());

        let existing = target.join("existing");
        fs::create_dir(&existing).unwrap();
        let error = prepare_artifact_directory(&link.join("existing"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-symlink directory"), "{error}");
    }

    #[test]
    fn rejected_report_is_immutable_but_never_becomes_a_release() {
        let mut fixture = fixture();
        let mut resources: ResourceComparison =
            serde_json::from_slice(&fs::read(&fixture.resources_path).unwrap()).unwrap();
        for trial in &mut resources.wake_trials {
            trial.candidate.request_latency_ms = vec![100.0];
        }
        resources.execution.observations_sha256 = resources.observations_sha256().unwrap();
        let digest = write_json(&fixture.resources_path, &resources);
        fixture.workflow.phases[0]
            .promotion
            .as_mut()
            .unwrap()
            .resources
            .sha256 = format!("sha256:{digest}");
        let mut state = WorkflowRunState::new(&fixture.workflow, Some(fixture.checkpoint)).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Promotion, NativePromotionExecutor)
            .unwrap();
        let error = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut FailPreparedOnce::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("failed native promotion gates"), "{error}");
        assert!(state.completed_phases().is_empty());
        let path = fs::read_dir(&fixture.artifact_directory)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let json: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(json["report"]["accepted"], false);
    }
}
