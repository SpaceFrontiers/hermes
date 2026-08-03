//! Strict, content-pinned execution bridge for promotion resource evidence.
//!
//! The worker receives verified model transports, but experiment identity is
//! computed only from content hashes and policy/workload fields. Its response
//! contains raw observations only; the host supplies every identity, seals the
//! receipt under a content-addressed filename, and reopens it before returning.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
#[cfg(test)]
use std::io::{BufRead, BufReader};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::artifact_io::{
    json_sha256_identity, sha256_hex, validate_sha256_hex, validate_sha256_identity,
};

use crate::acceptance::{
    AcceptancePolicy, CapacityObservation, ExactResumeArtifact, ExactResumeEvidence,
    KernelParityEvidence, KernelParitySample, PairedWakeTrial, RESOURCE_COMPARISON_VERSION,
    RESOURCE_EXECUTION_PROTOCOL_VERSION, ResourceComparison, ResourceExecutionReceipt,
};
use crate::benchmark::{
    BenchmarkTarget, ModelRepresentationIdentity, VerifiedBenchmarkRun, VerifiedResourceComparison,
    verified_resource_benchmark_context, verify_resource_comparison_artifacts,
};
#[cfg(unix)]
use crate::pinned_executable::PinnedExecutable;
#[cfg(test)]
use crate::pinned_executable::file_sha256 as pinned_file_sha256;
#[cfg(unix)]
use crate::protocol_process::{ProtocolRead, SupervisedProcess};
const MAX_RESOURCE_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_RESOURCE_EVALUATOR_TIMEOUT: Duration = Duration::from_secs(3_600);
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceTargetIdentity {
    pub id: String,
    pub checkpoint_manifest_sha256: String,
    pub training_evidence_sha256: String,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub stored_bytes: u64,
    pub representation: ModelRepresentationIdentity,
}

impl From<&BenchmarkTarget> for ResourceTargetIdentity {
    fn from(target: &BenchmarkTarget) -> Self {
        Self {
            id: target.id.clone(),
            checkpoint_manifest_sha256: target.checkpoint_manifest_sha256.clone(),
            training_evidence_sha256: target.training_evidence_sha256.clone(),
            training_gpu_hours: target.training_gpu_hours,
            parameters: target.parameters,
            routed_active_parameters: target.routed_active_parameters,
            stored_bytes: target.stored_bytes,
            representation: target.representation_identity(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceFixtureBinding {
    pub sha256: String,
    pub minimum_samples: usize,
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcePolicyBinding {
    pub evaluator_id: String,
    pub evaluator_sha256: String,
    pub minimum_wake_trials: usize,
    pub minimum_wake_latency_samples: usize,
    pub minimum_wake_throughput_ratio: f64,
    pub maximum_wake_latency_ratio: f64,
    pub grouped_mm: ResourceFixtureBinding,
    pub pytorch: ResourceFixtureBinding,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSemanticRequest {
    pub version: u32,
    pub selected_benchmark_run_sha256: String,
    pub comparison_run_sha256: Vec<String>,
    pub strongest_baseline_id: String,
    pub baseline: ResourceTargetIdentity,
    pub candidate: ResourceTargetIdentity,
    pub policy_sha256: String,
    pub policy: ResourcePolicyBinding,
    pub evaluator_arguments: Vec<String>,
    pub artifact_roots: Vec<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceArtifactTransport {
    pub relative_path: PathBuf,
    pub absolute_path: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceWorkerRequest {
    pub version: u32,
    pub semantic: ResourceSemanticRequest,
    pub baseline_transport: BenchmarkTarget,
    pub candidate_transport: BenchmarkTarget,
    pub artifact_transports: Vec<ResourceArtifactTransport>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceWorkerResponse {
    pub version: u32,
    pub wake_trials: Vec<PairedWakeTrial>,
    pub candidate_capacity: Vec<CapacityObservation>,
    pub grouped_mm_samples: Vec<KernelParitySample>,
    pub pytorch_samples: Vec<KernelParitySample>,
    pub exact_resume: ExactResumeEvidence,
}

#[derive(Clone, Debug)]
pub struct ResourceEvidencePublication {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug)]
pub struct ExternalResourceEvaluator {
    executable: PathBuf,
    arguments: Vec<String>,
    expected_sha256: String,
    timeout: Duration,
}

impl ExternalResourceEvaluator {
    pub fn new(
        executable: impl Into<PathBuf>,
        arguments: Vec<OsString>,
        expected_sha256: impl Into<String>,
    ) -> Result<Self> {
        let executable = executable.into();
        let arguments = arguments
            .into_iter()
            .map(|argument| {
                argument
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("resource evaluator argument is not valid UTF-8"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            arguments.len() <= 64
                && arguments
                    .iter()
                    .all(|argument| argument.len() <= 4096 && !argument.contains('\0')),
            "resource evaluator arguments exceed protocol limits"
        );
        let evaluator = Self {
            executable: validate_real_path(&executable, RealPathKind::File, "resource evaluator")?,
            arguments,
            expected_sha256: expected_sha256.into(),
            timeout: DEFAULT_RESOURCE_EVALUATOR_TIMEOUT,
        };
        evaluator.verify_identity()?;
        Ok(evaluator)
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        ensure!(
            !timeout.is_zero(),
            "resource evaluator timeout must be positive"
        );
        ensure!(
            Instant::now().checked_add(timeout).is_some(),
            "resource evaluator timeout is too large for the monotonic clock"
        );
        self.timeout = timeout;
        Ok(self)
    }

    pub fn expected_sha256(&self) -> &str {
        &self.expected_sha256
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    fn validate_path_identity(&self) -> Result<()> {
        validate_sha256_identity(&self.expected_sha256, "resource evaluator")?;
        let resolved =
            validate_real_path(&self.executable, RealPathKind::File, "resource evaluator")?;
        ensure!(
            resolved == self.executable,
            "resource evaluator path identity changed"
        );
        Ok(())
    }

    #[cfg(unix)]
    pub fn verify_identity(&self) -> Result<()> {
        self.validate_path_identity()?;
        PinnedExecutable::verify(
            &self.executable,
            &self.expected_sha256,
            "resource evaluator",
        )
    }

    #[cfg(not(unix))]
    pub fn verify_identity(&self) -> Result<()> {
        bail!("external resource evaluators require a Unix process host")
    }

    #[cfg(unix)]
    fn execute(&self, request: &ResourceWorkerRequest) -> Result<ResourceWorkerResponse> {
        let mut bytes = serde_json::to_vec(request)?;
        ensure!(
            bytes.len() < MAX_RESOURCE_MESSAGE_BYTES,
            "resource evaluator request exceeds {MAX_RESOURCE_MESSAGE_BYTES} bytes"
        );
        bytes.push(b'\n');
        // Walk every path ancestor immediately before opening the executable,
        // then hash that opened generation and execute a private
        // materialization of its exact bytes. Replacing the configured path
        // after this point cannot change the program selected for exec.
        self.validate_path_identity()?;
        let executable = PinnedExecutable::open(
            &self.executable,
            &self.expected_sha256,
            "resource evaluator",
            "resource-evaluator",
        )?;
        let mut command = executable.command();
        command
            .args(&self.arguments)
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Worker diagnostics are not evidence and must not become an
            // unbounded side channel or captured output.
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn().with_context(|| {
            format!(
                "failed to start resource evaluator {}",
                self.executable.display()
            )
        })?;
        let mut child = SupervisedProcess::new(
            child,
            executable.into_staged(),
            "resource evaluator",
            MAX_RESOURCE_MESSAGE_BYTES,
        )?;
        let deadline = Instant::now()
            .checked_add(self.timeout)
            .context("resource evaluator timeout exceeds the monotonic clock")?;
        let mut written = 0_usize;
        let mut response = None;
        loop {
            child.write_available(&bytes, &mut written)?;
            if written == bytes.len() {
                child.close_input();
            }
            drain_resource_output(&mut child, &mut response)?;
            if let Some(status) = child.try_wait()? {
                // Leader exit is a protocol boundary even if a descendant has
                // escaped the worker group with setsid() and retained stdout.
                // Drain every byte already committed by the leader, then close
                // our nonblocking pipes instead of joining an EOF waiter.
                child.terminate_process_group();
                drain_resource_output(&mut child, &mut response)?;
                child.finish_output_at_leader_exit()?;
                ensure!(
                    written == bytes.len(),
                    "resource evaluator exited before consuming its complete request"
                );
                ensure!(
                    status.success(),
                    "resource evaluator exited with status {status}"
                );
                return response.context("resource evaluator exited before responding");
            }
            if Instant::now() >= deadline {
                bail!(
                    "resource evaluator exceeded its {:?} execution timeout",
                    self.timeout
                );
            }
            child.wait_for_activity(written < bytes.len(), deadline)?;
        }
    }

    #[cfg(not(unix))]
    fn execute(&self, _request: &ResourceWorkerRequest) -> Result<ResourceWorkerResponse> {
        bail!("external resource evaluators require a Unix process host")
    }
}

#[cfg(unix)]
fn drain_resource_output(
    child: &mut SupervisedProcess,
    response: &mut Option<ResourceWorkerResponse>,
) -> Result<()> {
    loop {
        match child.read_line()? {
            ProtocolRead::Line(line) => {
                ensure!(
                    response.is_none(),
                    "resource evaluator emitted output after its response"
                );
                *response = Some(parse_resource_response(&line)?);
            }
            ProtocolRead::Pending | ProtocolRead::Eof => return Ok(()),
        }
    }
}

#[derive(Clone, Copy)]
enum RealPathKind {
    File,
    Directory,
}

/// Resolve a path lexically and reject every symlink in the supplied path,
/// rather than accepting a canonical path that silently traversed one.
fn validate_real_path(path: &Path, kind: RealPathKind, label: &str) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "{label} path is empty");
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .with_context(|| format!("resolving current directory for {label}"))?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => bail!("{label} path must not contain `..`"),
        }
    }
    ensure!(
        normalized.is_absolute(),
        "{label} path did not resolve absolutely"
    );

    let mut cursor = PathBuf::new();
    let component_count = normalized.components().count();
    for (index, component) in normalized.components().enumerate() {
        cursor.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&cursor)
            .with_context(|| format!("inspecting {label} ancestor {}", cursor.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "{label} path traverses symlink {}",
            cursor.display()
        );
        let final_component = index + 1 == component_count;
        if final_component {
            match kind {
                RealPathKind::File => ensure!(
                    metadata.file_type().is_file(),
                    "{label} {} must be a regular file",
                    cursor.display()
                ),
                RealPathKind::Directory => ensure!(
                    metadata.file_type().is_dir(),
                    "{label} {} must be a directory",
                    cursor.display()
                ),
            }
        } else {
            ensure!(
                metadata.file_type().is_dir(),
                "{label} ancestor {} must be a directory",
                cursor.display()
            );
        }
    }
    ensure!(
        normalized.canonicalize()? == normalized,
        "{label} path does not have a stable canonical identity"
    );
    Ok(normalized)
}

fn stable_file_bytes(path: &Path, maximum_bytes: u64, label: &str) -> Result<Vec<u8>> {
    validate_real_path(path, RealPathKind::File, label)?;
    let before = fs::symlink_metadata(path)?;
    let mut file =
        File::open(path).with_context(|| format!("opening {label} {}", path.display()))?;
    let opened = file.metadata()?;
    ensure_same_file(&before, &opened, label)?;
    ensure!(
        opened.len() <= maximum_bytes,
        "{label} exceeds its {maximum_bytes}-byte limit"
    );
    let capacity = usize::try_from(opened.len())
        .with_context(|| format!("{label} exceeds this process address space"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .with_context(|| format!("reserving bounded buffer for {label}"))?;
    file.read_to_end(&mut bytes)?;
    let after = fs::symlink_metadata(path)?;
    ensure!(
        after.file_type().is_file() && !after.file_type().is_symlink(),
        "{label} became a symlink or non-file while reading"
    );
    ensure_same_file(&after, &opened, label)?;
    ensure!(
        opened.len() == bytes.len() as u64,
        "{label} changed length while reading"
    );
    Ok(bytes)
}

#[cfg(unix)]
fn ensure_same_file(left: &fs::Metadata, right: &fs::Metadata, label: &str) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    ensure!(
        left.dev() == right.dev() && left.ino() == right.ino(),
        "{label} changed identity while open"
    );
    Ok(())
}

#[cfg(not(unix))]
fn ensure_same_file(left: &fs::Metadata, right: &fs::Metadata, label: &str) -> Result<()> {
    ensure!(
        left.file_type().is_file() && right.file_type().is_file(),
        "{label} changed type while open"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_resource_benchmark(
    selected_run: &VerifiedBenchmarkRun,
    comparison_runs: &[VerifiedBenchmarkRun],
    policy: &AcceptancePolicy,
    policy_sha256: &str,
    evaluator: &ExternalResourceEvaluator,
    artifact_roots: &[PathBuf],
    output_directory: &Path,
) -> Result<ResourceEvidencePublication> {
    policy.validate()?;
    validate_sha256_hex(policy_sha256, "resource policy")?;
    ensure!(
        evaluator.expected_sha256() == policy.resource_evaluator_version,
        "pinned resource evaluator differs from acceptance policy"
    );
    let strongest_baseline_id = verified_resource_benchmark_context(selected_run, comparison_runs)?;
    let vault = prepare_existing_vault(output_directory)?;
    let (relative_roots, transports) = prepare_artifact_roots(&vault, artifact_roots)?;
    let semantic = semantic_request(
        selected_run,
        comparison_runs,
        policy,
        policy_sha256,
        &strongest_baseline_id,
        evaluator.arguments().to_vec(),
        relative_roots,
    )?;
    let request_sha256 = json_sha256_identity(&semantic)?;
    let request = ResourceWorkerRequest {
        version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
        semantic: semantic.clone(),
        baseline_transport: selected_run.run().metadata.baseline.clone(),
        candidate_transport: selected_run.run().metadata.candidate.clone(),
        artifact_transports: transports,
    };
    let response = evaluator.execute(&request)?;
    validate_response_artifact_scope(&response, &vault, &semantic.artifact_roots)?;

    let baseline_target_sha256 = json_sha256_identity(&semantic.baseline)?;
    let candidate_target_sha256 = json_sha256_identity(&semantic.candidate)?;
    let mut comparison = ResourceComparison {
        version: RESOURCE_COMPARISON_VERSION,
        baseline_id: semantic.baseline.id.clone(),
        candidate_id: semantic.candidate.id.clone(),
        benchmark_run_sha256: semantic.selected_benchmark_run_sha256.clone(),
        strongest_baseline_id: strongest_baseline_id.clone(),
        measurement_evaluator_id: semantic.policy.evaluator_id.clone(),
        measurement_evaluator_version: evaluator.expected_sha256().to_owned(),
        wake_trials: response.wake_trials,
        candidate_capacity: response.candidate_capacity,
        grouped_mm_parity: KernelParityEvidence {
            fixture_sha256: semantic.policy.grouped_mm.sha256.clone(),
            samples: response.grouped_mm_samples,
        },
        pytorch_parity: KernelParityEvidence {
            fixture_sha256: semantic.policy.pytorch.sha256.clone(),
            samples: response.pytorch_samples,
        },
        exact_resume: response.exact_resume,
        execution: ResourceExecutionReceipt {
            protocol_version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
            evaluator_sha256: evaluator.expected_sha256().to_owned(),
            request_sha256,
            observations_sha256: String::new(),
            baseline_target_sha256,
            candidate_target_sha256,
            policy_sha256: policy_sha256.to_owned(),
            evaluator_arguments: evaluator.arguments().to_vec(),
            approved_artifact_roots: semantic.artifact_roots,
        },
    };
    comparison.execution.observations_sha256 = comparison.observations_sha256()?;
    comparison.validate()?;

    let bytes = pretty_json_bytes(&comparison)?;
    let digest = sha256_hex(&bytes);
    let target = vault.join(format!("sha256-{digest}.json"));
    validate_execution_receipt(
        selected_run,
        comparison_runs,
        policy,
        policy_sha256,
        &strongest_baseline_id,
        &comparison,
        &target,
    )?;
    verify_resource_comparison_artifacts(&comparison, &target)?;
    publish_immutable(&target, &bytes)?;

    let verified = VerifiedResourceComparison::load(&target, &digest)?;
    validate_execution_receipt(
        selected_run,
        comparison_runs,
        policy,
        policy_sha256,
        &strongest_baseline_id,
        verified.comparison(),
        verified.path(),
    )?;
    Ok(ResourceEvidencePublication {
        path: target,
        sha256: digest,
    })
}

pub(crate) fn validate_execution_receipt(
    selected_run: &VerifiedBenchmarkRun,
    comparison_runs: &[VerifiedBenchmarkRun],
    policy: &AcceptancePolicy,
    policy_sha256: &str,
    strongest_baseline_id: &str,
    comparison: &ResourceComparison,
    source_path: &Path,
) -> Result<()> {
    policy.validate()?;
    validate_sha256_hex(policy_sha256, "resource policy")?;
    ensure!(
        comparison
            .execution
            .policy_sha256
            .eq_ignore_ascii_case(policy_sha256),
        "resource evidence was executed under another acceptance policy"
    );
    ensure!(
        comparison.execution.evaluator_sha256 == policy.resource_evaluator_version,
        "resource execution receipt names another evaluator"
    );
    let semantic = semantic_request(
        selected_run,
        comparison_runs,
        policy,
        policy_sha256,
        strongest_baseline_id,
        comparison.execution.evaluator_arguments.clone(),
        comparison.execution.approved_artifact_roots.clone(),
    )?;
    ensure!(
        comparison.execution.request_sha256 == json_sha256_identity(&semantic)?,
        "resource execution request receipt does not match the verified experiment"
    );
    ensure!(
        comparison.execution.baseline_target_sha256 == json_sha256_identity(&semantic.baseline)?
            && comparison.execution.candidate_target_sha256
                == json_sha256_identity(&semantic.candidate)?,
        "resource execution receipt addresses different benchmark targets"
    );
    ensure!(
        comparison.execution.observations_sha256 == comparison.observations_sha256()?,
        "resource execution receipt does not match the retained raw observations"
    );
    let vault = source_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    validate_exact_artifact_scope(
        &comparison.exact_resume,
        vault,
        &comparison.execution.approved_artifact_roots,
    )
}

/// Derive the host-owned receipt used by tests and alternate first-party
/// frontends after they have collected raw observations. The worker never
/// gets to provide any of these identities.
#[cfg(test)]
pub(crate) fn derive_execution_receipt(
    selected_run: &VerifiedBenchmarkRun,
    comparison_runs: &[VerifiedBenchmarkRun],
    policy: &AcceptancePolicy,
    policy_sha256: &str,
    evaluator_arguments: Vec<String>,
    artifact_roots: Vec<PathBuf>,
    comparison: &ResourceComparison,
) -> Result<ResourceExecutionReceipt> {
    policy.validate()?;
    validate_sha256_hex(policy_sha256, "resource policy")?;
    let strongest_baseline_id = verified_resource_benchmark_context(selected_run, comparison_runs)?;
    let semantic = semantic_request(
        selected_run,
        comparison_runs,
        policy,
        policy_sha256,
        &strongest_baseline_id,
        evaluator_arguments.clone(),
        artifact_roots.clone(),
    )?;
    Ok(ResourceExecutionReceipt {
        protocol_version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
        evaluator_sha256: policy.resource_evaluator_version.clone(),
        request_sha256: json_sha256_identity(&semantic)?,
        observations_sha256: comparison.observations_sha256()?,
        baseline_target_sha256: json_sha256_identity(&semantic.baseline)?,
        candidate_target_sha256: json_sha256_identity(&semantic.candidate)?,
        policy_sha256: policy_sha256.to_owned(),
        evaluator_arguments,
        approved_artifact_roots: artifact_roots,
    })
}

fn semantic_request(
    selected_run: &VerifiedBenchmarkRun,
    comparison_runs: &[VerifiedBenchmarkRun],
    policy: &AcceptancePolicy,
    policy_sha256: &str,
    strongest_baseline_id: &str,
    evaluator_arguments: Vec<String>,
    mut artifact_roots: Vec<PathBuf>,
) -> Result<ResourceSemanticRequest> {
    artifact_roots.sort();
    for root in &artifact_roots {
        validate_safe_relative(root, "resource artifact root")?;
    }
    let mut comparison_run_sha256 = comparison_runs
        .iter()
        .map(|run| run.sha256().to_owned())
        .collect::<Vec<_>>();
    comparison_run_sha256.sort();
    ensure!(
        comparison_run_sha256
            .windows(2)
            .all(|pair| pair[0] != pair[1]),
        "resource comparison set repeats a benchmark-run identity"
    );
    Ok(ResourceSemanticRequest {
        version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
        selected_benchmark_run_sha256: selected_run.sha256().to_owned(),
        comparison_run_sha256,
        strongest_baseline_id: strongest_baseline_id.to_owned(),
        baseline: ResourceTargetIdentity::from(&selected_run.run().metadata.baseline),
        candidate: ResourceTargetIdentity::from(&selected_run.run().metadata.candidate),
        policy_sha256: policy_sha256.to_owned(),
        policy: ResourcePolicyBinding {
            evaluator_id: policy.resource_evaluator_id.clone(),
            evaluator_sha256: policy.resource_evaluator_version.clone(),
            minimum_wake_trials: policy.minimum_wake_trials,
            minimum_wake_latency_samples: policy.minimum_wake_latency_samples,
            minimum_wake_throughput_ratio: policy.minimum_wake_throughput_ratio,
            maximum_wake_latency_ratio: policy.maximum_wake_latency_ratio,
            grouped_mm: ResourceFixtureBinding {
                sha256: policy.grouped_mm_parity.fixture_sha256.clone(),
                minimum_samples: policy.grouped_mm_parity.minimum_samples,
                maximum_absolute_error: policy.grouped_mm_parity.maximum_absolute_error,
                maximum_relative_error: policy.grouped_mm_parity.maximum_relative_error,
            },
            pytorch: ResourceFixtureBinding {
                sha256: policy.pytorch_parity.fixture_sha256.clone(),
                minimum_samples: policy.pytorch_parity.minimum_samples,
                maximum_absolute_error: policy.pytorch_parity.maximum_absolute_error,
                maximum_relative_error: policy.pytorch_parity.maximum_relative_error,
            },
        },
        evaluator_arguments,
        artifact_roots,
    })
}

fn prepare_existing_vault(path: &Path) -> Result<PathBuf> {
    validate_real_path(path, RealPathKind::Directory, "resource evidence vault")
}

fn prepare_artifact_roots(
    vault: &Path,
    roots: &[PathBuf],
) -> Result<(Vec<PathBuf>, Vec<ResourceArtifactTransport>)> {
    ensure!(!roots.is_empty(), "resource benchmark has no artifact root");
    let mut relative = roots.to_vec();
    relative.sort();
    ensure!(
        relative.windows(2).all(|pair| pair[0] != pair[1]),
        "resource benchmark repeats an artifact root"
    );
    ensure!(
        relative.iter().enumerate().all(|(index, root)| {
            relative[index + 1..]
                .iter()
                .all(|other| !root.starts_with(other) && !other.starts_with(root))
        }),
        "resource benchmark artifact roots must not overlap"
    );
    let mut transports = Vec::with_capacity(relative.len());
    for root in &relative {
        validate_safe_relative(root, "resource artifact root")?;
        let absolute = create_safe_subdirectory(vault, root)?;
        transports.push(ResourceArtifactTransport {
            relative_path: root.clone(),
            absolute_path: absolute,
        });
    }
    Ok((relative, transports))
}

fn create_safe_subdirectory(vault: &Path, relative: &Path) -> Result<PathBuf> {
    let mut directory = vault.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            bail!("resource artifact root contains an unsafe path component")
        };
        directory.push(component);
        match fs::create_dir(&directory) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("creating resource artifact root {}", directory.display())
                });
            }
        }
        let metadata = fs::symlink_metadata(&directory)?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "resource artifact root {} became a symlink",
            directory.display()
        );
    }
    Ok(directory)
}

fn validate_response_artifact_scope(
    response: &ResourceWorkerResponse,
    vault: &Path,
    roots: &[PathBuf],
) -> Result<()> {
    validate_exact_artifact_scope(&response.exact_resume, vault, roots)
}

fn validate_exact_artifact_scope(
    exact: &ExactResumeEvidence,
    vault: &Path,
    roots: &[PathBuf],
) -> Result<()> {
    let canonical_vault = vault
        .canonicalize()
        .with_context(|| format!("canonicalizing resource artifact vault {}", vault.display()))?;
    for artifact in exact_artifacts(exact) {
        validate_safe_relative(&artifact.path, "exact-resume artifact path")?;
        ensure!(
            roots.iter().any(|root| artifact.path.starts_with(root)),
            "exact-resume artifact {} is outside the approved output roots",
            artifact.path.display()
        );
        let joined = canonical_vault.join(&artifact.path);
        let canonical = joined.canonicalize().with_context(|| {
            format!("canonicalizing exact-resume artifact {}", joined.display())
        })?;
        ensure!(
            canonical == joined && canonical.starts_with(&canonical_vault),
            "exact-resume artifact {} traverses or uses a symlink",
            artifact.path.display()
        );
        let metadata = fs::symlink_metadata(&joined)?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "exact-resume artifact {} must be a regular non-symlink file",
            artifact.path.display()
        );
    }
    Ok(())
}

fn exact_artifacts(exact: &ExactResumeEvidence) -> [&ExactResumeArtifact; 5] {
    [
        &exact.interrupted_checkpoint,
        &exact.uninterrupted_final_state,
        &exact.resumed_final_state,
        &exact.uninterrupted_metrics,
        &exact.resumed_metrics,
    ]
}

fn validate_safe_relative(path: &Path, name: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{name} is empty");
    ensure!(!path.is_absolute(), "{name} must be relative");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{name} must not contain prefixes, `.` or `..`"
    );
    Ok(())
}

#[cfg(test)]
fn read_resource_response(stdout: impl Read) -> Result<ResourceWorkerResponse> {
    let mut reader = BufReader::new(stdout);
    let line =
        read_bounded_line(&mut reader)?.context("resource evaluator exited before responding")?;
    ensure!(
        read_bounded_line(&mut reader)?.is_none(),
        "resource evaluator emitted output after its response"
    );
    parse_resource_response(&line)
}

fn parse_resource_response(line: &[u8]) -> Result<ResourceWorkerResponse> {
    let response: ResourceWorkerResponse =
        serde_json::from_slice(line).context("resource evaluator emitted invalid protocol JSON")?;
    ensure!(
        response.version == RESOURCE_EXECUTION_PROTOCOL_VERSION,
        "unsupported resource evaluator response version {}",
        response.version
    );
    Ok(response)
}

#[cfg(test)]
fn read_bounded_line(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            ensure!(
                line.is_empty(),
                "resource evaluator response is unterminated"
            );
            return Ok(None);
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        ensure!(
            line.len() + take <= MAX_RESOURCE_MESSAGE_BYTES,
            "resource evaluator response exceeds {MAX_RESOURCE_MESSAGE_BYTES} bytes"
        );
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.ends_with(b"\n") {
            return Ok(Some(line));
        }
    }
}

fn pretty_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    let expected_bytes =
        u64::try_from(bytes.len()).context("resource evidence length exceeds u64")?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let verified_parent = validate_real_path(
        parent,
        RealPathKind::Directory,
        "resource publication parent",
    )?;
    ensure!(
        verified_parent == parent,
        "resource publication parent changed identity"
    );
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "content-addressed resource evidence must be a regular non-symlink file"
            );
            ensure!(
                stable_file_bytes(path, expected_bytes, "content-addressed resource evidence",)?
                    == bytes,
                "content-addressed resource evidence already exists with different bytes"
            );
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspecting resource evidence publication"),
    }
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".resource-evidence.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let result = (|| -> Result<()> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        output.write_all(bytes)?;
        output.sync_all()?;
        drop(output);
        match fs::hard_link(&temporary, path) {
            Ok(()) => {
                ensure!(
                    stable_file_bytes(path, expected_bytes, "published resource evidence")?
                        == bytes,
                    "published resource evidence changed during publication"
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure!(
                    stable_file_bytes(path, expected_bytes, "concurrent resource evidence")?
                        == bytes,
                    "resource evidence publication race produced different bytes"
                );
            }
            Err(error) => return Err(error).context("publishing resource evidence"),
        }
        fs::remove_file(&temporary)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::acceptance::{RESOURCE_EXECUTION_PROTOCOL_VERSION, WakeMeasurement};

    fn target(id: &str) -> BenchmarkTarget {
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest: format!("/{id}/checkpoint.json").into(),
            checkpoint_manifest_sha256: "1".repeat(64),
            training_evidence: format!("/{id}/training.json").into(),
            training_evidence_sha256: "2".repeat(64),
            training_gpu_hours: 1.0,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: 400,
            representation: crate::benchmark::ModelRepresentationTarget::FullPrecision,
        }
    }

    fn exact_resume(path: impl Into<PathBuf>) -> ExactResumeEvidence {
        let path = path.into();
        let artifact = |sha256: char| ExactResumeArtifact {
            path: path.clone(),
            sha256: sha256.to_string().repeat(64),
        };
        ExactResumeEvidence {
            interrupted_checkpoint: artifact('1'),
            uninterrupted_final_state: artifact('2'),
            resumed_final_state: artifact('2'),
            uninterrupted_metrics: artifact('3'),
            resumed_metrics: artifact('4'),
            interruption_step: 1,
            resumed_from_step: 1,
        }
    }

    fn response() -> ResourceWorkerResponse {
        ResourceWorkerResponse {
            version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
            wake_trials: vec![PairedWakeTrial {
                trial: 0,
                baseline: WakeMeasurement {
                    tokens: 1,
                    elapsed_seconds: 1.0,
                    request_latency_ms: vec![1.0],
                },
                candidate: WakeMeasurement {
                    tokens: 1,
                    elapsed_seconds: 1.0,
                    request_latency_ms: vec![1.0],
                },
            }],
            candidate_capacity: vec![CapacityObservation {
                completed_sleep_cycles: 0,
                routed_active_parameters: 80,
                stored_parameters: 100,
                stored_bytes: 400,
            }],
            grouped_mm_samples: vec![KernelParitySample {
                reference: 1.0,
                candidate: 1.0,
            }],
            pytorch_samples: vec![KernelParitySample {
                reference: 1.0,
                candidate: 1.0,
            }],
            exact_resume: exact_resume("artifacts/exact.json"),
        }
    }

    fn request(arguments: Vec<String>) -> ResourceWorkerRequest {
        ResourceWorkerRequest {
            version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
            semantic: ResourceSemanticRequest {
                version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
                selected_benchmark_run_sha256: "3".repeat(64),
                comparison_run_sha256: vec!["3".repeat(64)],
                strongest_baseline_id: "baseline".into(),
                baseline: ResourceTargetIdentity::from(&target("baseline")),
                candidate: ResourceTargetIdentity::from(&target("candidate")),
                policy_sha256: "4".repeat(64),
                policy: ResourcePolicyBinding {
                    evaluator_id: "resource-test".into(),
                    evaluator_sha256: format!("sha256:{}", "5".repeat(64)),
                    minimum_wake_trials: 1,
                    minimum_wake_latency_samples: 1,
                    minimum_wake_throughput_ratio: 0.95,
                    maximum_wake_latency_ratio: 1.05,
                    grouped_mm: ResourceFixtureBinding {
                        sha256: "6".repeat(64),
                        minimum_samples: 1,
                        maximum_absolute_error: 1e-5,
                        maximum_relative_error: 1e-4,
                    },
                    pytorch: ResourceFixtureBinding {
                        sha256: "7".repeat(64),
                        minimum_samples: 1,
                        maximum_absolute_error: 1e-5,
                        maximum_relative_error: 1e-4,
                    },
                },
                evaluator_arguments: arguments,
                artifact_roots: vec!["artifacts".into()],
            },
            baseline_transport: target("baseline"),
            candidate_transport: target("candidate"),
            artifact_transports: vec![ResourceArtifactTransport {
                relative_path: "artifacts".into(),
                absolute_path: "/vault/artifacts".into(),
            }],
        }
    }

    #[cfg(unix)]
    fn executable(root: &Path, name: &str, source: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = root.join(name);
        fs::write(&path, source).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[cfg(unix)]
    fn python_with_setsid() -> PathBuf {
        [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
        .expect("setsid worker tests require an absolute Python 3 interpreter")
    }

    #[cfg(unix)]
    fn kill_detached_pid(path: &Path) {
        for _ in 0..100 {
            if let Ok(value) = fs::read_to_string(path)
                && let Ok(pid) = value.trim().parse::<i32>()
            {
                // SAFETY: the test helper wrote its own positive process ID.
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("detached worker did not publish its pid");
    }

    #[test]
    fn immutable_resource_comparison_rejects_oversized_existing_file_before_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let path = root.join("oversized.json");
        let file = File::create(&path).unwrap();
        file.set_len(1_025).unwrap();
        drop(file);

        let error = stable_file_bytes(&path, 1_024, "resource fixture")
            .unwrap_err()
            .to_string();
        assert!(error.contains("1024-byte limit"), "{error}");
    }

    #[test]
    fn evaluator_arguments_are_part_of_the_semantic_request_identity() {
        let first = request(vec!["--mode=a".into()]);
        let second = request(vec!["--mode=b".into()]);
        assert_ne!(
            json_sha256_identity(&first.semantic).unwrap(),
            json_sha256_identity(&second.semantic).unwrap()
        );
    }

    #[test]
    fn response_reader_rejects_unterminated_and_oversized_messages() {
        let error = read_resource_response(Cursor::new(b"{}".to_vec()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unterminated"), "{error}");

        let oversized = vec![b'x'; MAX_RESOURCE_MESSAGE_BYTES + 1];
        let error = read_bounded_line(&mut Cursor::new(oversized))
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds"), "{error}");
    }

    #[test]
    fn evaluator_rejects_a_timeout_outside_the_monotonic_clock_range() {
        let evaluator = ExternalResourceEvaluator {
            executable: PathBuf::from("unused-test-worker"),
            arguments: Vec::new(),
            expected_sha256: format!("sha256:{}", "0".repeat(64)),
            timeout: DEFAULT_RESOURCE_EVALUATOR_TIMEOUT,
        };

        let error = evaluator
            .with_timeout(Duration::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("monotonic clock"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn hanging_evaluator_is_killed_at_the_bound() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let worker = executable(
            &root,
            "hang.sh",
            "#!/bin/sh\nIFS= read -r request\n/bin/sleep 30\n",
        );
        let digest = pinned_file_sha256(&worker).unwrap();
        let evaluator = ExternalResourceEvaluator::new(&worker, Vec::new(), digest)
            .unwrap()
            .with_timeout(Duration::from_millis(100))
            .unwrap();
        let started = Instant::now();
        let error = evaluator
            .execute(&request(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("timeout"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[test]
    fn timeout_is_bounded_when_setsid_descendant_retains_protocol_pipes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let detached_pid = root.join("detached.pid");
        let worker = executable(
            &root,
            "setsid-timeout.sh",
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                "/bin/sleep 30\n"
            ),
        );
        let digest = pinned_file_sha256(&worker).unwrap();
        let evaluator = ExternalResourceEvaluator::new(
            &worker,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
            digest,
        )
        .unwrap()
        .with_timeout(Duration::from_secs(3))
        .unwrap();

        let started = Instant::now();
        let result = evaluator.execute(&request(Vec::new()));
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("timeout"), "{error}");
        assert!(
            elapsed < Duration::from_secs(5),
            "setsid descendant delayed resource timeout for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn protocol_error_is_bounded_when_setsid_descendant_retains_stdout() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let detached_pid = root.join("detached.pid");
        let worker = executable(
            &root,
            "setsid-error.sh",
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                "printf '{not-json}\\n'\n",
                "/bin/sleep 30\n"
            ),
        );
        let digest = pinned_file_sha256(&worker).unwrap();
        let evaluator = ExternalResourceEvaluator::new(
            &worker,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
            digest,
        )
        .unwrap()
        .with_timeout(Duration::from_secs(3))
        .unwrap();

        let started = Instant::now();
        let result = evaluator.execute(&request(Vec::new()));
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("invalid protocol JSON"), "{error}");
        assert!(
            elapsed < Duration::from_secs(2),
            "setsid descendant delayed resource protocol error for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn evaluator_rejects_output_after_the_one_response() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let response_path = root.join("response.jsonl");
        let mut response_bytes = serde_json::to_vec(&response()).unwrap();
        response_bytes.push(b'\n');
        fs::write(&response_path, response_bytes).unwrap();
        let worker = executable(
            &root,
            "extra.sh",
            &format!(
                "#!/bin/sh\nIFS= read -r request\n/bin/cat '{}'\nprintf '{{}}\\n'\n",
                response_path.display()
            ),
        );
        let digest = pinned_file_sha256(&worker).unwrap();
        let evaluator = ExternalResourceEvaluator::new(&worker, Vec::new(), digest).unwrap();
        let error = evaluator
            .execute(&request(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("after its response"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn exited_evaluator_cannot_leave_a_descendant_holding_protocol_pipes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let response_path = root.join("response.jsonl");
        let mut response_bytes = serde_json::to_vec(&response()).unwrap();
        response_bytes.push(b'\n');
        fs::write(&response_path, response_bytes).unwrap();
        let worker = executable(
            &root,
            "orphan.sh",
            &format!(
                "#!/bin/sh\nIFS= read -r request\n/bin/cat '{}'\n/bin/sleep 30 &\nexit 0\n",
                response_path.display()
            ),
        );
        let digest = pinned_file_sha256(&worker).unwrap();
        let evaluator = ExternalResourceEvaluator::new(&worker, Vec::new(), digest)
            .unwrap()
            // Leave enough scheduling margin when all process-lifecycle tests
            // execute concurrently; the elapsed-time assertion still catches
            // a descendant retaining the pipe for its 30-second lifetime.
            .with_timeout(Duration::from_secs(3))
            .unwrap();

        let started = Instant::now();
        evaluator.execute(&request(Vec::new())).unwrap();
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[cfg(unix)]
    #[test]
    fn leader_exit_is_bounded_when_setsid_descendant_retains_stdout() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let response_path = root.join("response.jsonl");
        let detached_pid = root.join("detached.pid");
        let mut response_bytes = serde_json::to_vec(&response()).unwrap();
        response_bytes.push(b'\n');
        fs::write(&response_path, response_bytes).unwrap();
        let worker = executable(
            &root,
            "setsid-exit.sh",
            &format!(
                concat!(
                    "#!/bin/sh\n",
                    "IFS= read -r request\n",
                    "test -n \"$request\"\n",
                    "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                    "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                    "/bin/cat '{}'\n"
                ),
                response_path.display()
            ),
        );
        let digest = pinned_file_sha256(&worker).unwrap();
        let evaluator = ExternalResourceEvaluator::new(
            &worker,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
            digest,
        )
        .unwrap()
        .with_timeout(Duration::from_secs(2))
        .unwrap();

        let started = Instant::now();
        let result = evaluator.execute(&request(Vec::new()));
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        result.unwrap();
        assert!(
            elapsed < Duration::from_secs(3),
            "setsid descendant delayed resource leader exit for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn evaluator_rejects_direct_and_ancestor_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let real = root.join("real");
        fs::create_dir(&real).unwrap();
        let worker = executable(&real, "worker.sh", "#!/bin/sh\nexit 0\n");
        let digest = pinned_file_sha256(&worker).unwrap();
        let direct = root.join("worker-link");
        symlink(&worker, &direct).unwrap();
        assert!(ExternalResourceEvaluator::new(&direct, Vec::new(), &digest).is_err());

        let linked_parent = root.join("linked-parent");
        symlink(&real, &linked_parent).unwrap();
        let error =
            ExternalResourceEvaluator::new(linked_parent.join("worker.sh"), Vec::new(), digest)
                .unwrap_err()
                .to_string();
        assert!(error.contains("traverses symlink"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn artifact_scope_rejects_traversal_absolute_and_symlink_paths() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let vault = temporary.path().canonicalize().unwrap();
        fs::create_dir(vault.join("approved")).unwrap();
        fs::write(vault.join("outside"), b"outside").unwrap();
        for path in [PathBuf::from("../outside"), vault.join("outside")] {
            let error =
                validate_exact_artifact_scope(&exact_resume(path), &vault, &["approved".into()])
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("relative") || error.contains("must not contain"),
                "{error}"
            );
        }
        symlink(vault.join("outside"), vault.join("approved/link")).unwrap();
        let error = validate_exact_artifact_scope(
            &exact_resume("approved/link"),
            &vault,
            &["approved".into()],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("symlink") || error.contains("traverses"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn immutable_publication_rejects_a_preexisting_symlink() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let victim = root.join("victim");
        fs::write(&victim, b"victim").unwrap();
        let target = root.join("sha256-evidence.json");
        symlink(&victim, &target).unwrap();
        let error = publish_immutable(&target, b"evidence")
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-symlink"), "{error}");
        assert_eq!(fs::read(victim).unwrap(), b"victim");
    }
}
