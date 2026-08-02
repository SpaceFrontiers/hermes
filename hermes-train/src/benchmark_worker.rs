//! Content-pinned JSONL bridge for benchmark evaluators.
//!
//! Benchmark orchestration verifies suites and model manifests in-process. A
//! long-lived local worker performs model-specific evaluation without making
//! the benchmark framework depend on a particular generation or retrieval
//! implementation. The executable is hashed before it starts, requests are
//! strictly versioned, and every response is validated by `BenchmarkRunner`.

use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::acceptance::SuiteVisibility;
use crate::benchmark::{
    BenchmarkEvaluator, BenchmarkSpec, BenchmarkTarget, EvaluationMeasurement, EvaluationRequest,
    TargetRole, VerifiedArtifact, VerifiedModelRepresentation,
};

pub const BENCHMARK_WORKER_PROTOCOL_VERSION: u32 = 2;
const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkWorkerRequest<'a> {
    pub version: u32,
    pub evaluator_sha256: &'a str,
    pub suite_id: &'a str,
    pub visibility: SuiteVisibility,
    pub case: &'a BenchmarkSpec,
    pub artifact: &'a VerifiedArtifact,
    pub target: &'a BenchmarkTarget,
    pub model: &'a VerifiedModelRepresentation,
    pub role: TargetRole,
    pub model_seed: u64,
    pub example_order_seed: u64,
    pub pair_ordinal: usize,
    pub max_gpu_hours: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkWorkerResponse {
    version: u32,
    measurement: EvaluationMeasurement,
}

struct RunningWorker {
    child: Child,
    input: Option<BufWriter<ChildStdin>>,
    output: BufReader<ChildStdout>,
}

/// Persistent evaluator process used for a complete paired benchmark run.
/// No shell is involved and the worker receives only already-verified local
/// artifact paths and immutable target metadata.
pub struct ExternalBenchmarkEvaluator {
    executable: PathBuf,
    arguments: Vec<OsString>,
    expected_sha256: String,
    running: Option<RunningWorker>,
}

impl ExternalBenchmarkEvaluator {
    pub fn new(
        executable: impl Into<PathBuf>,
        arguments: Vec<OsString>,
        expected_sha256: impl Into<String>,
    ) -> Result<Self> {
        let evaluator = Self {
            executable: executable.into(),
            arguments,
            expected_sha256: expected_sha256.into(),
            running: None,
        };
        evaluator.verify_identity()?;
        Ok(evaluator)
    }

    pub fn expected_sha256(&self) -> &str {
        &self.expected_sha256
    }

    pub fn verify_identity(&self) -> Result<()> {
        validate_sha256(&self.expected_sha256)?;
        let metadata = std::fs::symlink_metadata(&self.executable).with_context(|| {
            format!(
                "benchmark evaluator {} is unavailable",
                self.executable.display()
            )
        })?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "benchmark evaluator {} must be a regular non-symlink file",
            self.executable.display()
        );
        ensure!(
            file_sha256(&self.executable)? == self.expected_sha256,
            "benchmark evaluator {} does not match its pinned SHA-256",
            self.executable.display()
        );
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        if self.running.is_some() {
            return Ok(());
        }
        self.verify_identity()?;
        let mut child = Command::new(&self.executable)
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start benchmark evaluator {}",
                    self.executable.display()
                )
            })?;
        let input = child
            .stdin
            .take()
            .context("benchmark evaluator stdin is unavailable")?;
        let output = child
            .stdout
            .take()
            .context("benchmark evaluator stdout is unavailable")?;
        self.running = Some(RunningWorker {
            child,
            input: Some(BufWriter::new(input)),
            output: BufReader::new(output),
        });
        Ok(())
    }

    /// Close the protocol cleanly and require a successful worker exit. This
    /// must be called after `BenchmarkRunner::run`; `Drop` only provides
    /// best-effort cleanup on an earlier error.
    pub fn finish(mut self) -> Result<()> {
        let Some(mut running) = self.running.take() else {
            return Ok(());
        };
        if let Some(mut input) = running.input.take() {
            input.flush()?;
            drop(input);
        }
        ensure!(
            read_bounded_line(&mut running.output)?.is_none(),
            "benchmark evaluator emitted an unsolicited response"
        );
        let status = running.child.wait()?;
        ensure!(
            status.success(),
            "benchmark evaluator exited with status {status}"
        );
        Ok(())
    }
}

impl BenchmarkEvaluator for ExternalBenchmarkEvaluator {
    fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
        self.start()?;
        let running = self
            .running
            .as_mut()
            .expect("benchmark worker was started above");
        let wire = BenchmarkWorkerRequest {
            version: BENCHMARK_WORKER_PROTOCOL_VERSION,
            evaluator_sha256: &self.expected_sha256,
            suite_id: request.suite_id,
            visibility: request.visibility,
            case: request.case,
            artifact: request.artifact,
            target: request.target,
            model: request.model,
            role: request.role,
            model_seed: request.model_seed,
            example_order_seed: request.example_order_seed,
            pair_ordinal: request.pair_ordinal,
            max_gpu_hours: request.max_gpu_hours,
        };
        let input = running
            .input
            .as_mut()
            .context("benchmark evaluator input is closed")?;
        serde_json::to_writer(&mut *input, &wire)?;
        input.write_all(b"\n")?;
        input.flush()?;

        let line = read_bounded_line(&mut running.output)?
            .context("benchmark evaluator exited before responding")?;
        let response: BenchmarkWorkerResponse = serde_json::from_slice(&line)
            .context("benchmark evaluator emitted invalid protocol JSON")?;
        ensure!(
            response.version == BENCHMARK_WORKER_PROTOCOL_VERSION,
            "unsupported benchmark evaluator response version {}",
            response.version
        );
        Ok(response.measurement)
    }
}

impl Drop for ExternalBenchmarkEvaluator {
    fn drop(&mut self) {
        if let Some(running) = &mut self.running {
            let _ = running.input.take();
            let _ = running.child.kill();
            let _ = running.child.wait();
        }
    }
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut input = BufReader::new(
        File::open(path).with_context(|| format!("failed to hash {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn read_bounded_line(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            ensure!(
                line.is_empty(),
                "benchmark evaluator response is unterminated"
            );
            return Ok(None);
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        ensure!(
            line.len() as u64 + take as u64 <= MAX_RESPONSE_BYTES,
            "benchmark evaluator response exceeds {MAX_RESPONSE_BYTES} bytes"
        );
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.ends_with(b"\n") {
            return Ok(Some(line));
        }
    }
}

fn validate_sha256(value: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .context("evaluator identity must use `sha256:<64 lowercase hex>`")?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "evaluator identity must use `sha256:<64 lowercase hex>`"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::acceptance::{BenchmarkFamily, SuiteVisibility};
    use crate::benchmark::{BenchmarkArtifact, BenchmarkSpec, MetricDirection, TargetRole};

    #[cfg(unix)]
    #[test]
    fn persistent_worker_receives_strict_requests_and_finishes() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("evaluator.sh");
        std::fs::write(
            &worker,
            concat!(
                "#!/bin/sh\n",
                "while IFS= read -r request; do\n",
                "  test -n \"$request\" || exit 7\n",
                "  case \"$request\" in *'\"model\":{\"type\":\"full_precision\",'*) ;; *) exit 8 ;; esac\n",
                "  printf '%s\\n' '{\"version\":2,\"measurement\":{\"score\":0.75,\"gpu_hours\":0.01,\"examples\":4}}'\n",
                "done\n"
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&worker).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&worker, permissions).unwrap();
        let hash = file_sha256(&worker).unwrap();
        let mut evaluator = ExternalBenchmarkEvaluator::new(&worker, Vec::new(), &hash).unwrap();
        let artifact = VerifiedArtifact {
            path: directory.path().join("fixture.jsonl"),
            sha256: "1".repeat(64),
            bytes: 1,
        };
        let case = BenchmarkSpec {
            id: "reasoning".into(),
            catalog_id: None,
            family: BenchmarkFamily::Reasoning,
            metric: "accuracy".into(),
            direction: MetricDirection::Maximize,
            stable_anchor: false,
            artifact: BenchmarkArtifact {
                path: artifact.path.clone(),
                sha256: artifact.sha256.clone(),
            },
            evaluator: BTreeMap::new(),
        };
        let target = BenchmarkTarget {
            id: "candidate".into(),
            checkpoint_manifest: directory.path().join("checkpoint.json"),
            checkpoint_manifest_sha256: "2".repeat(64),
            training_evidence: directory.path().join("training-evidence.json"),
            training_evidence_sha256: "3".repeat(64),
            training_gpu_hours: 1.0,
            parameters: 10,
            routed_active_parameters: 8,
            stored_bytes: 100,
            representation: crate::benchmark::ModelRepresentationTarget::FullPrecision,
        };
        let model = VerifiedModelRepresentation::FullPrecision {
            weights: directory.path().join("weights.safetensors"),
            weights_sha256: "4".repeat(64),
            stored_bytes: 100,
        };
        let request = EvaluationRequest {
            suite_id: "public",
            visibility: SuiteVisibility::Public,
            case: &case,
            artifact: &artifact,
            target: &target,
            model: &model,
            role: TargetRole::Candidate,
            model_seed: 7,
            example_order_seed: 9,
            pair_ordinal: 0,
            max_gpu_hours: 0.1,
        };
        let first = evaluator.evaluate(&request).unwrap();
        let second = evaluator.evaluate(&request).unwrap();
        assert_eq!(first.score, 0.75);
        assert_eq!(second.examples, 4);
        evaluator.finish().unwrap();
    }
}
