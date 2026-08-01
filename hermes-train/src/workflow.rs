//! Strict version-2 training workflow configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::task::{TaskAdapter, TaskConfig, TaskExecution};

pub const WORKFLOW_VERSION: u32 = 2;

fn default_one_f64() -> f64 {
    1.0
}

fn default_group_size() -> usize {
    128
}

fn default_true() -> bool {
    true
}

/// Coarse executor class. It keeps orchestration generic while making state
/// transitions and checkpoint boundaries explicit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseClass {
    Optimization,
    ModelMutation,
    Assessment,
    Release,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    Pretrain,
    ContinuedPretrain,
    Sft,
    Preference,
    Rl,
    Distillation,
    Sleep,
    Quantization,
    Evaluation,
    Promotion,
}

impl PhaseKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pretrain => "pretrain",
            Self::ContinuedPretrain => "continued_pretrain",
            Self::Sft => "sft",
            Self::Preference => "preference",
            Self::Rl => "rl",
            Self::Distillation => "distillation",
            Self::Sleep => "sleep",
            Self::Quantization => "quantization",
            Self::Evaluation => "evaluation",
            Self::Promotion => "promotion",
        }
    }

    pub fn class(self) -> PhaseClass {
        match self {
            Self::Pretrain
            | Self::ContinuedPretrain
            | Self::Sft
            | Self::Preference
            | Self::Rl
            | Self::Distillation => PhaseClass::Optimization,
            Self::Sleep | Self::Quantization => PhaseClass::ModelMutation,
            Self::Evaluation => PhaseClass::Assessment,
            Self::Promotion => PhaseClass::Release,
        }
    }

    pub fn uses_task_data(self) -> bool {
        !matches!(self, Self::Sleep | Self::Promotion)
    }

    pub fn updates_model(self) -> bool {
        matches!(
            self.class(),
            PhaseClass::Optimization | PhaseClass::ModelMutation
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    BinaryG128,
    TernaryG128,
    TernaryEntropyG128,
}

/// Training recipe for a quantized candidate. The task and data remain on the
/// enclosing phase so QAT and distillation use the same adapter contracts as
/// every other optimization phase.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum QuantizationTraining {
    Qat {
        #[serde(default)]
        warmup_steps: usize,
        #[serde(default = "default_true")]
        straight_through: bool,
    },
    Distillation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        teacher_checkpoint: Option<PathBuf>,
        #[serde(default = "default_one_f64")]
        temperature: f64,
        #[serde(default = "default_one_f64")]
        loss_weight: f64,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    pub format: QuantizationFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_format: Option<QuantizationFormat>,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    #[serde(default)]
    pub start_step: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_step: Option<usize>,
    #[serde(default = "default_true")]
    pub embeddings: bool,
    #[serde(default = "default_true")]
    pub lm_head: bool,
    pub training: QuantizationTraining,
}

impl QuantizationConfig {
    fn validate(&self, phase_name: &str) -> Result<()> {
        ensure!(
            self.group_size == 128,
            "workflow phase `{phase_name}` quantization group_size must be 128 for a *_g128 format"
        );
        ensure!(
            self.end_step
                .is_none_or(|end_step| end_step > self.start_step),
            "workflow phase `{phase_name}` quantization end_step must be greater than start_step"
        );
        match &self.training {
            QuantizationTraining::Qat { .. } => {}
            QuantizationTraining::Distillation {
                teacher_checkpoint,
                temperature,
                loss_weight,
            } => {
                ensure!(
                    teacher_checkpoint
                        .as_ref()
                        .is_some_and(|path| !path.as_os_str().is_empty()),
                    "workflow phase `{phase_name}` quantization distillation requires teacher_checkpoint"
                );
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "workflow phase `{phase_name}` quantization temperature must be finite and positive"
                );
                ensure!(
                    loss_weight.is_finite() && *loss_weight >= 0.0,
                    "workflow phase `{phase_name}` quantization loss_weight must be finite and non-negative"
                );
            }
        }
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        if let QuantizationTraining::Distillation {
            teacher_checkpoint: Some(teacher_checkpoint),
            ..
        } = &mut self.training
            && teacher_checkpoint.is_relative()
        {
            *teacher_checkpoint = base.join(&*teacher_checkpoint);
        }
    }
}

/// Serialized phase definition. Algorithm-specific knobs are namespaced under
/// `parameters`; task packages never interpret them.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseV2 {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: PhaseKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_accumulation: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epochs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_buffer: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_rate_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

impl PhaseV2 {
    pub fn epochs_or_default(&self) -> usize {
        self.epochs.unwrap_or(1)
    }

    pub fn shuffle_buffer_or_default(&self) -> usize {
        self.shuffle_buffer.unwrap_or(8192)
    }

    pub fn loss_weight_or_default(&self) -> f64 {
        self.loss_weight.unwrap_or(1.0)
    }

    pub fn learning_rate_scale_or_default(&self) -> f64 {
        self.learning_rate_scale.unwrap_or(1.0)
    }

    fn validate(&self) -> Result<()> {
        let name = &self.name;
        ensure!(
            !name.trim().is_empty(),
            "workflow phase name must not be empty"
        );
        if let Some(task) = &self.task {
            task.validate()
                .with_context(|| format!("invalid task in workflow phase `{name}`"))?;
        }

        validate_positive(self.sequence_length, "sequence_length", name)?;
        validate_positive(self.batch_size, "batch_size", name)?;
        validate_positive(self.gradient_accumulation, "gradient_accumulation", name)?;
        validate_positive(self.epochs, "epochs", name)?;
        validate_positive(self.steps, "steps", name)?;
        if let Some(loss_weight) = self.loss_weight {
            ensure!(
                loss_weight.is_finite() && loss_weight > 0.0,
                "workflow phase `{name}` loss_weight must be finite and positive"
            );
        }
        if let Some(scale) = self.learning_rate_scale {
            ensure!(
                scale.is_finite() && scale > 0.0,
                "workflow phase `{name}` learning_rate_scale must be finite and positive"
            );
        }

        if self.kind.uses_task_data() {
            ensure!(
                self.task.is_some(),
                "workflow phase `{name}` ({}) requires a task",
                self.kind.name()
            );
            ensure!(
                self.data
                    .as_ref()
                    .is_some_and(|path| !path.as_os_str().is_empty()),
                "workflow phase `{name}` ({}) requires a data path",
                self.kind.name()
            );
            ensure!(
                self.sequence_length.is_some(),
                "workflow phase `{name}` ({}) requires sequence_length",
                self.kind.name()
            );
            ensure!(
                self.batch_size.is_some(),
                "workflow phase `{name}` ({}) requires batch_size",
                self.kind.name()
            );
        } else {
            ensure!(
                self.task.is_none() && self.data.is_none(),
                "workflow phase `{name}` ({}) must not read task data",
                self.kind.name()
            );
        }

        match self.kind {
            PhaseKind::Pretrain
            | PhaseKind::ContinuedPretrain
            | PhaseKind::Sft
            | PhaseKind::Preference
            | PhaseKind::Rl
            | PhaseKind::Distillation
            | PhaseKind::Quantization => {
                ensure!(
                    self.gradient_accumulation.is_some(),
                    "workflow phase `{name}` ({}) requires gradient_accumulation",
                    self.kind.name()
                );
            }
            PhaseKind::Evaluation => {
                ensure!(
                    self.gradient_accumulation.is_none(),
                    "workflow phase `{name}` (evaluation) must not set gradient_accumulation"
                );
                ensure!(
                    self.loss_weight.is_none() && self.learning_rate_scale.is_none(),
                    "workflow phase `{name}` (evaluation) must not set optimizer weights"
                );
            }
            PhaseKind::Sleep | PhaseKind::Promotion => {
                ensure!(
                    self.sequence_length.is_none()
                        && self.batch_size.is_none()
                        && self.gradient_accumulation.is_none()
                        && self.epochs.is_none()
                        && self.shuffle_buffer.is_none()
                        && self.steps.is_none()
                        && self.loss_weight.is_none()
                        && self.learning_rate_scale.is_none(),
                    "workflow phase `{name}` ({}) must not set task execution or optimizer geometry",
                    self.kind.name()
                );
            }
        }

        let execution = self.task.as_ref().map(|task| task.contract().execution);
        match self.kind {
            PhaseKind::Sft => ensure!(
                execution == Some(TaskExecution::SupervisedGeneration),
                "workflow phase `{name}` (sft) requires a supervised-generation task"
            ),
            PhaseKind::Preference => ensure!(
                execution == Some(TaskExecution::PairwisePreference),
                "workflow phase `{name}` (preference) requires a pairwise-preference task"
            ),
            PhaseKind::Rl => ensure!(
                execution == Some(TaskExecution::VerifiableReward),
                "workflow phase `{name}` (rl) requires a verifiable-reward task"
            ),
            _ => {}
        }

        match (&self.kind, &self.quantization) {
            (PhaseKind::Quantization, Some(config)) => config.validate(name)?,
            (PhaseKind::Quantization, None) => {
                bail!("workflow phase `{name}` (quantization) requires quantization settings")
            }
            (_, Some(_)) => bail!(
                "workflow phase `{name}` ({}) must not set quantization settings",
                self.kind.name()
            ),
            (_, None) => {}
        }
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        if let Some(data) = &mut self.data
            && data.is_relative()
        {
            *data = base.join(&*data);
        }
        if let Some(quantization) = &mut self.quantization {
            quantization.resolve_paths(base);
        }
    }
}

fn validate_positive(value: Option<usize>, field: &str, phase_name: &str) -> Result<()> {
    ensure!(
        value.is_none_or(|value| value > 0),
        "workflow phase `{phase_name}` {field} must be positive when set"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowV2 {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub phases: Vec<PhaseV2>,
}

impl WorkflowV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == WORKFLOW_VERSION,
            "unsupported workflow version {}; this build supports version {WORKFLOW_VERSION}",
            self.version
        );
        if let Some(name) = &self.name {
            ensure!(!name.trim().is_empty(), "workflow name must not be empty");
        }
        ensure!(!self.phases.is_empty(), "workflow contains no phases");
        let mut names = BTreeSet::new();
        for phase in &self.phases {
            ensure!(
                names.insert(phase.name.clone()),
                "duplicate workflow phase name `{}`",
                phase.name
            );
            phase.validate()?;
        }
        Ok(())
    }

    pub fn resolve(mut self, source: &Path) -> Result<ResolvedWorkflow> {
        self.validate()?;
        let base = source.parent().unwrap_or_else(|| Path::new("."));
        for phase in &mut self.phases {
            phase.resolve_paths(base);
        }
        Ok(ResolvedWorkflow {
            version: self.version,
            name: self.name,
            phases: self.phases,
        })
    }
}

/// Fully validated workflow with file references resolved against its source.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedWorkflow {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub phases: Vec<PhaseV2>,
}

impl ResolvedWorkflow {
    pub fn validate(&self) -> Result<()> {
        WorkflowV2 {
            version: self.version,
            name: self.name.clone(),
            phases: self.phases.clone(),
        }
        .validate()
    }
}

pub fn load_workflow(path: &Path) -> Result<ResolvedWorkflow> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read workflow {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid workflow JSON in {}", path.display()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("workflow {} has no integer `version`", path.display()))?;
    ensure!(
        version == u64::from(WORKFLOW_VERSION),
        "unsupported workflow version {version}; this build supports version {WORKFLOW_VERSION}"
    );
    let workflow: WorkflowV2 = serde_json::from_value(value)
        .with_context(|| format!("invalid workflow JSON in {}", path.display()))?;
    workflow.resolve(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training_phase(name: &str, kind: &str, task: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "type": kind,
            "task": task,
            "data": format!("data/{name}.jsonl"),
            "sequence_length": 512,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "steps": 10
        })
    }

    #[test]
    fn workflow_v2_supports_the_complete_phase_lifecycle() {
        let phases = vec![
            training_phase(
                "pretrain",
                "pretrain",
                serde_json::json!({"type": "causal_lm"}),
            ),
            training_phase(
                "continued",
                "continued_pretrain",
                serde_json::json!({"type": "retrieval_representation"}),
            ),
            training_phase(
                "sft",
                "sft",
                serde_json::json!({"type": "instruction_tuning"}),
            ),
            training_phase(
                "preference",
                "preference",
                serde_json::json!({"type": "pairwise_preference"}),
            ),
            training_phase(
                "rl",
                "rl",
                serde_json::json!({
                    "type": "verifiable_rl",
                    "verifier": {"adapter": "exact_answer"}
                }),
            ),
            training_phase(
                "distill",
                "distillation",
                serde_json::json!({"type": "causal_lm"}),
            ),
            serde_json::json!({
                "name": "sleep",
                "type": "sleep",
                "parameters": {"policy": "cms_v1"}
            }),
            {
                let mut phase = training_phase(
                    "quantize",
                    "quantization",
                    serde_json::json!({"type": "causal_lm"}),
                );
                phase["quantization"] = serde_json::json!({
                    "format": "binary_g128",
                    "training": {"type": "qat"}
                });
                phase
            },
            serde_json::json!({
                "name": "evaluate",
                "type": "evaluation",
                "task": {"type": "qa_reasoning"},
                "data": "data/eval.jsonl",
                "sequence_length": 512,
                "batch_size": 8
            }),
            serde_json::json!({
                "name": "promote",
                "type": "promotion",
                "parameters": {"suite": "sealed"}
            }),
        ];
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "name": "full-lifecycle",
            "phases": phases
        }))
        .unwrap();

        workflow.validate().unwrap();
        assert_eq!(workflow.phases.len(), 10);
        assert_eq!(workflow.phases[6].kind.class(), PhaseClass::ModelMutation);
        assert_eq!(workflow.phases[9].kind.class(), PhaseClass::Release);
    }

    #[test]
    fn load_workflow_resolves_paths_and_applies_accessible_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "foundation",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data/foundation.jsonl",
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10
                }]
            }"#,
        )
        .unwrap();

        let workflow = load_workflow(&path).unwrap();
        let phase = &workflow.phases[0];
        assert_eq!(
            phase.data.as_deref(),
            Some(dir.path().join("data/foundation.jsonl").as_path())
        );
        assert_eq!(phase.epochs_or_default(), 1);
        assert_eq!(phase.shuffle_buffer_or_default(), 8192);
        assert_eq!(phase.loss_weight_or_default(), 1.0);
        assert_eq!(phase.learning_rate_scale_or_default(), 1.0);
    }

    #[test]
    fn version_one_workflows_are_rejected_without_a_legacy_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(&path, r#"{"version":1,"stages":[]}"#).unwrap();

        let error = load_workflow(&path).unwrap_err().to_string();
        assert!(error.contains("unsupported workflow version 1"), "{error}");
    }

    #[test]
    fn phase_names_and_fields_are_strictly_validated() {
        let duplicate: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [
                training_phase("same", "pretrain", serde_json::json!({"type": "causal_lm"})),
                training_phase("same", "pretrain", serde_json::json!({"type": "causal_lm"}))
            ]
        }))
        .unwrap();
        let error = duplicate.validate().unwrap_err().to_string();
        assert!(error.contains("duplicate workflow phase"), "{error}");

        let unknown = serde_json::from_value::<WorkflowV2>(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "train",
                "type": "pretrain",
                "task": {"type": "causal_lm"},
                "data": "data.jsonl",
                "sequence_length": 32,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "sequnce_length": 32
            }]
        }))
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("sequnce_length"), "{unknown}");
    }

    #[test]
    fn algorithm_specific_phases_reject_incompatible_task_signals() {
        let preference: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [training_phase(
                "preference",
                "preference",
                serde_json::json!({"type": "causal_lm"})
            )]
        }))
        .unwrap();
        let error = preference.validate().unwrap_err().to_string();
        assert!(error.contains("pairwise-preference task"), "{error}");

        let rl: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [training_phase(
                "rl",
                "rl",
                serde_json::json!({"type": "qa_reasoning"})
            )]
        }))
        .unwrap();
        let error = rl.validate().unwrap_err().to_string();
        assert!(error.contains("verifiable-reward task"), "{error}");
    }

    #[test]
    fn lifecycle_phases_cannot_silently_consume_or_optimize_data() {
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "sleep",
                "type": "sleep",
                "task": {"type": "causal_lm"},
                "data": "external.jsonl"
            }]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("must not read task data"), "{error}");

        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "eval",
                "type": "evaluation",
                "task": {"type": "causal_lm"},
                "data": "eval.txt",
                "sequence_length": 32,
                "batch_size": 2,
                "gradient_accumulation": 1
            }]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(
            error.contains("must not set gradient_accumulation"),
            "{error}"
        );
    }

    #[test]
    fn quantization_configuration_is_typed_and_resolves_teacher_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "quantize",
                    "type": "quantization",
                    "task": {"type": "causal_lm"},
                    "data": "calibration.jsonl",
                    "sequence_length": 128,
                    "batch_size": 4,
                    "gradient_accumulation": 1,
                    "steps": 20,
                    "quantization": {
                        "format": "ternary_g128",
                        "training": {
                            "type": "distillation",
                            "teacher_checkpoint": "teacher.safetensors",
                            "temperature": 2.0
                        }
                    }
                }]
            }"#,
        )
        .unwrap();

        let workflow = load_workflow(&path).unwrap();
        let quantization = workflow.phases[0].quantization.as_ref().unwrap();
        assert_eq!(quantization.group_size, 128);
        let QuantizationTraining::Distillation {
            teacher_checkpoint, ..
        } = &quantization.training
        else {
            panic!("expected distillation")
        };
        assert_eq!(
            teacher_checkpoint.as_deref(),
            Some(dir.path().join("teacher.safetensors").as_path())
        );
    }
}
