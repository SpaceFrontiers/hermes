//! Strict WorkflowV2 projection for the current wake-training loop.
//!
//! Full lifecycle orchestration consumes [`hermes_train::workflow`] directly.
//! This module keeps the existing trainer executable while making unsupported
//! phase/task execution fail loudly instead of dropping workflow phases.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use hermes_train::task::{TaskAdapter, TaskConfig};
use hermes_train::workflow::{PhaseKind, ResolvedWorkflow, WORKFLOW_VERSION, load_workflow};
use serde::Serialize;

pub(crate) const CURRICULUM_VERSION: u32 = WORKFLOW_VERSION;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ObjectiveConfig {
    CausalLm,
    Summarization {
        instruction: String,
    },
    RetrievalPlanning {
        instruction: String,
    },
    ContrastiveRetrieval {
        temperature: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        layer: Option<usize>,
        query_prefix: String,
        document_prefix: String,
    },
}

impl ObjectiveConfig {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::CausalLm => "causal_lm",
            Self::Summarization { .. } => "summarization",
            Self::RetrievalPlanning { .. } => "retrieval_planning",
            Self::ContrastiveRetrieval { .. } => "retrieval_representation",
        }
    }

    pub(crate) fn retrieval_layer(&self) -> Option<usize> {
        match self {
            Self::ContrastiveRetrieval { layer, .. } => *layer,
            _ => None,
        }
    }

    pub(crate) fn temperature(&self) -> Option<f64> {
        match self {
            Self::ContrastiveRetrieval { temperature, .. } => Some(*temperature),
            _ => None,
        }
    }
}

fn project_task(phase_name: &str, task: &TaskConfig) -> Result<ObjectiveConfig> {
    match task {
        TaskConfig::CausalLm {} => Ok(ObjectiveConfig::CausalLm),
        TaskConfig::Summarization { instruction } => Ok(ObjectiveConfig::Summarization {
            instruction: instruction.clone(),
        }),
        TaskConfig::RetrievalPlanning { instruction } => Ok(ObjectiveConfig::RetrievalPlanning {
            instruction: instruction.clone(),
        }),
        TaskConfig::RetrievalRepresentation {
            temperature,
            layer,
            query_prefix,
            document_prefix,
        } => Ok(ObjectiveConfig::ContrastiveRetrieval {
            temperature: *temperature,
            layer: *layer,
            query_prefix: query_prefix.clone(),
            document_prefix: document_prefix.clone(),
        }),
        task => bail!(
            "workflow phase `{phase_name}` task `{}` is not executable by the wake trainer; dispatch it through its WorkflowV2 task executor",
            task.name()
        ),
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedStage {
    pub(crate) name: String,
    pub(crate) phase_kind: PhaseKind,
    pub(crate) data: PathBuf,
    pub(crate) objective: ObjectiveConfig,
    pub(crate) sequence_length: usize,
    pub(crate) batch_size: usize,
    pub(crate) gradient_accumulation: usize,
    pub(crate) epochs: usize,
    pub(crate) shuffle_buffer: usize,
    pub(crate) steps: Option<usize>,
    pub(crate) loss_weight: f64,
    pub(crate) learning_rate_scale: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedCurriculum {
    pub(crate) version: u32,
    pub(crate) stages: Vec<ResolvedStage>,
}

fn project_wake_workflow(workflow: ResolvedWorkflow) -> Result<ResolvedCurriculum> {
    let mut stages = Vec::with_capacity(workflow.phases.len());
    for phase in workflow.phases {
        if !matches!(
            phase.kind,
            PhaseKind::Pretrain | PhaseKind::ContinuedPretrain | PhaseKind::Sft
        ) {
            bail!(
                "workflow phase `{}` ({}) requires a WorkflowV2 phase executor and cannot be run by the wake-only trainer",
                phase.name,
                phase.kind.name()
            );
        }
        let task = phase
            .task
            .as_ref()
            .expect("validated task-data phase has a task");
        let objective = project_task(&phase.name, task)?;
        let epochs = phase.epochs_or_default();
        let shuffle_buffer = phase.shuffle_buffer_or_default();
        let loss_weight = phase.loss_weight_or_default();
        let learning_rate_scale = phase.learning_rate_scale_or_default();
        stages.push(ResolvedStage {
            name: phase.name,
            phase_kind: phase.kind,
            data: phase
                .data
                .expect("validated task-data phase has a data path"),
            objective,
            sequence_length: phase
                .sequence_length
                .expect("validated task-data phase has sequence_length"),
            batch_size: phase
                .batch_size
                .expect("validated task-data phase has batch_size"),
            gradient_accumulation: phase
                .gradient_accumulation
                .expect("validated optimization phase has gradient_accumulation"),
            epochs,
            shuffle_buffer,
            steps: phase.steps,
            loss_weight,
            learning_rate_scale,
        });
    }
    Ok(ResolvedCurriculum {
        version: workflow.version,
        stages,
    })
}

pub(crate) fn load_curriculum(path: &Path) -> Result<ResolvedCurriculum> {
    project_wake_workflow(load_workflow(path)?)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn workflow_v2_resolves_paths_and_projects_supported_wake_phases() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "summaries",
                    "type": "sft",
                    "data": "data/summaries.jsonl",
                    "task": {"type": "summarization"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10,
                    "learning_rate_scale": 0.25
                }]
            }"#,
        )
        .unwrap();

        let curriculum = load_curriculum(&path).unwrap();
        let stage = &curriculum.stages[0];
        assert_eq!(curriculum.version, 2);
        assert_eq!(stage.phase_kind, PhaseKind::Sft);
        assert_eq!(stage.data, dir.path().join("data/summaries.jsonl"));
        assert_eq!(stage.sequence_length, 512);
        assert_eq!(stage.batch_size, 8);
        assert_eq!(stage.gradient_accumulation, 2);
        assert_eq!(stage.epochs, 1);
        assert_eq!(stage.shuffle_buffer, 8192);
        assert_eq!(stage.steps, Some(10));
        assert_eq!(stage.learning_rate_scale, 0.25);
        assert_eq!(stage.objective.name(), "summarization");
    }

    #[test]
    fn curriculum_version_one_has_no_legacy_parser() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(&path, r#"{"version":1,"stages":[]}"#).unwrap();
        let error = load_curriculum(&path).unwrap_err().to_string();
        assert!(error.contains("unsupported workflow version 1"), "{error}");
    }

    #[test]
    fn unsupported_phase_is_never_silently_projected_away() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "sleep",
                    "type": "sleep"
                }]
            }"#,
        )
        .unwrap();
        let error = load_curriculum(&path).unwrap_err().to_string();
        assert!(error.contains("sleep"), "{error}");
        assert!(error.contains("cannot be run"), "{error}");
    }

    #[test]
    fn unsupported_task_is_never_coerced_to_a_different_loss() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "reasoning",
                    "type": "sft",
                    "data": "reasoning.jsonl",
                    "task": {"type": "qa_reasoning"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10
                }]
            }"#,
        )
        .unwrap();
        let error = load_curriculum(&path).unwrap_err().to_string();
        assert!(error.contains("qa_reasoning"), "{error}");
        assert!(error.contains("not executable"), "{error}");
    }

    #[test]
    fn unknown_phase_fields_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "retrieval",
                    "type": "continued_pretrain",
                    "data": "pairs.jsonl",
                    "task": {"type": "retrieval_representation"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "sequnce_length": 512
                }]
            }"#,
        )
        .unwrap();
        let error = format!("{:#}", load_curriculum(&path).unwrap_err());
        assert!(error.contains("sequnce_length"), "{error}");
    }
}
