# Training workflows and task contracts

Hermes training consumes a strict `WorkflowV2` document. A workflow is an
ordered sequence of named phases; every task-bearing phase selects one task
adapter and declares its data and execution geometry. Relative paths resolve
against the workflow file.

Version 1 trainer curricula are not accepted. There is no implicit conversion
or fallback: regenerate them as version 2 workflows so phase intent is
explicit.

## Phases

| Phase                | Purpose                                    | Required task signal  |
| -------------------- | ------------------------------------------ | --------------------- |
| `pretrain`           | Initial model optimization                 | Any registered task   |
| `continued_pretrain` | Domain/capability continuation             | Any registered task   |
| `sft`                | Supervised fine-tuning                     | Supervised generation |
| `preference`         | Pairwise preference optimization           | Pairwise preference   |
| `rl`                 | Reward-driven post-training                | Verifiable reward     |
| `distillation`       | Teacher/student optimization               | Any registered task   |
| `sleep`              | In-model memory consolidation and dreaming | No external task data |
| `quantization`       | QAT or quantization distillation           | Any registered task   |
| `evaluation`         | Read-only acceptance measurement           | Any registered task   |
| `promotion`          | Candidate acceptance and publication       | No task data          |

Optimization phases require `task`, `data`, `sequence_length`, `batch_size`,
and `gradient_accumulation`. `epochs` defaults to one, `shuffle_buffer` to
8192, and the loss and learning-rate weights to one. `steps`, when present,
caps the phase. Evaluation requires task/data plus sequence and batch size but
rejects optimizer-only fields. Sleep and promotion reject task data and
optimizer geometry. Executor-specific settings are contained in `parameters`
instead of being interpreted by task adapters.

```json
{
  "version": 2,
  "name": "retrieval-model",
  "phases": [
    {
      "name": "foundation",
      "type": "pretrain",
      "task": { "type": "causal_lm" },
      "data": "corpus/foundation.jsonl.zst",
      "sequence_length": 2048,
      "batch_size": 16,
      "gradient_accumulation": 8,
      "steps": 10000
    },
    {
      "name": "representations",
      "type": "continued_pretrain",
      "task": {
        "type": "retrieval_representation",
        "temperature": 0.05,
        "layer": 24
      },
      "data": "corpus/retrieval.jsonl.zst",
      "sequence_length": 512,
      "batch_size": 16,
      "gradient_accumulation": 8,
      "steps": 2000
    }
  ]
}
```

## Built-in task packages

`TaskAdapter` exposes a stable name, data framing, input/target fields, loss-mask
policy, execution signal, requested metrics, configuration validation, and
record validation. It contains no optimizer, model, search-engine, or
RL-algorithm choices.

| Task                       | JSONL fields                                              | Signal                            |
| -------------------------- | --------------------------------------------------------- | --------------------------------- |
| `causal_lm`                | `text` (plain text is also accepted by the data reader)   | Next-token prediction             |
| `summarization`            | `document`, `summary`                                     | Target-only supervised generation |
| `retrieval_representation` | `query`, `positive`, optional `negatives[]`               | Contrastive representation        |
| `retrieval_ranking`        | `query`, `documents[]` with `document` and `relevance`    | Listwise ranking                  |
| `retrieval_planning`       | `request`, `plan`, optional `context`                     | Target-only supervised generation |
| `instruction_tuning`       | `instruction`, `response`, optional `system` and `input`  | Target-only supervised generation |
| `qa_reasoning`             | `question`, `answer`, optional `reasoning`                | Target-only supervised generation |
| `pairwise_preference`      | `prompt`, `chosen`, `rejected`                            | Pairwise preference               |
| `verifiable_rl`            | `prompt`, `verifier_payload`, optional `reference_answer` | Named verifier reward             |

Retrieval ranking requires at least one positive and one non-positive
candidate. Pairwise responses must differ. A `qa_reasoning` task may require
the reasoning field. Verifiable RL configuration names a verifier adapter and
passes opaque parameters to that adapter; the task schema does not execute
commands or embed a particular RL algorithm.

Existing causal, summarization, retrieval-representation, and
retrieval-planning batches retain their fixed-shape execution. Padding and
prompts do not contribute to supervised target loss, complete targets are
reserved before truncating source text, and retrieval embeddings use the last
meaningful token at the configured one-based layer.

## Quantization phase

The quantization phase accepts `binary_g128`, `ternary_g128`, and
`ternary_entropy_g128`. `group_size` defaults to and must remain 128.
`embeddings` and `lm_head` default to true. `start_step`, optional `end_step`,
and optional `warmup_format` control progressive quantization.

The nested `training` object is either:

- `{"type":"qat","warmup_steps":0,"straight_through":true}`; or
- `{"type":"distillation","teacher_checkpoint":"...","temperature":1.0,"loss_weight":1.0}`.

Distillation requires a teacher checkpoint; its relative path resolves against
the workflow. Temperature must be positive and distillation loss weight must
be non-negative.

## Determinism and metrics

Resolved workflow configuration is part of the run signature. Resume must
preserve phase position, phase-local counters, optimizer/RNG state, and task
configuration; reordered or changed workflows fail rather than replaying data
under a different objective.

Metrics identify the phase, phase kind, task, named loss, raw and weighted
loss, optimizer geometry, token/example/truncation counts, throughput, and
task-specific measurements. The JSONL stream remains the source for the W&B
sidecar; lifecycle executors add their own namespaced metrics without changing
task contracts.
