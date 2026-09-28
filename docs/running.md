# Running experiments

All examples assume the repository root, an installed package, and data prepared as described in [the data guide](data.md). `--data-dir` is the common data root from that guide; `--output-dir` is a run-specific directory.

## Configuration inspection

```bash
python -m textualrl train --config configs/searchqa.yaml --dry-run
python -m textualrl eval --config configs/searchqa.yaml --dry-run
```

Dry-run prints resolved settings without loading a benchmark adapter or calling an endpoint. It needs neither datasets nor credentials, and does not measure rollout behavior or model quality.

The six standalone configuration files are:

| Benchmark | Configuration |
| --- | --- |
| SearchQA | [`configs/searchqa.yaml`](../configs/searchqa.yaml) |
| SpreadsheetBench | [`configs/spreadsheet.yaml`](../configs/spreadsheet.yaml) |
| OfficeQA | [`configs/officeqa.yaml`](../configs/officeqa.yaml) |
| DocVQA | [`configs/docvqa.yaml`](../configs/docvqa.yaml) |
| LiveMathematicianBench | [`configs/livemath.yaml`](../configs/livemath.yaml) |
| ALFWorld | [`configs/alfworld.yaml`](../configs/alfworld.yaml) |

## Training and evaluation

Training requires both target and optimizer endpoint credentials. Evaluation requires the target role only; the optimizer is not called to improve a supplied skill during evaluation.

```bash
python -m textualrl train \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/searchqa/run1

python -m textualrl eval \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --skill ./outputs/searchqa/run1/best_skill.md \
  --split test \
  --output-dir ./outputs/searchqa/eval-best
```

Omitting `--skill` evaluates the configured initial skill. Select another target or optimizer with `--target-model MODEL` or `--optimizer-model MODEL`; the chosen service must serve that exact model/deployment name. Evaluation defaults to the test split. The adapters accept `train`, `val`, and `test`, with `valid_seen` and `valid_unseen` retained as aliases. `--split all` is available for the noninteractive data adapters; evaluate ALFWorld one split at a time. The data guide explains the on-disk split names.

Each fresh experiment should use its own output directory. Resume keeps the existing run directory:

```bash
python -m textualrl train \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --resume-from ./outputs/searchqa/run1
```

Use the same data, model settings, and configuration when resuming. `--resume-from` selects the trainer's saved history, skills, and runtime state; it is not a new optimization run initialized from `best_skill.md`. Do not combine it with a different `--output-dir`, move the directory while resuming, or have two processes write to the same run. A partially completed step may run again and make additional calls.

## Default optimization settings

| Setting | Value in the public configurations |
| --- | --- |
| Target | `Qwen3.8-27B`, Chat Completions, thinking explicitly disabled |
| Optimizer | `gpt-5.5`, Responses API, reasoning effort `medium` |
| Epochs | 4 |
| Tasks per training batch | Up to 40 |
| Rollouts per task | 4 |
| Analyst / merge minibatch size | 8 / 8, subject to task-block grouping |
| Edit budget | Cosine schedule from 4 to 2 |
| Training sampling / evaluation request seed | 42 / 42 |
| Target training / evaluation temperature | 0.7 / 0.0 |
| Target rollout / API workers | 32 for SearchQA and ALFWorld, 16 for the other benchmarks |
| Analyst workers | 16 |

Worker counts follow the saved source-run configurations. Per-endpoint quotas are deployment settings and default to the corresponding worker capacities. Set them to match the serving endpoint. Dataset size determines the number of batches and the final partial batch; grouping can skip homogeneous routes that lack two distinct tasks.

Target output-token limits are:

| Benchmark | Target output-token limit |
| --- | ---: |
| SearchQA | 8,000 |
| SpreadsheetBench | 8,000 |
| OfficeQA | 8,000 |
| DocVQA | 16,384 |
| LiveMathematicianBench | 16,384 |
| ALFWorld | 4,096 |

These are per-request output limits, not total run token budgets. Iterative environments can make several target requests per task. The optimizer omits an explicit output-token cap and temperature in its Responses requests; the service still applies its own limits. Changing the provider, context window, or token limit changes the experiment.

The configuration's sampling seed and evaluation request seed have different roles. Sampling controls task order and reproducible batching; evaluation request controls are sent where supported by the target backend. A seed and zero temperature do not guarantee deterministic behavior from every service. Use the resolved configuration saved with the run to establish the exact request settings.

## Best-on-validation and final contexts

`best_skill.md` is the context with the best recorded validation score. The active context after the last update is the final context. These can differ, and the final context may not yet have a validation score.

The active context is saved under `skills/skill_vNNNN.md`; `runtime_state.json` identifies its `current_skill_path`, `current_origin`, and the best checkpoint. Use that recorded path to locate the final context rather than guessing a step number. The initial context is `skills/skill_v0000.md`.

With `eval_test: true`, the trainer attempts a final-context validation before test evaluation. If that score beats the stored best, it promotes the final context to `best_skill.md`. It then reports the initial, best-on-validation, and final test results separately, reusing evaluation results when final and best are identical. Inspect the run log and summary for failed or absent evaluations; a missing score is not zero. With final test evaluation disabled, the final validation/promotion pass is also skipped.

For test reporting, normally evaluate `best_skill.md` and identify it as best-on-validation. If reporting the final context, label it explicitly and obtain its path from the runtime state. Never select a context based on which has the best test score.

## Saved artifacts

Artifact paths below are relative to the training output directory. Some artifacts appear only when a step reaches that stage or the relevant option is enabled.

| Path | Contents |
| --- | --- |
| `effective_config.json` | Resolved CLI settings; credentials are supplied only through environment variables. |
| `runtime_requests.jsonl` | Portable runtime request diagnostics for endpoint roles and request controls. |
| `config.json` | Resolved trainer settings with credential fields redacted. |
| `history.json`, `runtime_state.json` | Completed-step history, active/best checkpoint state, and resume information. |
| `best_skill.md`, `skills/skill_vNNNN.md` | Best-on-validation context and active context snapshots. |
| `steps/step_NNNN/rollout/` | Training predictions and benchmark-specific trajectory records. |
| `steps/step_NNNN/patches/` | Analyst responses and edit/evidence proposals. |
| `steps/step_NNNN/cross_group_evidence.json` | Independently pooled evidence cards from the current step. |
| `steps/step_NNNN/merged_patch.json`, `ranked_edits.json` | Consolidated and selected edit operations. |
| `steps/step_NNNN/candidate_skill.md`, `edit_apply_report.json` | Evaluated candidate and patch-application outcomes. |
| `steps/step_NNNN/selection_eval/`, `step_record.json` | Candidate validation rollouts and step decision. |
| `steps/step_NNNN/trajectory_digest.json` | Within-epoch feedback, including rejected combinations when applicable. |
| `summary.json` | Validation/test scores, context origins, step counts, and token accounting. |
| `test_eval_baseline/`, `test_eval/`, `test_eval_final/` | Initial, best-on-validation, and final test outputs when test evaluation is enabled. |

When accumulation exceeds one, rollout and patch folders are nested in `steps/step_NNNN/batch_A/`. Standalone `eval` writes predictions and `eval_summary.json` to its own output directory. The global `summary.json` uses `test_hard` for the best-on-validation test score and `final_test_hard` for the final-context test score.

Token accounting uses reported usage when available. Some backends estimate missing usage, so token summaries are diagnostic accounting rather than provider invoices. Keep task content and model traces in local output directories.

## Execution environment

SpreadsheetBench executes generated Python and OfficeQA provides file tools to the target. Run these benchmarks in an environment appropriate for untrusted model-generated operations, with only the task files the experiment should access. Keep the existing execution limits and restrictions in place. See the benchmark modules for the actual subprocess and file-tool behavior.
