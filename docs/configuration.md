# Configuration reference

The public CLI accepts flat, self-contained YAML files. Start from one of the six files in [`configs/`](../configs/) and save a copy when changing an experiment.

## Paths and models

The CLI substitutes these variables in YAML:

| Variable | Default or source |
| --- | --- |
| `${TEXTUALRL_REPO_ROOT}` | Installed package root, used to locate bundled initial skills. |
| `${TEXTUALRL_DATA_DIR}` | `<package root>/data`; override with `--data-dir` or the environment variable. |
| `${TEXTUALRL_OUTPUT_DIR}` | `<package root>/outputs`; override with the environment variable. `--output-dir` selects the exact run directory. |
| `${TEXTUALRL_TARGET_MODEL}` | `Qwen3.8-27B`; override with `--target-model` or the environment variable. |
| `${TEXTUALRL_OPTIMIZER_MODEL}` | `gpt-5.5`; override with `--optimizer-model` or the environment variable. |
| `${TARGET_BASE_URL}` | Target endpoint; absent an override, `http://localhost:8000/v1`. |
| `${OPTIMIZER_BASE_URL}` | Optimizer endpoint; absent an override, `https://api.openai.com/v1`. |

For an installed wheel, explicitly supply writable data and output paths. Initial skills remain available inside the installed package; the example YAML files come from the checkout or source archive. The CLI writes its resolved settings to `effective_config.json`, and the trainer writes the derived settings to `config.json`.

Credentials are read separately from `TARGET_API_KEY` and `OPTIMIZER_API_KEY`. Do not put keys in YAML, URL query strings, or command-line model names. Dry-run displays resolved noncredential settings without loading the data or making requests.

## Role-specific request settings

Both public roles use the `qwen_chat` transport, which supports the two API styles needed by these presets. The name does not force both roles to use a Qwen model.

| Flat key | Default target | Default optimizer |
| --- | --- | --- |
| `<role>_model` | `Qwen3.8-27B` | `gpt-5.5` |
| `<role>_api_style` | `chat_completions` | `responses` |
| `<role>_thinking_api` | `chat_template` | `reasoning_effort` |
| `<role>_reasoning_effort` | `none` | `medium` |
| `<role>_qwen_chat_enable_thinking` | `false` | `true` |
| `<role>_qwen_chat_temperature` | `0.7` for training | `none`, omitted from the request |
| `<role>_endpoint_concurrency` | `32` for SearchQA/ALFWorld, `16` otherwise | `16` |

For Qwen targets, the request explicitly carries `chat_template_kwargs.enable_thinking: false`. The optimizer uses Responses requests with medium reasoning and no explicit output-token cap. This requires endpoints that accept those fields and API styles. Editing only the model name is insufficient when moving to a service with different request semantics; adjust the role's API and thinking fields together.

Target output limits are benchmark-specific and listed in [running experiments](running.md#default-optimization-settings). `evaluation_target_temperature: 0.0` and `evaluation_target_seed: 42` override target generation during validation and evaluation. `seed: 42` controls training sampling, while `split_seed` belongs to data-split configuration. The release presets read existing split directories rather than silently regenerating them.

An endpoint base URL may contain a comma-separated pool. A role's `endpoint_concurrency` may be one positive integer for every endpoint, or a list with one capacity per endpoint. The presets set these deployment limits to the corresponding worker capacities; adjust them to the capacity of your endpoint. If omitted, target capacity defaults to `max_api_workers` (or `workers`), and optimizer capacity defaults to `analyst_workers`, with a fallback of 16. If both roles resolve to the same request URL, the runtime shares that URL's capacity using the smaller declared limit. `workers` controls target rollout workers and `analyst_workers` controls parallel critique workers; endpoint capacity limits actual in-flight HTTP calls. Inspect `runtime_requests.jsonl` for request settings and concurrency observations.

## Optimization controls

| Flat key | Purpose |
| --- | --- |
| `num_epochs`, `train_size`, `batch_size` | Training schedule and unique-task batch size. |
| `same_task_rollouts`, `same_task_rollout_temperature` | Group size and training rollout sampling temperature. |
| `group_relative_outcome_stratified_reflection` | Enables outcome-based task-group routing in the provided presets. |
| `minibatch_size`, `merge_batch_size` | Critique and hierarchical merge batch sizes. |
| `use_cross_group_evidence` | Pools evidence independently of edit proposals for cross-group review. |
| `edit_budget`, `min_edit_budget`, `lr_scheduler` | Maximum edits per step and their schedule. |
| `use_gate`, `gate_metric` | Existing step-level validation selection; presets use hard success score. |
| `reject_unobservable_runtime_edits` | Retains the existing filter for instructions depending on unavailable runtime information. |
| `quarantine_rejected_edits` | Rejected-edit handling and its configured thresholds. |
| `eval_test` | Runs final validation/promotion and initial/best/final test evaluation after optimization. |

See [the method guide](method.md) for the default algorithm.

An existing `stop_after_step` diagnostic setting can stop after a requested completed step. A nonzero value creates a step-limited run and omits the final test stage; do not report it as a completed main experiment. The public presets set it to zero.
