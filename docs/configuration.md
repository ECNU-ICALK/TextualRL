# Configuration reference

The public CLI accepts flat, self-contained YAML files. Start from one of the six API files in [`configs/`](../configs/) or the ten native CLI files in [`configs/harness/`](../configs/harness/) and save a copy when changing an experiment. The [harness guide](harness.md) covers Codex and Claude Code setup and benchmark support.

## Paths and models

The CLI substitutes these variables in YAML:

| Variable | Default or source |
| --- | --- |
| `${TEXTUALRL_REPO_ROOT}` | Installed package root, used to locate bundled initial skills. |
| `${TEXTUALRL_DATA_DIR}` | `<package root>/data`; override with `--data-dir` or the environment variable. |
| `${TEXTUALRL_OUTPUT_DIR}` | `<package root>/outputs`; override with the environment variable. `--output-dir` selects the exact run directory. |
| `${TEXTUALRL_TARGET_MODEL}` | `Qwen3.8-27B` for API targets, `gpt-5.5` for native targets; override with `--target-model` or the environment variable. |
| `${TEXTUALRL_OPTIMIZER_MODEL}` | `gpt-5.5`; override with `--optimizer-model` or the environment variable. |
| `${TARGET_BASE_URL}` | Target endpoint; absent an override, `http://localhost:8000/v1`. |
| `${OPTIMIZER_BASE_URL}` | Optimizer endpoint; absent an override, `https://api.openai.com/v1`. |

For an installed wheel, explicitly supply writable data and output paths. Initial skills remain available inside the installed package; the example YAML files come from the checkout or source archive. The CLI writes its resolved settings to `effective_config.json`, and the trainer writes the derived settings to `config.json`.

Direct API credentials are read separately from `TARGET_API_KEY` and `OPTIMIZER_API_KEY`. Native target backends use their CLI's own authentication instead of `TARGET_API_KEY`. Do not put keys in YAML, URL query strings, or command-line model names. Dry-run displays resolved noncredential settings without loading the data or making requests.

The harness presets set both model fields to the literal `gpt-5.5`; `--target-model` and `--optimizer-model` override these values. Model environment variables substitute `${TEXTUALRL_TARGET_MODEL}` and `${TEXTUALRL_OPTIMIZER_MODEL}` where those placeholders occur, such as in the root API presets.

## Backend selection

`target_backend` accepts `qwen_chat`, `codex_exec`, or `claude_code_exec`; `--target-backend` overrides the YAML field. `optimizer_backend` remains `qwen_chat` for all public presets. The backend selects the transport or execution harness; `target_model` selects the deployment inside it. Changing a model name does not change the transport, install a CLI, or configure a provider route.

The native presets explicitly select `codex_exec_use_sdk: cli` or `claude_code_exec_use_sdk: cli`, the executable name, and medium effort. Codex presets retain `workspace-write`, `full_auto: false`, disabled sandbox network access and web search, and approval policy `never`. Claude Code presets set `claude_code_exec_max_thinking_tokens: 0`, omitting an explicit thinking-token flag. Full key names and practical behavior are described in [execution controls](harness.md#execution-controls-and-results).

Native presets omit target HTTP and Qwen request settings. Their optimizer still uses the endpoint and request controls below. Use the corresponding harness preset when switching backends, especially for SpreadsheetBench, whose exec adapter requires `mode: single`. Adjust the native presets' worker counts to your deployment capacity.

## Role-specific request settings

Both roles in the root API presets use the `qwen_chat` transport, which supports the two API styles needed by these presets. The name does not force either role to use a Qwen model. With a native target, this section applies only to the optimizer's HTTP calls.

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

API target output limits are benchmark-specific and listed in [running experiments](running.md#default-optimization-settings). `evaluation_target_temperature: 0.0` and `evaluation_target_seed: 42` override API target generation during validation and evaluation; native CLI calls do not forward these sampling fields. `seed: 42` controls training sampling, while `split_seed` belongs to data-split configuration. The release presets read existing split directories rather than silently regenerating them.

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
| `slow_update_gate_with_selection` | Defaults to `true`; epoch-end slow updates use held-out validation and require a strict score improvement before adoption. |
| `reject_unobservable_runtime_edits` | Retains the existing filter for instructions depending on unavailable runtime information. |
| `quarantine_rejected_edits` | Rejected-edit handling and its configured thresholds. |
| `eval_test` | Runs final validation/promotion and initial/best/final test evaluation after optimization. |

See [the method guide](method.md) for the default algorithm.

For the `w/o outcome routing` ablation, keep `same_task_rollouts: 4`,
`use_cross_group_evidence: true`, and `failure_only: false`, and set
`group_relative_outcome_stratified_reflection: false`. All original trajectories
are shuffled into unified batches of `minibatch_size` trajectories (eight in the
presets), without selecting representatives, deduplicating, or separating outcome
categories. Each batch returns edits and validated evidence cards, including cards
without edits, for the usual proposal consolidation and evidence review. The
existing `K=1` analysis behavior is unchanged.

An existing `stop_after_step` diagnostic setting can stop after a requested completed step. A nonzero value creates a step-limited run and omits the final test stage; do not report it as a completed main experiment. The public presets set it to zero.
