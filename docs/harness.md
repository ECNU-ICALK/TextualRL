# Native CLI harnesses

TextualRL can run target rollouts through Codex CLI (`codex_exec`) or Claude Code CLI (`claude_code_exec`). The optimizer still uses the `qwen_chat` HTTP transport with the Responses API. A **model** is the inference deployment, such as `gpt-5.5`; a **harness** is the program that gives that model its prompt, files, tools, and execution loop. Choosing a harness does not provide access to its configured model.

The ten flat, self-contained presets in [`configs/harness/`](../configs/harness/) use `gpt-5.5` for both target and optimizer. They retain the optimization settings from the corresponding root benchmark preset, including four same-task rollouts, outcome-stratified critique, cross-group evidence, and existing edit and validation checks. Adjust CLI settings and worker counts for your deployment.

## Presets and data

| Benchmark | Codex preset | Claude Code preset | Rollout workers |
| --- | --- | --- | ---: |
| SearchQA | [`codex_searchqa.yaml`](../configs/harness/codex_searchqa.yaml) | [`claude_code_searchqa.yaml`](../configs/harness/claude_code_searchqa.yaml) | 8 |
| LiveMathematicianBench | [`codex_livemath.yaml`](../configs/harness/codex_livemath.yaml) | [`claude_code_livemath.yaml`](../configs/harness/claude_code_livemath.yaml) | 8 |
| SpreadsheetBench | [`codex_spreadsheet.yaml`](../configs/harness/codex_spreadsheet.yaml) | [`claude_code_spreadsheet.yaml`](../configs/harness/claude_code_spreadsheet.yaml) | 4 |
| DocVQA | [`codex_docvqa.yaml`](../configs/harness/codex_docvqa.yaml) | [`claude_code_docvqa.yaml`](../configs/harness/claude_code_docvqa.yaml) | 4 |
| OfficeQA | [`codex_officeqa.yaml`](../configs/harness/codex_officeqa.yaml) | [`claude_code_officeqa.yaml`](../configs/harness/claude_code_officeqa.yaml) | 2 |

Use the same [dataset preparation and split layout](data.md) as the root presets. SpreadsheetBench requires its Python extra and uses `mode: single`; native exec targets do not support its `multi` or `react` modes. OfficeQA uses its local document corpus with `search_mode: offline`. DocVQA requires image access through the selected CLI and model deployment. ALFWorld has no native CLI preset; use its root API preset.

`workers` and `max_api_workers` have the table's value. The optimizer retains `analyst_workers: 16` and `optimizer_endpoint_concurrency: 16`. Adjust worker counts to your machine and provider capacity. Native target processes are not scheduled by the optimizer HTTP endpoint limiter.

## Install and authenticate the target CLI

Install the chosen CLI using the official [Codex CLI installation guide](https://learn.chatgpt.com/docs/codex/cli) or [Claude Code setup guide](https://code.claude.com/docs/en/setup). The executable must be available to the same shell that runs TextualRL. Check the installation without starting a benchmark:

```bash
codex --version
codex exec --help
# Or, for Claude Code:
claude --version
claude --help
```

The presets set `codex_exec_path: codex` or `claude_code_exec_path: claude`. Change the relevant key in your own config if you need a different executable path. They also set `*_exec_use_sdk: cli`, so installing a Python SDK cannot silently change the execution path.

For Codex, authenticate with `codex login`, or configure an API key or provider through the CLI's own supported authentication mechanism. `codex login status` reports the active method. See [Codex authentication](https://learn.chatgpt.com/docs/auth). Access to the selected `gpt-5.5` deployment must be available through that native configuration.

For Claude Code with **GPT-5.5**, configure a gateway that accepts the Anthropic Messages API and routes the requested model identifier to GPT-5.5, including the tool and image behavior needed by your benchmark. Set `ANTHROPIC_BASE_URL` and the gateway's required `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY`, then follow its model mapping instructions. This requires a compatible gateway deployment rather than a stock Anthropic account: routing to non-Claude models is outside Anthropic's supported [gateway configuration](https://code.claude.com/docs/en/llm-gateway). To use a Claude deployment instead, authenticate through Claude Code and pass its model identifier with `--target-model`.

`TARGET_API_KEY` and `TARGET_BASE_URL` configure the direct API target backend; they do not authenticate either native CLI. Native evaluation requires the selected CLI's own authentication and no optimizer credential. Native training additionally requires `OPTIMIZER_API_KEY` and an optimizer endpoint configured by `OPTIMIZER_BASE_URL`. `.env` is not loaded automatically; export the variables before running TextualRL.

## Inspect, train, evaluate, and resume

Dry-run resolves configuration without starting a CLI, calling a model, or loading benchmark data:

```bash
python -m textualrl train \
  --config configs/harness/codex_searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/harness/codex/searchqa/run1 \
  --dry-run
```

After configuring native CLI authentication and the optimizer endpoint, run training:

```bash
python -m textualrl train \
  --config configs/harness/codex_searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/harness/codex/searchqa/run1
```

To use Claude Code instead, select `configs/harness/claude_code_searchqa.yaml` and a separate output directory. Replace `searchqa` with another preset's benchmark name as needed. `--target-model` and `--optimizer-model` override the model fields; use the deployment identifiers your native CLI/gateway and optimizer endpoint actually expose. `--target-backend codex_exec` or `--target-backend claude_code_exec` can override a backend, but starting from the matching harness preset also supplies its execution settings and the required SpreadsheetBench mode.

Evaluate the selected skill explicitly:

```bash
python -m textualrl eval \
  --config configs/harness/codex_searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/harness/codex/searchqa/eval-best \
  --skill ./outputs/harness/codex/searchqa/run1/best_skill.md \
  --split test
```

Without `--skill`, evaluation uses the preset's initial skill. To resume TextualRL optimization, use the same configuration, data, and run directory:

```bash
python -m textualrl train \
  --config configs/harness/codex_searchqa.yaml \
  --data-dir ./data \
  --resume-from ./outputs/harness/codex/searchqa/run1
```

`--resume-from` restores TextualRL's optimization state; it is separate from a native CLI's conversation-resume command. See [running experiments](running.md) for checkpoint selection and saved artifacts.

## Execution controls and results

Codex presets select `workspace-write`, disable `full_auto`, network access for sandboxed commands, and web search, and set the approval policy to `never`. They use `codex_exec_reasoning_effort: medium`. Keep these controls when adapting a preset. The configured network restriction concerns agent commands; the CLI still needs its authenticated model connection. See the official [Codex command reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli) for native flag semantics.

Claude Code presets use `claude_code_exec_effort: medium` and `claude_code_exec_max_thinking_tokens: 0`. Zero means that TextualRL omits an explicit thinking-token option; it does not guarantee that the model disables thinking. The existing adapter chooses the tools and permission mode for each benchmark invocation. These settings do not create a Codex-style sandbox. Native flags and supported effort levels depend on the CLI and deployment; consult the [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference).

The harness adapter prepares a task workspace and the selected `textualrl-target` skill for each rollout. The benchmark evaluator then scores the resulting answer or spreadsheet solution. The saved `effective_config.json`, benchmark predictions, rollout workspaces, and CLI output provide execution evidence. Some existing workspace and artifact names contain `codex` even when the selected backend is Claude Code.

Optimizer HTTP calls appear in `runtime_requests.jsonl`. Inspect native target execution and any CLI-reported usage in the rollout workspace artifacts, separately from API token accounting. Target sampling and output limits follow the native CLI/provider configuration; the common temperature, seed, and output-token fields retained from root presets apply to the API path. Dry-run checks configuration, while a benchmark run exercises the authenticated CLI and scores its outputs.
