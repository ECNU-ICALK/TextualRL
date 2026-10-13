# Add a benchmark to TextualRL

Work in a source checkout installed with `python -m pip install -e .`.
The [template](../../textualrl/envs/_template/README.md) supplies an instantiable
`EnvAdapter`, a JSON/JSONL loader, and a flat public CLI config. Its rollout is
an explicit placeholder: it makes no model calls and returns failure rows.
Implement task execution and scoring before using it for a real run.

## Copy and name the adapter

Run from the repository root, replacing `your_benchmark` with your module name:

```bash
cp -r textualrl/envs/_template textualrl/envs/your_benchmark
mv textualrl/envs/your_benchmark/env_template.py textualrl/envs/your_benchmark/adapter.py
mv textualrl/envs/your_benchmark/loader_template.py textualrl/envs/your_benchmark/dataloader.py
cp textualrl/envs/your_benchmark/config_template.yaml configs/your_benchmark.yaml
```

Rename `TemplateBenchmarkEnv` to `YourBenchmarkAdapter` and
`TemplateBenchmarkLoader` to `YourBenchmarkLoader`. In `adapter.py`, replace
both the loader import and its constructor call:

```python
from textualrl.envs.your_benchmark.dataloader import YourBenchmarkLoader
```

Constructor parameter names must match flat config keys: the train and eval
factories inspect `__init__` and forward matching values. Keep `setup` calling
both `super().setup(cfg)` and `self.dataloader.setup(cfg)`.

## Load tasks

By default the config reads this layout under `TEXTUALRL_DATA_DIR`:

```text
your_benchmark/splits/
  train/items.json
  val/items.json
  test/items.json
```

Use one JSON array or one JSONL file per split. For example, `train/items.json`
can contain:

```json
[
  {"id": "train-001", "question": "What is 2 + 2?", "ground_truth": "4", "task_type": "arithmetic"}
]
```

Adapt `_normalize_item` to preserve the fields your rollout and scorer need.
IDs must be non-empty, unique across tasks, and safe as directory names because
trajectory paths contain them. The template also accepts `uid`, `prompt`,
`answer`, and `category` as aliases. Reference answers belong in the scorer or
reflection context; keep them out of the target model's prompt.

For a single raw JSON/JSONL file, change the copied config to:

```yaml
split_mode: ratio
split_dir: ''
data_path: ${TEXTUALRL_DATA_DIR}/your_benchmark/items.jsonl
split_ratio: '2:1:7'
split_seed: 42
```

The inherited loader makes deterministic train/val/test splits under the run's
`_generated_splits/` directory, then normalizes the loaded rows. Override
`load_raw_items` only if your raw data uses another format. Use enough items
for every requested split to be non-empty, or provide official splits directly.

## Implement execution and reflection inputs

Replace the TODO block in `YourBenchmarkAdapter.rollout`. For each task, build
its prompt with the current `skill_content`, execute the task with the target
model, and score the prediction. The shared `textualrl.model.chat_target` or
`chat_target_messages` entry points use the configured target endpoint.
Return one dictionary per execution with at least:

```python
{"id": "train-001", "hard": 1, "soft": 1.0, "task_type": "arithmetic"}
```

`hard` is 0 or 1, and `soft` is a score between 0 and 1. Include useful fields
such as `predicted_answer` and `fail_reason`. Save the actual non-empty
conversation as a JSON list of message dictionaries to
`<out_dir>/predictions/<id>/conversation.json`; inherited `reflect` reads it.
A simple conversation has `{"role": "user", "content": "..."}` and
`{"role": "assistant", "content": "..."}` entries. Record tool observations
when the task uses tools.

Generic reflection prompts work by default. Add benchmark-specific
`prompts/analyst_error.md` and `prompts/analyst_success.md` only if needed.
Keep `same_task_rollouts: 1` until you implement `expand_same_task_rollouts`
with distinct execution IDs for each repeated task.

## Register and configure

Add this block inside `_register_builtins()` in **both** `scripts/train.py`
and `scripts/eval_only.py`:

```python
    try:
        from textualrl.envs.your_benchmark.adapter import YourBenchmarkAdapter
        _ENV_REGISTRY["your_benchmark"] = YourBenchmarkAdapter
    except ImportError:
        pass
```

These are separate registries; `textualrl/envs/__init__.py` has no registry.
An import error leaves the adapter unavailable, so import the adapter directly
to diagnose missing optional dependencies.

Replace `your_benchmark` throughout `configs/your_benchmark.yaml`. Keep all
settings at the top level; the public CLI accepts self-contained flat YAML.
The template uses `${TEXTUALRL_DATA_DIR}`, `${TEXTUALRL_OUTPUT_DIR}`,
`${TEXTUALRL_TARGET_MODEL}`, `${TEXTUALRL_OPTIMIZER_MODEL}`, `${TARGET_BASE_URL}`,
and `${OPTIMIZER_BASE_URL}`. The CLI supplies defaults and supports data,
output, and model overrides; see [configuration](../configuration.md).
Credentials are supplied through `TARGET_API_KEY` and `OPTIMIZER_API_KEY`.
The template's `skill_init` points to the copied `skills/initial.md` using
`${TEXTUALRL_REPO_ROOT}`. Edit this file for your task, or make it an empty
file to train from an empty skill. Keep the `skill_init` setting: the trainer
expects this key. Evaluation uses it unless you supply `--skill`.

## Check the adapter and run

Resolve the template or copied config without credentials, data, or API calls:

```bash
textualrl train --config configs/your_benchmark.yaml --dry-run
```

Dry run checks configuration only. Exercise loading and batch construction on
your fixture data without executing the target model:

```python
from textualrl.envs.your_benchmark.adapter import YourBenchmarkAdapter

adapter = YourBenchmarkAdapter(split_dir="data/your_benchmark/splits")
adapter.setup({"env": "your_benchmark"})
train_items = adapter.build_train_env(batch_size=2, seed=42)
val_items = adapter.build_eval_env(env_num=0, split="val", seed=42)
assert train_items and val_items
assert adapter.get_task_types()
```

Test your scorer on known correct and incorrect answers, and run a rollout
against a local model fixture to inspect its returned scores and saved
conversation. The template's own offline checks can be run with
`python -m pytest tests/test_benchmark_template.py`.
After implementation, registration, data setup, and endpoint configuration:

```bash
textualrl train --config configs/your_benchmark.yaml
textualrl eval --config configs/your_benchmark.yaml --skill outputs/your_benchmark/best_skill.md --split test
```
