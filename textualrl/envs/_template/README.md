# TextualRL benchmark template

Copy this scaffold to add a benchmark to a source checkout:

```bash
cp -r textualrl/envs/_template textualrl/envs/your_benchmark
mv textualrl/envs/your_benchmark/env_template.py textualrl/envs/your_benchmark/adapter.py
mv textualrl/envs/your_benchmark/loader_template.py textualrl/envs/your_benchmark/dataloader.py
cp textualrl/envs/your_benchmark/config_template.yaml configs/your_benchmark.yaml
```

1. Rename `TemplateBenchmarkEnv` and `TemplateBenchmarkLoader`, and update
   the loader import in `adapter.py` to your new `dataloader` module.
2. Adapt `_normalize_item` to your data. The loader reads one JSON array or
   JSONL file per `train/`, `val/`, and `test/` directory. The inherited raw
   loader also supports ratio splitting from a single JSON/JSONL file.
3. Implement the TODO in `rollout`: call the target model, score its output,
   return `id`/`hard`/`soft`, and save each non-empty trajectory to
   `<out_dir>/predictions/<id>/conversation.json` for inherited reflection.
   The supplied rollout only returns placeholder failures; it makes no API
   calls and produces no measured benchmark scores or trajectory files.
4. Register your class in `_register_builtins()` in **both** `scripts/train.py`
   and `scripts/eval_only.py`; each has its own `_ENV_REGISTRY`.
5. Replace `your_benchmark` in the copied flat config and provide your data.
   Edit the copied `skills/initial.md` to supply initial task instructions.
   Keep the YAML self-contained; the public CLI does not accept `_base_` or
   nested model/training sections.

Check configuration resolution before implementing the benchmark:

```bash
textualrl train --config textualrl/envs/_template/config_template.yaml --dry-run
```

This check needs no credentials or data and does not validate registration
or execute rollouts. See the [new benchmark guide](../../../docs/guide/new-benchmark.md)
for registration code, data examples, and adapter checks.
