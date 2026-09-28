# TextualRL

TextualRL improves an agent's shared textual instructions while keeping the target model's weights fixed. It collects multiple rollouts for each training task, critiques outcome groups, reviews proposed edits against pooled trajectory evidence, and selects candidate contexts using held-out validation rewards.

This source release builds on [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt). The `skillopt` Python namespace is retained for compatibility; `textualrl` provides the public command-line interface. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licenses.

## What is included

- Groupwise Policy Critique and Cross-Group Policy Update implementations, prompts, and initial skills.
- Configurations and adapters for SearchQA, SpreadsheetBench, OfficeQA, DocVQA, LiveMathematicianBench, and ALFWorld.
- A portable training/evaluation CLI with separate target and optimizer credentials, dry-run configuration inspection, and resume support.

Benchmark datasets, trained skills, run outputs, model weights, and paper-result artifacts are not bundled. This is a prepared source release, not a claim that the paper's experiments have been rerun with this package. [Release scope](docs/reproducibility.md) explains the source provenance and configuration differences.

## Install

Use Python 3.10 or newer from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the extra needed by your benchmark, for example:

```bash
python -m pip install -e '.[searchqa]'
python -m pip install -e '.[spreadsheetbench]'
python -m pip install -e '.[alfworld]'
```

See [installation](docs/installation.md) for all six benchmarks. Obtain and prepare datasets separately using the [data guide](docs/data.md).

## Inspect, train, and evaluate

First inspect the resolved configuration. Dry-run does not make model calls or require API credentials or benchmark data:

```bash
python -m textualrl train \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/searchqa/run1 \
  --dry-run
```

Copy `.env.example` to `.env`, replace its placeholders with your two endpoints and credentials, and export them:

```bash
cp .env.example .env
# Edit .env before sourcing it.
set -a
source .env
set +a
```

The default target is `Qwen3.8-27B` with thinking explicitly disabled. The default optimizer is `gpt-5.5` with medium reasoning. The endpoint must expose the API and model name configured for its role; model names may need to match your provider's deployment names.

```bash
python -m textualrl train \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/searchqa/run1

python -m textualrl eval \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --output-dir ./outputs/searchqa/eval-best \
  --skill ./outputs/searchqa/run1/best_skill.md \
  --split test
```

Evaluation uses the target model; training also requires the optimizer. To resume an interrupted run, supply the same configuration, data, and run directory:

```bash
python -m textualrl train \
  --config configs/searchqa.yaml \
  --data-dir ./data \
  --resume-from ./outputs/searchqa/run1
```

See [running experiments](docs/running.md) for configuration controls, checkpoint selection, and artifacts. A normal run can make many model calls: four rollouts for each of 40 tasks already produce 160 training trajectories per full batch, before critique and validation.

## Method and implementation

**Groupwise Policy Critique** keeps same-task rollout groups together. Mixed outcomes support within-task contrast; all-success and all-failure groups support cross-task preservation or repair. Critics return edit proposals and evidence cards separately, including evidence with no proposed edit.

**Cross-Group Policy Update** consolidates proposals, reviews their scope against current-step evidence, and applies a bounded patch. The ordinary step accepts a candidate only when its configured validation score improves.

The configurations also retain inherited **Meta** and **Slow** mechanisms. In particular, unconditional Slow updates can change the active context at epoch boundaries without the ordinary step-level validation comparison. Consequently, not every context change is validation-accepted, and the final context can differ from `best_skill.md`. See the [method-to-code map](docs/method.md) and [checkpoint explanation](docs/running.md#best-on-validation-and-final-contexts).

## Documentation

- [Installation and optional dependencies](docs/installation.md)
- [Datasets and split layout](docs/data.md)
- [Configuration and endpoint controls](docs/configuration.md)
- [Training, evaluation, resume, and saved artifacts](docs/running.md)
- [Method-to-code map](docs/method.md)
- [Release provenance and reproducibility scope](docs/reproducibility.md)

## License and acknowledgements

The copied [MIT license](LICENSE) and Microsoft copyright notice are preserved verbatim. Bundled ALFWorld environment helpers carry Apache-2.0 attribution; the corresponding license and notices are included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Benchmark data and separately installed dependencies retain their own terms.

## Offline checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

The tests cover outcome routing, evidence review, edit checks, context overflow
splitting, generation settings, checkpoint behavior, and split preparation. An
end-to-end smoke test runs the trainer, completed-run resume, and evaluator
against a local HTTP fixture using synthetic tasks. These checks use no paid
model service. They do not measure benchmark accuracy.
