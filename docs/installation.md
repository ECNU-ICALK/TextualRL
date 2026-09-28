# Installation

Run the following from a checkout or unpacked source archive. Python 3.10 or newer is required; a dedicated virtual environment keeps benchmark dependencies separate from other projects.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m textualrl --help
```

`python -m pip install -r requirements.txt` is equivalent to the core editable installation. The `textualrl` console command and `python -m textualrl` use the same entrypoint. The CLI, runtime, and benchmark adapters share the `textualrl` package namespace.

## Benchmark extras

| Benchmark | Installation | Additional requirements |
| --- | --- | --- |
| SearchQA | `python -m pip install -e '.[searchqa]'` | `datasets` supports dataset preparation; prepared JSON splits can be read by the core runtime. |
| SpreadsheetBench | `python -m pip install -e '.[spreadsheetbench]'` | `openpyxl` for workbooks and evaluation; `pandas` for generated spreadsheet programs. |
| OfficeQA | `python -m pip install -e '.[officeqa]'` | No additional Python package for the provided text/file runtime; acquire the document corpus and page evidence separately. |
| DocVQA | `python -m pip install -e '.[docvqa]'` | No additional Python package for the provided image path runtime; the target endpoint must accept image inputs. |
| LiveMathematicianBench | `python -m pip install -e '.[livemathematicianbench]'` | No additional Python package for the prepared multiple-choice data. |
| ALFWorld | `python -m pip install -e '.[alfworld]'` | ALFWorld, Gymnasium, NumPy, and OmegaConf; acquire the environment assets separately. |

`python -m pip install -e '.[benchmarks]'` installs all listed Python extras. Prepare datasets and environment assets as described in [data preparation](data.md) before launching a real run. ALFWorld's dependencies may require platform-specific build tools; its text environment does not require serving the target model on the same machine.

Connect the CLI to remote or locally hosted OpenAI-compatible model endpoints for the target and optimizer roles.

## Credentials

The CLI reads four environment variables:

| Variable | Role |
| --- | --- |
| `TARGET_API_KEY` | Credential for target rollouts and evaluation. |
| `TARGET_BASE_URL` | Base URL for the target model service. |
| `OPTIMIZER_API_KEY` | Separate credential for optimizer calls during training. |
| `OPTIMIZER_BASE_URL` | Base URL for the optimizer service. |

Copy and edit [`.env.example`](../.env.example), then export its variables with `set -a; source .env; set +a`. The CLI does not automatically load `.env`. The placeholder endpoint domains are deliberately nonfunctional; replace them with endpoints you control or are authorized to use. Evaluation needs only the target role. `--dry-run` needs neither role's credentials.

## Build a distributable package

```bash
python -m pip install -e '.[dev]'
python -m build
```

The wheel contains Python code, runtime prompts, initial skills, and the ALFWorld YAML runtime configuration. The source archive also includes the example configurations, identifier/path manifests, and documentation. Use a checkout or unpacked source archive when running data preparation commands and `--config configs/...` examples, which refer to repository-level manifests and configuration files.
