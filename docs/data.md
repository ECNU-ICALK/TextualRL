# Data preparation

Prepare benchmark splits from the identifiers and ALFWorld game paths in
`data/manifests/`. Obtain the original examples and assets from their providers
under the applicable access terms. The preparation command selects those exact
identifiers in manifest order. It does not resample splits or change answers.

| Benchmark | Train / validation / test | Original data |
|---|---:|---|
| SearchQA | 400 / 200 / 1400 | [lucadiliello/searchqa](https://huggingface.co/datasets/lucadiliello/searchqa) |
| Spreadsheet | 80 / 40 / 280 | [SpreadsheetBench Verified 400](https://huggingface.co/datasets/KAKA22/SpreadsheetBench) |
| OfficeQA | 50 / 24 / 172 | [OfficeQA Full](https://huggingface.co/datasets/databricks/officeqa) |
| DocVQA | 107 / 53 / 374 | [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA), selected validation examples |
| LiveMath | 35 / 17 / 125 | [LiveMathematicianBench](https://huggingface.co/datasets/LiveMathematicianBench/LiveMathematicianBench), 202511–202602 |
| ALFWorld | 39 / 18 / 134 | [ALFWorld](https://github.com/alfworld/alfworld), `json_2.1.1` |

ALFWorld uses the released SkillOpt path manifest with 18 selection environments;
the SkillOpt paper describes 140 selection environments. Use the included
manifest to reproduce the 39/18/134 split in this setup.

## Commands

Run from the repository root after installation. Except for SearchQA's optional
Hugging Face download, this helper consumes locally obtained records. It supports
JSON arrays, JSONL, and CSV. Missing or duplicate requested IDs are reported before
any split is written.

```bash
# Downloads the SearchQA records using the datasets extra.
python -m scripts.prepare_data searchqa

# JSON records from the separately extracted Verified 400 distribution.
python -m scripts.prepare_data spreadsheet --source /path/to/verified400_records.json

# OfficeQA may require provider access approval.
python -m scripts.prepare_data officeqa --source /path/to/officeqa_full.csv

# Local records must contain image filenames, not in-memory image objects.
python -m scripts.prepare_data docvqa --source /path/to/docvqa_records.json \
  --image-root /path/to/docvqa_images

# Directory containing the official qa_<month>_final.json files.
python -m scripts.prepare_data livemath --source /path/to/LiveMathematicianBench

# Materializes paths only. Download the ALFWorld assets separately.
python -m scripts.prepare_data alfworld
```

Use `--data-dir /path/to/data` for another output root, then pass the same root to
`python -m textualrl train` or `eval`. The resulting layout is:

```text
data/
  manifests/                 # released identifiers/path metadata
  <benchmark>/splits/
    train/items.json
    val/items.json
    test/items.json
  officeqa/documents/        # downloaded local text documents
  spreadsheet/workbooks/
    spreadsheet/<task-id>/   # input and golden workbooks
  alfworld_data/json_2.1.1/  # separately downloaded ALFWorld games
```

DocVQA writes `items.csv` instead of JSON because its existing loader consumes
CSV. Its image paths are resolved to local absolute paths during preparation.
`configs/alfworld.yaml` sets `ALFWORLD_DATA` from `alfworld_data_root`. Set that
field to the location produced by your ALFWorld download if it differs from the
layout above. OfficeQA document names must match the referenced source files.

## Record fields and evaluation

| Benchmark | Required source fields |
|---|---|
| SearchQA | `key` or `id`, `question`, `context`, `answers` |
| Spreadsheet | `id`, `instruction`, `answer_position`, plus task metadata from the distribution. `spreadsheet_path` defaults to `spreadsheet/<id>`. |
| OfficeQA | `uid` or `id`, `question`, `ground_truth` or `answer`, and the original `source_files` / `source_docs` metadata |
| DocVQA | `questionId` or `id`, `question`, `answers` / `answer` / `ground_truth`, and a local `image_path` or `image` filename |
| LiveMath | The official monthly records, or normalized records with `id`, `question`, `choices`, `correct_choice`, and the original theorem metadata |
| ALFWorld | Supplied by the released path manifest. Assets must exist under `ALFWORLD_DATA`. |

For Spreadsheet, retain the distribution's input/golden workbook pairs. The
loader recognizes numbered `*_init.xlsx` / `*_golden.xlsx` pairs and bare
`initial.xlsx` / `golden.xlsx` pairs. Scoring uses the existing benchmark evaluator without
offline formula recalculation.

OfficeQA's default is local-document `search_mode: offline`. If enabling its
optional custom search provider, set `search_mode: custom_search` and a non-empty
`search_api_url` in the YAML configuration. Export the authentication token in
the environment variable named by `search_auth_env`, which defaults to
`OFFICEQA_CUSTOM_SEARCH_AUTH` in the provided configuration.

Raw data, downloaded assets, local paths in prepared records, and execution
outputs are ignored by Git. The six lightweight manifests remain tracked.
