"""Materialize the released splits from separately obtained benchmark records.

Run from a checkout with ``python -m scripts.prepare_data --help``.
No API calls, training, formula recalculation, or answer changes are performed.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
import csv
import json
from pathlib import Path

BENCHMARKS = ("searchqa", "spreadsheet", "officeqa", "docvqa", "livemath", "alfworld")
ROOT = Path(__file__).resolve().parents[1]


def read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("Source must contain an array of record objects (or a DocVQA data array)")
    return value


def normalize(benchmark: str, row: dict, image_root: Path | None = None) -> dict:
    item = dict(row)
    key = next((item[k] for k in ("id", "key", "uid", "questionId")
                if k in item and item[k] is not None and str(item[k]).strip()), None)
    if key is None:
        raise ValueError("Source record has no id, key, uid, or questionId")
    item["id"] = str(key)
    required = {
        "searchqa": ("question", "context", "answers"),
        "spreadsheet": ("instruction", "answer_position"),
        "officeqa": ("question",),
        "docvqa": ("question",),
        "livemath": ("question", "choices", "correct_choice"),
        "alfworld": ("gamefile",),
    }[benchmark]
    missing = [field for field in required if field not in item or item[field] in (None, "")]
    if missing:
        raise ValueError(f"{benchmark} {key}: missing fields {missing}")
    if benchmark in {"officeqa", "docvqa"}:
        answers = item.get("answers") or item.get("answer") or item.get("ground_truth")
        if answers in (None, "", []):
            raise ValueError(f"{benchmark} {key}: no answer field")
        if benchmark == "officeqa":
            item.setdefault("ground_truth", answers[0] if isinstance(answers, list) else answers)
        else:
            # DocVQA's engine reads a CSV answer cell as a Python literal list.
            if isinstance(answers, str):
                try:
                    parsed = ast.literal_eval(answers)
                except (SyntaxError, ValueError):
                    parsed = answers
                answers = parsed if isinstance(parsed, list) else [str(parsed)]
            item["answer"] = repr(answers)
            image = item.get("image_path") or item.get("image")
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"DocVQA {key}: provide an image_path or an image filename")
            image_path = Path(image).expanduser()
            if not image_path.is_absolute():
                if image_root is None:
                    raise ValueError("Relative DocVQA image paths require --image-root")
                image_path = image_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"DocVQA image missing: {image_path}")
            item["image_path"] = str(image_path.resolve())
    if benchmark == "spreadsheet":
        item.setdefault("spreadsheet_path", f"spreadsheet/{key}")
    return item


def materialize(benchmark: str, manifest: dict, records: Iterable[dict], output: Path,
                image_root: Path | None = None) -> dict[str, int]:
    splits = manifest["splits"]
    ids = [str(row["id"]) for split in ("train", "val", "test") for row in splits[split]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest IDs must be unique within and across splits")
    wanted = set(ids)
    selected = {}
    for row in records:
        key = next((str(row[k]) for k in ("id", "key", "uid", "questionId")
                    if k in row and row[k] is not None and str(row[k]).strip()), "")
        if key not in wanted:
            continue
        if key in selected:
            raise ValueError(f"Duplicate source ID: {key}")
        selected[key] = normalize(benchmark, row, image_root)
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} manifest IDs, including {sorted(missing)[:5]}")
    # Validate all records before writing any split. Preserve manifest order.
    counts = {}
    for split in ("train", "val", "test"):
        rows = [selected[str(row["id"])] for row in splits[split]]
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)
        if benchmark == "docvqa":
            fields = sorted({field for row in rows for field in row}) or ["id", "question", "answer", "image_path"]
            with (folder / "items.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            (folder / "items.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        counts[split] = len(rows)
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--source", type=Path, help="Full source records (JSON/JSONL/CSV), or LiveMath monthly directory")
    parser.add_argument("--manifest", type=Path, help="Defaults to data/manifests/<benchmark>.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, help="Exact split directory override")
    parser.add_argument("--image-root", type=Path, help="Root of relative DocVQA image filenames")
    args = parser.parse_args(argv)
    manifest_path = args.manifest or ROOT / "data" / "manifests" / f"{args.benchmark}.json"
    manifest = json.loads(manifest_path.read_text())
    if args.benchmark == "alfworld" and args.source is None:
        rows = [row for split in manifest["splits"].values() for row in split]
    elif args.benchmark == "searchqa" and args.source is None:
        from datasets import load_dataset
        dataset = load_dataset("lucadiliello/searchqa")
        rows = (row for split in dataset.values() for row in split)
    elif args.source is None:
        parser.error("--source is required for this benchmark; see docs/data.md")
    elif args.benchmark == "livemath" and args.source.is_dir():
        from skillopt.envs.livemathematicianbench.dataloader import load_items
        rows = load_items(str(args.source))
    else:
        rows = read_records(args.source)
    output = args.output_dir or args.data_dir / args.benchmark / "splits"
    counts = materialize(args.benchmark, manifest, rows, output, args.image_root)
    print(json.dumps({"benchmark": args.benchmark, "output": str(output.resolve()), "counts": counts}))


if __name__ == "__main__":
    main()
