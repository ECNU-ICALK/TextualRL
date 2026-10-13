"""
TextualRL Benchmark Data Loader Template
================================
Copy this file and adapt ``_normalize_item`` to your benchmark data.
The loader is a :class:`textualrl.datasets.base.SplitDataLoader`
subclass — the base class handles both ``split_mode="split_dir"`` (read
an existing train/val/test layout) and ``split_mode="ratio"`` (build the
splits from a single raw file deterministically).

For a fully worked example see
``textualrl/envs/officeqa/dataloader.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from textualrl.datasets.base import SplitDataLoader


def _normalize_item(raw: dict) -> dict:
    """
    Normalise one raw entry into the dict shape TextualRL expects.

    The required ``"id"`` must be non-empty, unique across tasks, and safe
    as a directory name. Add whatever extra fields your rollout needs.
    """
    item_id = raw.get("uid")
    if item_id is None or item_id == "":
        item_id = raw.get("id")
    if item_id is None or not str(item_id).strip():
        raise ValueError("Each benchmark item requires a non-empty id or uid")
    answer = raw.get("ground_truth")
    if answer is None or answer == "":
        answer = raw.get("answer", "")
    return {
        "id": str(item_id),
        "question": str(raw.get("question") or raw.get("prompt") or ""),
        "ground_truth": "" if answer is None else str(answer),
        "task_type": str(raw.get("category") or raw.get("task_type") or "template"),
        # TODO: add benchmark-specific keys here.
    }


class TemplateBenchmarkLoader(SplitDataLoader):
    """
    Data loader for <Your Benchmark Name>.

    Subclass note: you usually only need to implement
    :meth:`load_split_items`. The base class drives ``setup(cfg)``,
    materialises ratio-mode splits, exposes ``train_items``,
    ``val_items``, ``test_items``, and builds ``BatchSpec`` objects on
    demand.

    The inherited :meth:`load_raw_items` already supports JSON/JSONL in
    ``split_mode="ratio"``. It writes the raw splits, which are then read
    and normalized by :meth:`load_split_items`. Override it only for a
    different source format.
    """

    def load_split_items(self, split_path: str) -> list[dict]:
        """Load all items for one split directory.

        ``split_path`` is e.g. ``data/your_benchmark/train/``. Return a
        list of dicts, each shaped like :func:`_normalize_item`'s output.
        """
        path = Path(split_path)

        json_files = sorted(path.glob("*.json"))
        if json_files:
            with json_files[0].open(encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(
                    f"Expected JSON array at top level of {json_files[0]}"
                )
            return [_normalize_item(row) for row in payload]

        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            items: list[dict] = []
            with jsonl_files[0].open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(_normalize_item(json.loads(line)))
            return items

        raise FileNotFoundError(
            f"No .json or .jsonl file found in {split_path}"
        )

    # Optional — override only for a raw format other than JSON/JSONL.
    # def load_raw_items(self, data_path: str) -> list[dict]:
    #     ...
