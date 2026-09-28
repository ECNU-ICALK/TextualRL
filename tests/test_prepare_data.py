import json
from pathlib import Path

import pytest

from scripts.prepare_data import materialize, read_records
from skillopt.envs.docvqa.dataloader import DocVQADataLoader
from skillopt.envs.officeqa.dataloader import OfficeQADataLoader
from skillopt.envs.searchqa.dataloader import SearchQADataLoader


def manifest(*ids):
    return {"splits": {split: [{"id": value}] for split, value in zip(("train", "val", "test"), ids)}}


def test_searchqa_actual_loader_and_manifest_order(tmp_path):
    rows = [{"key": str(i), "question": "Q?", "context": ["evidence"], "answers": [str(i)]}
            for i in (3, 1, 2)]
    counts = materialize("searchqa", manifest("2", "1", "3"), iter(rows), tmp_path)
    assert counts == {"train": 1, "val": 1, "test": 1}
    loader = SearchQADataLoader()
    loaded = loader.load_split_items(str(tmp_path / "train"))
    assert loaded[0]["id"] == "2"
    assert loaded[0]["answers"] == ["2"]


def test_missing_or_duplicate_ids_do_not_write_partial_splits(tmp_path):
    rows = [{"id": "1", "question": "Q", "context": "C", "answers": ["A"]}]
    with pytest.raises(ValueError, match="Missing"):
        materialize("searchqa", manifest("1", "2", "3"), rows, tmp_path)
    assert not (tmp_path / "train").exists()
    with pytest.raises(ValueError, match="Duplicate source"):
        materialize("searchqa", manifest("1", "2", "3"), rows * 2, tmp_path)


def test_docvqa_csv_roundtrip_preserves_multiple_answers_and_image(tmp_path):
    (tmp_path / "example.png").write_bytes(b"fixture")
    rows = [{"questionId": str(i), "question": "Q", "answers": ["2", "two"], "image": "example.png"}
            for i in (1, 2, 3)]
    out = tmp_path / "splits"
    materialize("docvqa", manifest("1", "2", "3"), rows, out, tmp_path)
    item = DocVQADataLoader().load_split_items(str(out / "test"))[0]
    assert item["answers"] == ["2", "two"]
    assert item["image_path"] == str(tmp_path / "example.png")


def test_officeqa_csv_source_roundtrip(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text('uid,question,ground_truth,source_files\nU1,Q,42,doc.txt\nU2,Q,43,doc.txt\nU3,Q,44,doc.txt\n')
    out = tmp_path / "splits"
    materialize("officeqa", manifest("U1", "U2", "U3"), read_records(source), out)
    item = OfficeQADataLoader().load_split_items(str(out / "val"))[0]
    assert item["ground_truth"] == "43"
    assert item["source_files"] == ["doc.txt"]


def test_published_manifest_sizes_and_alfworld_paths():
    root = Path(__file__).resolve().parents[1] / "data" / "manifests"
    expected = {"searchqa": (400, 200, 1400), "spreadsheet": (80, 40, 280),
                "officeqa": (50, 24, 172), "docvqa": (107, 53, 374),
                "livemath": (35, 17, 125), "alfworld": (39, 18, 134)}
    for name, counts in expected.items():
        data = json.loads((root / f"{name}.json").read_text())
        assert tuple(len(data["splits"][s]) for s in ("train", "val", "test")) == counts
        ids = [r["id"] for rows in data["splits"].values() for r in rows]
        assert len(ids) == len(set(ids))
        if name == "alfworld":
            assert all(not Path(r["gamefile"]).is_absolute() for rows in data["splits"].values() for r in rows)


def test_custom_search_requires_explicit_service(monkeypatch):
    from skillopt.envs.officeqa import tool_runtime
    monkeypatch.delenv("OFFICEQA_CUSTOM_SEARCH_URL", raising=False)
    with pytest.raises(ValueError, match="URL missing"):
        tool_runtime.custom_search("query", auth_token="test-only")
