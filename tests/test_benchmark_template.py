"""Exercise the copyable benchmark scaffold without model requests."""
import importlib
import json
import socket
from pathlib import Path

import pytest

from textualrl import cli
from textualrl.envs._template.env_template import TemplateBenchmarkEnv
from textualrl.envs._template.loader_template import TemplateBenchmarkLoader

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "textualrl/envs/_template/config_template.yaml"


def arguments(*extra):
    return cli.parser().parse_args(["train", "--config", str(CONFIG), *extra])


def test_template_resolves_public_defaults_and_explicit_overrides(tmp_path):
    defaults = cli.resolve_config(arguments(), environ={})
    assert defaults["env"] == "your_benchmark"
    assert defaults["split_dir"] == str(ROOT / "data/your_benchmark/splits")
    assert defaults["out_root"] == str(ROOT / "outputs/your_benchmark")
    assert defaults["skill_init"] == str(ROOT / "textualrl/envs/your_benchmark/skills/initial.md")
    assert defaults["target_model"] == "Qwen3.8-27B"
    assert defaults["optimizer_model"] == "gpt-5.5"
    assert defaults["target_backend"] == defaults["optimizer_backend"] == "qwen_chat"
    assert defaults["target_qwen_chat_enable_thinking"] is False
    assert defaults["optimizer_qwen_chat_enable_thinking"] is True

    env = {
        "TEXTUALRL_DATA_DIR": str(tmp_path / "environment-data"),
        "TEXTUALRL_OUTPUT_DIR": str(tmp_path / "environment-output"),
        "TEXTUALRL_TARGET_MODEL": "target-from-environment",
        "TEXTUALRL_OPTIMIZER_MODEL": "optimizer-from-environment",
        "TARGET_BASE_URL": "http://localhost:8010/v1",
        "OPTIMIZER_BASE_URL": "http://localhost:8020/v1",
    }
    config = cli.resolve_config(arguments(), environ=env)
    assert config["split_dir"] == str(tmp_path / "environment-data/your_benchmark/splits")
    assert config["out_root"] == str(tmp_path / "environment-output/your_benchmark")
    assert config["target_model"] == env["TEXTUALRL_TARGET_MODEL"]
    assert config["optimizer_model"] == env["TEXTUALRL_OPTIMIZER_MODEL"]
    assert config["target_qwen_chat_base_url"] == env["TARGET_BASE_URL"]
    assert config["optimizer_qwen_chat_base_url"] == env["OPTIMIZER_BASE_URL"]

    config = cli.resolve_config(arguments(
        "--data-dir", str(tmp_path / "data with spaces"),
        "--output-dir", str(tmp_path / "exact-run"),
        "--target-model", "target-from-cli", "--optimizer-model", "optimizer-from-cli",
    ), environ=env)
    assert config["split_dir"] == str(tmp_path / "data with spaces/your_benchmark/splits")
    assert config["out_root"] == str(tmp_path / "exact-run")
    assert config["target_model"] == "target-from-cli"
    assert config["optimizer_model"] == "optimizer-from-cli"


@pytest.mark.parametrize("action", ["train", "eval"])
def test_template_dry_run_does_not_execute_or_create_output(tmp_path, monkeypatch, capsys, action):
    for name in (
        "TEXTUALRL_DATA_DIR", "TEXTUALRL_OUTPUT_DIR", "TEXTUALRL_TARGET_MODEL",
        "TEXTUALRL_OPTIMIZER_MODEL", "TARGET_BASE_URL", "OPTIMIZER_BASE_URL",
        "TARGET_API_KEY", "OPTIMIZER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected_run(*args):
        pytest.fail("dry run must not execute training or evaluation")

    monkeypatch.setattr(cli, "run", unexpected_run)
    output = tmp_path / "absent-run"
    command = [action, "--config", str(CONFIG), "--dry-run",
               "--data-dir", str(tmp_path / "absent-data"), "--output-dir", str(output)]
    if action == "eval":
        command.extend(["--skill", str(tmp_path / "absent-skill.md")])
    assert cli.main(command) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["dry_run"] is True
    assert resolved["action"] == action
    assert resolved["config"]["out_root"] == str(output)
    assert not output.exists()


@pytest.mark.parametrize("entrypoint", ["scripts.train", "scripts.eval_only"])
def test_template_instantiates_through_both_factories_and_loads_batches(tmp_path, monkeypatch, entrypoint):
    data_root = tmp_path / "data"
    split_dir = data_root / "your_benchmark/splits"
    rows = {
        "train": [{"uid": 0, "prompt": "Zero?", "answer": 0, "category": "numbers"},
                  {"id": "train-1", "question": "One?", "ground_truth": "1", "task_type": "numbers"}],
        "val": [{"id": "val-1", "question": "Text?", "ground_truth": "yes", "task_type": "text"}],
        "test": [{"id": "test-1", "question": "Test?", "ground_truth": "yes", "task_type": "text"}],
    }
    for split, items in rows.items():
        folder = split_dir / split
        folder.mkdir(parents=True)
        if split == "val":
            (folder / "items.jsonl").write_text("\n".join(json.dumps(row) for row in items) + "\n\n")
        else:
            (folder / "items.json").write_text(json.dumps(items))

    config = cli.resolve_config(arguments("--data-dir", str(data_root)), environ={})
    entry = importlib.import_module(entrypoint)
    # The scaffold intentionally has no built-in registration. Exercise the
    # real constructor forwarding once a user supplies their registry entry.
    monkeypatch.setattr(entry, "_ENV_REGISTRY", {"your_benchmark": TemplateBenchmarkEnv})
    monkeypatch.setattr(entry, "_register_builtins", lambda: None)
    adapter = entry.get_adapter(config)
    adapter.setup(config)
    loader = adapter.get_dataloader()
    assert isinstance(loader, TemplateBenchmarkLoader)
    assert loader.train_items[0]["id"] == "0"
    assert loader.train_items[0]["ground_truth"] == "0"
    assert loader.train_items[0]["question"] == "Zero?"
    assert adapter.get_task_types() == ["numbers", "text"]
    assert adapter.max_completion_tokens == config["max_completion_tokens"]

    train_items = adapter.build_train_env(batch_size=2, seed=42)
    assert {row["id"] for row in train_items} == {"0", "train-1"}
    assert adapter.build_eval_env(env_num=0, split="valid_seen", seed=42) == loader.val_items
    assert adapter.build_eval_env(env_num=0, split="test", seed=42) == loader.test_items
    results = adapter.rollout(train_items, "fixture skill", str(tmp_path / "rollout"))
    assert {row["id"] for row in results} == {"0", "train-1"}
    assert all(row["hard"] == 0 and row["soft"] == 0.0 for row in results)
    assert all(row["fail_reason"] == "template rollout — not implemented" for row in results)
    assert not (tmp_path / "rollout").exists()


def test_template_ratio_split_loads_normalized_data_deterministically(tmp_path):
    raw = tmp_path / "items.jsonl"
    raw.write_text("\n".join(json.dumps({"uid": i, "prompt": f"Task {i}", "answer": i}) for i in range(10)))
    adapters = []
    for name in ("first", "second"):
        adapter = TemplateBenchmarkEnv(data_path=str(raw), split_mode="ratio", split_seed=7)
        adapter.setup({"env": "your_benchmark", "out_root": str(tmp_path / name)})
        adapters.append(adapter)
    first, second = (adapter.get_dataloader() for adapter in adapters)
    assert [len(first.get_split_items(split)) for split in ("train", "val", "test")] == [2, 1, 7]
    for split in ("train", "val", "test"):
        assert first.get_split_items(split) == second.get_split_items(split)
    all_items = first.train_items + first.val_items + first.test_items
    assert {row["id"] for row in all_items} == {str(i) for i in range(10)}
    assert next(row for row in all_items if row["id"] == "0")["ground_truth"] == "0"


def test_template_rejects_missing_task_ids(tmp_path):
    (tmp_path / "items.json").write_text(json.dumps([{"question": "Missing identity"}]))
    with pytest.raises(ValueError, match="non-empty id"):
        TemplateBenchmarkLoader().load_split_items(str(tmp_path))


def test_template_supports_trainer_initialization_and_evaluation(tmp_path, monkeypatch):
    from textualrl.engine.trainer import TextualRLTrainer

    def unexpected_connection(*args, **kwargs):
        pytest.fail("the placeholder scaffold must not make network requests")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
    split_dir = tmp_path / "splits"
    for split in ("train", "val", "test"):
        folder = split_dir / split
        folder.mkdir(parents=True)
        (folder / "items.json").write_text(json.dumps([
            {"id": f"{split}-1", "question": "Placeholder task", "ground_truth": "42"},
        ]))
    config = cli.resolve_config(arguments("--output-dir", str(tmp_path / "run")), environ={})
    config.update(
        num_epochs=0,  # Exercise startup and final evaluation without reflection.
        split_dir=str(split_dir),
        skill_init=str(ROOT / "textualrl/envs/_template/skills/initial.md"),
    )
    adapter = TemplateBenchmarkEnv(split_dir=str(split_dir))
    TextualRLTrainer(config, adapter).train()
    assert (tmp_path / "run/best_skill.md").read_text() == Path(config["skill_init"]).read_text()
    assert (tmp_path / "run/summary.json").is_file()
