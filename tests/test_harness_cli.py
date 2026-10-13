"""Exercise public exec-target commands with native CLI and optimizer fixtures."""

import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXEC_BACKENDS = ("codex_exec", "claude_code_exec")
SKILL = "# Fixture guidance\n\nGround each answer in the supplied context. Fixture skill marker.\n"
OPTIMIZER_SECRET = "offline-optimizer-fixture-secret"


def _environment(**values):
    # Do not inherit installed CLI settings, credentials, or remote endpoints.
    result = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR") if name in os.environ}
    result["NO_PROXY"] = "127.0.0.1,localhost"
    result.update(values)
    return result


def _execute(*args, env, expected_code=0, script=None):
    command = ([sys.executable, "-c", script] if script else [sys.executable, "-m", "textualrl"])
    result = subprocess.run(command + list(args), cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == expected_code, result.stdout[-8000:] + result.stderr[-8000:]
    return result


def _json_lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def native_cli(tmp_path):
    """A real executable that reads exactly the workspace presented to the CLI."""
    captures = tmp_path / "native-calls"
    captures.mkdir()
    executable = tmp_path / "fake-native-cli"
    executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent(r'''
        import json
        import os
        from pathlib import Path
        import sys
        import uuid

        args = sys.argv[1:]
        work = Path.cwd()
        task = (work / "task.md").read_text()
        skill = (work / ".agents/skills/textualrl-target/SKILL.md").read_text()
        assert "What is the capital of France?" in task
        assert "Paris is the capital of France." in task
        assert "Fixture skill marker" in skill
        assert 'name: "textualrl-target"' in skill
        for name in ("TARGET_API_KEY", "OPTIMIZER_API_KEY", "TARGET_QWEN_CHAT_API_KEY",
                     "OPTIMIZER_QWEN_CHAT_API_KEY", "QWEN_CHAT_API_KEY"):
            assert name not in os.environ, name + " leaked into native CLI environment"
        answer = "<answer>Paris</answer>"
        if args[0] == "exec":
            assert "--output-last-message" in args
            assert Path(args[args.index("-C") + 1]).resolve() == work.resolve()
            final_path = Path(args[args.index("--output-last-message") + 1])
            assert final_path.parent.resolve() == work.resolve()
            final_path.write_text(answer)
            model = args[args.index("-m") + 1]
            backend = "codex_exec"
        else:
            assert args[0] == "-p"
            assert args[args.index("--output-format") + 1] == "text"
            assert ".agents/skills/textualrl-target/SKILL.md" in args[-1]
            model = args[args.index("--model") + 1]
            backend = "claude_code_exec"
        record = {"backend": backend, "args": args, "model": model,
                  "cwd": str(work), "task": task, "skill": skill}
        capture = Path(os.environ["FIXTURE_NATIVE_CALLS"]) / (uuid.uuid4().hex + ".json")
        capture.write_text(json.dumps(record))
        # Codex must consume --output-last-message, not this deliberately untagged trace.
        print("Fixture native CLI finished." if backend == "codex_exec" else answer)
    '''))
    executable.chmod(0o755)
    return executable, captures


@pytest.fixture
def optimizer_http():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "body": body,
                          "authorization": self.headers.get("Authorization")})
            text = json.dumps({"reasoning": "Fixture: guidance already succeeds", "edits": [],
                               "evidence_cards": []})
            result = {"id": "fixture", "object": "response", "status": "completed",
                      "output": [{"type": "message", "role": "assistant", "content": [
                          {"type": "output_text", "text": text}]}],
                      "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
            encoded = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _config(tmp_path, **overrides):
    skill = tmp_path / "initial.md"
    skill.write_text(SKILL)
    config = yaml.safe_load((ROOT / "configs/searchqa.yaml").read_text())
    config.update(num_epochs=1, train_size=2, batch_size=2, same_task_rollouts=2,
                  use_meta_skill=False, use_slow_update=False, workers=2, analyst_workers=1,
                  exec_timeout=5, target_qwen_chat_timeout_seconds=5,
                  optimizer_qwen_chat_timeout_seconds=5, skill_init=str(skill))
    config.update(overrides)
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _native_records(directory):
    return [json.loads(path.read_text()) for path in directory.glob("*.json")]


@pytest.mark.parametrize("backend", EXEC_BACKENDS)
def test_native_target_train_eval_and_completed_resume(tmp_path, native_cli, optimizer_http, backend):
    executable, captures = native_cli
    endpoint, http_calls = optimizer_http
    for split, size in (("train", 2), ("val", 1), ("test", 1)):
        folder = tmp_path / "data" / "searchqa" / "splits" / split
        folder.mkdir(parents=True)
        rows = [{"id": f"{split}-{i}", "question": "What is the capital of France?",
                 "context": "Paris is the capital of France.", "answers": ["Paris"]}
                for i in range(size)]
        (folder / "items.json").write_text(json.dumps(rows))
    config = _config(tmp_path, target_backend=backend,
                     codex_exec_path=str(executable), codex_exec_use_sdk="cli",
                     codex_exec_full_auto=False, codex_exec_sandbox="read-only",
                     claude_code_exec_path=str(executable), claude_code_exec_use_sdk="cli")
    env = _environment(OPTIMIZER_API_KEY=OPTIMIZER_SECRET, TARGET_BASE_URL=endpoint + "/target/v1",
                       OPTIMIZER_BASE_URL=endpoint + "/optimizer/v1", FIXTURE_NATIVE_CALLS=str(captures))
    model = "fixture-native-model"
    common = ["--config", str(config), "--data-dir", str(tmp_path / "data"),
              "--target-model", model, "--optimizer-model", "fixture-optimizer-model"]
    # Exercise the command-line override as well as a YAML backend selection.
    if backend == "codex_exec":
        config_data = yaml.safe_load(config.read_text())
        config_data["target_backend"] = "qwen_chat"
        config.write_text(yaml.safe_dump(config_data))
        common.extend(["--target-backend", backend])
    output = tmp_path / "train"
    trained = _execute("train", *common, "--output-dir", str(output), env=env)
    assert (output / "best_skill.md").read_text() == SKILL
    state = json.loads((output / "runtime_state.json").read_text())
    assert state["last_completed_step"] == 1
    records = _native_records(captures)
    assert len(records) >= 4
    assert {record["backend"] for record in records} == {backend}
    assert {record["model"] for record in records} == {model}
    for record in records:
        assert Path(record["cwd"]).resolve().is_relative_to(output.resolve())
        assert SKILL.strip() in record["skill"]
        if backend == "codex_exec":
            arguments = record["args"]
            assert arguments[arguments.index("--sandbox") + 1] == "read-only"
            assert "--full-auto" not in arguments
    assert http_calls, "The real optimizer must run, rather than being replaced with a stub trainer"
    assert {call["path"] for call in http_calls} == {"/optimizer/v1/responses"}
    assert {call["authorization"] for call in http_calls} == {f"Bearer {OPTIMIZER_SECRET}"}
    assert {call["body"]["model"] for call in http_calls} == {"fixture-optimizer-model"}
    audit = _json_lines(output / "runtime_requests.jsonl")
    assert len(audit) == len(http_calls)
    assert {row["role"] for row in audit} == {"optimizer"}
    effective = json.loads((output / "effective_config.json").read_text())
    assert effective["target_backend"] == backend
    assert effective["optimizer_backend"] == "qwen_chat"
    for path in (output / "effective_config.json", output / "config.json", output / "runtime_requests.jsonl"):
        if path.exists():
            assert OPTIMIZER_SECRET not in path.read_text()
    assert OPTIMIZER_SECRET not in trained.stdout + trained.stderr

    before_resume = set(captures.glob("*.json"))
    before_http = len(http_calls)
    _execute("train", *common, "--resume-from", str(output), env=env)
    # Completed runs may refresh selection/test evaluation, but may not replay training.
    resumed = [json.loads(path.read_text()) for path in set(captures.glob("*.json")) - before_resume]
    assert all(not Path(row["cwd"]).resolve().is_relative_to((output / "steps").resolve())
               for row in resumed)
    assert len(http_calls) == before_http
    assert len(_json_lines(output / "runtime_requests.jsonl")) == len(audit)
    before_eval = len(_native_records(captures))

    eval_env = dict(env)
    del eval_env["OPTIMIZER_API_KEY"]
    evaluation = tmp_path / "eval"
    _execute("eval", *common, "--output-dir", str(evaluation),
             "--skill", str(output / "best_skill.md"), "--split", "test", env=eval_env)
    summary = json.loads((evaluation / "eval_summary.json").read_text())
    assert summary["n_items"] == 1 and summary["hard"] == 1.0
    assert len(_native_records(captures)) == before_eval + 1
    assert len(http_calls) == before_http
    assert not _json_lines(evaluation / "runtime_requests.jsonl")


@pytest.mark.parametrize("backend", EXEC_BACKENDS)
@pytest.mark.parametrize("action", ("train", "eval"))
def test_native_dry_run_does_not_import_models_or_start_processes(tmp_path, backend, action):
    config = _config(tmp_path, target_backend=backend,
                     codex_exec_path=str(tmp_path / "missing-codex"),
                     claude_code_exec_path=str(tmp_path / "missing-claude"))
    output = tmp_path / "must-not-exist"
    script = ("from textualrl.cli import main; import sys; main(sys.argv[1:]); "
              "assert not any(k == 'textualrl.model' or k.startswith('textualrl.model.') "
              "for k in sys.modules)")
    result = _execute(action, "--config", str(config), "--output-dir", str(output),
                      "--data-dir", str(tmp_path / "missing-data"), "--dry-run", script=script,
                      env=_environment(TARGET_API_KEY="unprinted-target-secret",
                                       OPTIMIZER_API_KEY="unprinted-optimizer-secret"))
    resolved = json.loads(result.stdout)
    assert resolved["config"]["target_backend"] == backend
    assert resolved["required_credentials"] == (["OPTIMIZER_API_KEY"] if action == "train" else [])
    assert all("/chat/completions" not in endpoint for endpoint in resolved["endpoint_limits"])
    assert not output.exists()
    assert "unprinted-" not in result.stdout + result.stderr


@pytest.mark.parametrize("backend,action,credentials,missing", [
    ("codex_exec", "train", {}, ["OPTIMIZER_API_KEY"]),
    ("claude_code_exec", "train", {"TARGET_API_KEY": "unused-target"}, ["OPTIMIZER_API_KEY"]),
    ("qwen_chat", "train", {"OPTIMIZER_API_KEY": "optimizer"}, ["TARGET_API_KEY"]),
    ("qwen_chat", "train", {"TARGET_API_KEY": "target"}, ["OPTIMIZER_API_KEY"]),
    ("qwen_chat", "eval", {"OPTIMIZER_API_KEY": "optimizer"}, ["TARGET_API_KEY"]),
])
def test_required_credentials_are_checked_before_execution(tmp_path, backend, action, credentials, missing):
    config = _config(tmp_path, target_backend=backend)
    output = tmp_path / "must-not-exist"
    result = _execute(action, "--config", str(config), "--output-dir", str(output),
                      env=_environment(**credentials), expected_code=2)
    assert "Missing required environment variable" in result.stderr
    for name in missing:
        assert name in result.stderr
    if backend in EXEC_BACKENDS:
        assert "TARGET_API_KEY" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("backend", EXEC_BACKENDS)
@pytest.mark.parametrize("environment,overrides,diagnostic", [
    ("alfworld", {}, "alfworld"),
    ("spreadsheetbench", {"mode": "react"}, "single"),
])
def test_unsupported_exec_environments_fail_during_config_resolution(tmp_path, backend, environment,
                                                                   overrides, diagnostic):
    config = _config(tmp_path, target_backend=backend, env=environment, **overrides)
    output = tmp_path / "must-not-exist"
    result = _execute("train", "--config", str(config), "--output-dir", str(output), "--dry-run",
                      env=_environment(), expected_code=2)
    assert diagnostic in result.stderr.lower()
    assert not output.exists()
