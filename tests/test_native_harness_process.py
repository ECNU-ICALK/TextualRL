"""Exercise native harness boundaries with local CLI subprocess fixtures."""

import json
import os
import sys
import textwrap
from unittest.mock import patch

import pytest

from textualrl.model import codex_harness as harness


@pytest.fixture
def native_cli(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace with spaces"
    harness.prepare_workspace(
        work_dir=str(workspace),
        skill_md=harness.render_skill_md("Use the fixture guidance."),
        task_text="Answer the fixture task.",
    )
    executable = tmp_path / "fake native cli"
    executable.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import sys

            workspace = Path.cwd()
            fixture = json.loads((workspace / "process_fixture.json").read_text())
            args = sys.argv[1:]
            last_message = None
            if "--output-last-message" in args:
                last_message = Path(args[args.index("--output-last-message") + 1])
            record = {
                "argv": args,
                "cwd": str(workspace),
                "skill": (workspace / ".agents/skills/textualrl-target/SKILL.md").read_text(),
                "task": (workspace / "task.md").read_text(),
                "env": {name: os.environ[name] for name in fixture.get("inspect_env", [])
                        if name in os.environ},
                "last_message_existed": last_message.exists() if last_message else None,
            }
            with (workspace / "process_calls.jsonl").open("a") as stream:
                print(json.dumps(record), file=stream)
            if last_message is not None and fixture.get("answer") is not None:
                last_message.write_text(fixture["answer"])
            stdout = fixture.get("stdout", "")
            if last_message is None and fixture.get("answer") is not None:
                stdout = fixture["answer"]
            sys.stdout.write(stdout)
            sys.stderr.write(fixture.get("stderr", ""))
            sys.exit(fixture.get("exit_code", 0))
            """),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    codex_config = {
        "path": str(executable), "profile": "", "use_sdk": "cli",
        "reasoning_effort": "medium", "sandbox": "workspace-write",
        "full_auto": False, "network_access": False, "web_search": False,
        "approval_policy": "never", "empty_response_retries": 0,
    }
    claude_config = {
        "path": str(executable), "profile": "", "use_sdk": "cli",
        "effort": "medium", "max_thinking_tokens": 0, "empty_response_retries": 0,
    }
    monkeypatch.setattr(harness, "get_codex_exec_config", lambda: codex_config)
    monkeypatch.setattr(harness, "get_claude_code_exec_config", lambda: claude_config)

    def unexpected_sdk(**kwargs):
        pytest.fail("Native CLI mode must not attempt an SDK call")

    monkeypatch.setattr(harness, "_run_codex_sdk_exec", unexpected_sdk)
    monkeypatch.setattr(harness, "_run_claude_code_sdk_exec", unexpected_sdk)

    def configure(**fixture):
        (workspace / "process_fixture.json").write_text(json.dumps(fixture), encoding="utf-8")

    def calls():
        return [json.loads(line) for line in (workspace / "process_calls.jsonl").read_text().splitlines()]

    configure(answer="<answer>42</answer>")
    return workspace, codex_config, claude_config, configure, calls


def invoke(backend, workspace):
    runner = harness.run_codex_exec if backend == "codex" else harness.run_claude_code_exec
    return runner(work_dir=str(workspace), prompt="Read task.md and the local skill.",
                  model="fixture-model", timeout=10)


@pytest.mark.parametrize(
    "network_access,web_search,approval_policy,sandbox",
    [(False, False, "never", "workspace-write"), (True, True, "on-request", "read-only")],
)
def test_codex_forwards_execution_settings_to_real_process(
    native_cli, network_access, web_search, approval_policy, sandbox,
):
    workspace, config, _, _, calls = native_cli
    config.update(network_access=network_access, web_search=web_search,
                  approval_policy=approval_policy, sandbox=sandbox)
    response, _ = invoke("codex", workspace)
    assert response == "<answer>42</answer>"
    record = calls()[0]
    argv = record["argv"]
    assert argv[0] == "exec"
    assert argv[argv.index("--sandbox") + 1] == sandbox
    assert "--full-auto" not in argv
    assert argv[argv.index("-C") + 1] == str(workspace)
    assert argv[argv.index("-m") + 1] == "fixture-model"
    overrides = dict(argv[index + 1].split("=", 1) for index, arg in enumerate(argv) if arg == "-c")
    assert overrides["approval_policy"] == json.dumps(approval_policy)
    assert overrides["sandbox_workspace_write.network_access"] == str(network_access).lower()
    assert overrides["web_search"] == json.dumps("live" if web_search else "disabled")
    assert record["cwd"] == str(workspace)
    assert "Use the fixture guidance." in record["skill"]
    assert record["task"] == "Answer the fixture task."


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_child_credentials_are_isolated_without_changing_parent(native_cli, backend):
    workspace, _, _, configure, calls = native_cli
    excluded = {
        "OPTIMIZER_API_KEY": "fixture-optimizer-secret",
        "OPTIMIZER_QWEN_CHAT_API_KEY": "fixture-qwen-optimizer-secret",
        "OPTIMIZER_OPENAI_COMPATIBLE_API_KEY": "fixture-compatible-optimizer-secret",
        "OPTIMIZER_BASE_URL": "https://optimizer.invalid/v1",
        "OPTIMIZER_QWEN_CHAT_MODEL": "fixture-optimizer-model",
        "TARGET_API_KEY": "fixture-target-secret",
        "TARGET_BASE_URL": "https://target.invalid/v1",
        "TARGET_QWEN_CHAT_API_KEY": "fixture-qwen-target-secret",
        "QWEN_CHAT_API_KEY": "fixture-generic-secret",
    }
    preserved = {
        "OPENAI_API_KEY": "fixture-native-openai-secret",
        "ANTHROPIC_API_KEY": "fixture-native-anthropic-secret",
        "ANTHROPIC_BASE_URL": "https://native.invalid/v1",
        "CODEX_HOME": str(workspace / "native-auth"),
        "HTTPS_PROXY": "http://proxy.invalid:8080",
    }
    environment = {**excluded, **preserved}
    configure(answer="<answer>42</answer>", inspect_env=list(environment))
    with patch.dict(os.environ, environment, clear=True):
        before = dict(os.environ)
        response, _ = invoke(backend, workspace)
        assert dict(os.environ) == before
    assert response == "<answer>42</answer>"
    assert calls()[0]["env"] == preserved


def test_claude_nonzero_exit_with_stdout_is_not_an_answer(native_cli):
    workspace, _, _, configure, calls = native_cli
    configure(stdout="API Error: 401 unauthorized", exit_code=1)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        invoke("claude", workspace)
    assert len(calls()) == 1
    assert "API Error: 401 unauthorized" in (workspace.parent / "claude_raw.txt").read_text()


def test_codex_removes_previous_final_message_before_starting_process(native_cli):
    workspace, _, _, configure, calls = native_cli
    last_message = workspace / "codex_last_message.txt"
    last_message.write_text("<answer>stale answer</answer>", encoding="utf-8")
    configure(stdout="Trace without a new final answer.")
    response, raw = harness._run_codex_cli_exec(
        work_dir=str(workspace), prompt="Second turn", model="fixture-model", timeout=10,
    )
    assert calls()[0]["last_message_existed"] is False
    assert not last_message.exists()
    assert response == ""
    assert raw == "Trace without a new final answer."


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_empty_final_responses_exhaust_retries_and_raise_instead_of_returning_trace(native_cli, backend):
    workspace, codex_config, claude_config, configure, calls = native_cli
    config = codex_config if backend == "codex" else claude_config
    config["empty_response_retries"] = 2
    configure(stderr="Trace contains <answer>not a final response</answer>.")
    with pytest.raises(RuntimeError, match="no final response after retries"):
        invoke(backend, workspace)
    records = calls()
    assert len(records) == 3
    assert "Previous execution returned an empty final response" in records[-1]["argv"][-1]
    raw = (workspace.parent / f"{backend}_raw.txt").read_text()
    assert "not a final response" in raw
    assert raw.count(f"{backend.upper()} CLI ATTEMPT") == 3
