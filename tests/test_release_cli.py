"""Offline behavioral tests for the public launcher and its real wire payloads."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import yaml

from textualrl import cli
from textualrl.runtime import _safe_failure, configure_environment, endpoint_limits, request_runtime

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def real_backend():
    """Load the real urllib backend without importing unrelated SDK backends."""
    with patch.dict(sys.modules):
        for name, filename in (("textualrl.model.common", "common.py"), ("_release_test_qwen", "qwen_backend.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / "textualrl/model" / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield module


class ReleaseCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.config_path = self.root / "preset.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "env": "livemathematicianbench",
            "data_path": "${TEXTUALRL_DATA_DIR}/math.jsonl",
            "out_root": "${TEXTUALRL_OUTPUT_DIR}/math",
            "skill_init": "${TEXTUALRL_REPO_ROOT}/textualrl/envs/livemathematicianbench/skills/initial.md",
            "target_model": "${TEXTUALRL_TARGET_MODEL}",
            "optimizer_model": "${TEXTUALRL_OPTIMIZER_MODEL}",
            "target_qwen_chat_base_url": "${TARGET_BASE_URL}",
            "optimizer_qwen_chat_base_url": "${OPTIMIZER_BASE_URL}",
        }), encoding="utf-8")

    def arguments(self, action="train", *extra):
        return cli.parser().parse_args([action, "--config", str(self.config_path), *extra])

    def configuration(self, action="train", *extra):
        args = self.arguments(action, *extra)
        return args, cli.resolve_config(args, environ={})

    def test_substitution_and_exact_overrides(self):
        args, config = self.configuration("train", "--data-dir", str(self.root / "data: with spaces"),
                                          "--output-dir", str(self.root / "run"),
                                          "--target-model", "deployment:custom")
        self.assertEqual(config["data_path"], str(self.root / "data: with spaces/math.jsonl"))
        self.assertEqual(config["out_root"], str(self.root / "run"))
        self.assertEqual(config["target_model"], "deployment:custom")
        self.assertEqual(config["optimizer_model"], "gpt-5.5")
        self.assertEqual(config["optimizer_qwen_chat_temperature"], None)

    def test_empty_environment_reference_and_inline_secret_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "TARGET_BASE_URL"):
            cli.resolve_config(self.arguments(), environ={"TARGET_BASE_URL": ""})
        self.config_path.write_text("env: math\ntarget_qwen_chat_api_key: do-not-print-me\n")
        with self.assertRaises(ValueError) as caught:
            cli.resolve_config(self.arguments(), environ={})
        self.assertNotIn("do-not-print-me", str(caught.exception))
        self.config_path.write_text("env: math\ndata_path: '${TARGET_API_KEY}'\n")
        with self.assertRaisesRegex(ValueError, "Unsupported configuration variable"):
            cli.resolve_config(self.arguments(), environ={"TARGET_API_KEY": "secret"})

    def test_dry_run_needs_neither_keys_data_nor_backend_imports(self):
        output = self.root / "never-created"
        process = subprocess.run(
            [sys.executable, "-c", "from textualrl.cli import main; import sys; main(sys.argv[1:]); assert 'textualrl.model' not in sys.modules",
             "train", "--config", str(self.config_path), "--output-dir", str(output), "--dry-run"],
            cwd=ROOT, env={"PATH": os.environ.get("PATH", ""), "TARGET_API_KEY": "do-not-print-target",
                           "OPTIMIZER_API_KEY": "do-not-print-optimizer"},
            text=True, capture_output=True, check=True,
        )
        result = json.loads(process.stdout)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["config"]["optimizer_model"], "gpt-5.5")
        self.assertFalse(output.exists())
        self.assertNotIn("do-not-print", process.stdout + process.stderr)

    def test_credentials_are_separate_and_proxy_is_preserved(self):
        _, config = self.configuration()
        with patch.dict(os.environ, {"TARGET_API_KEY": "target-secret", "QWEN_CHAT_API_KEY": "generic-secret",
                                    "OPTIMIZER_QWEN_CHAT_API_KEY": "stale-secret", "HTTPS_PROXY": "http://proxy.invalid:8080"}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPTIMIZER_API_KEY"):
                configure_environment(config, "train")
            configure_environment(config, "eval")
            self.assertNotIn("QWEN_CHAT_API_KEY", os.environ)
            self.assertEqual(os.environ["TARGET_QWEN_CHAT_API_KEY"], "target-secret")
            self.assertEqual(os.environ["OPTIMIZER_QWEN_CHAT_API_KEY"], "")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.invalid:8080")
            with real_backend() as backend:
                self.assertEqual(backend.TARGET_CONFIG.api_key, "target-secret")
                self.assertEqual(backend.OPTIMIZER_CONFIG.api_key, "")

    def test_real_payloads_keep_role_settings_and_audit_excludes_content(self):
        _, config = self.configuration()
        sent = []
        audit = self.root / "requests.jsonl"
        with patch.dict(os.environ, {"TARGET_API_KEY": "target-secret", "OPTIMIZER_API_KEY": "optimizer-secret"}, clear=True):
            configure_environment(config, "train")
            with real_backend() as backend:
                def transport(payload, timeout, settings):
                    sent.append((payload, settings))
                    if settings.api_style == "responses":
                        return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "private-result"}]}], "usage": {"output_tokens": 4, "secret": "optimizer-secret"}}
                    return {"choices": [{"message": {"content": "private-result"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 4}}
                backend._post_chat_completion = transport
                with request_runtime(backend, config, audit):
                    for role in ("target", "optimizer"):
                        with backend.target_generation_overrides(temperature=0, seed=42):
                            backend._chat_messages_impl([{"role": "system", "content": "private-instructions"}, {"role": "user", "content": "target-secret optimizer-secret private-prompt"}], 99, 1, "test", role=role)
        target, optimizer = sent
        self.assertEqual(target[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(target[0]["seed"], 42)
        self.assertEqual(target[0]["temperature"], 0)
        self.assertEqual(target[1].api_key, "target-secret")
        self.assertEqual(optimizer[1].api_key, "optimizer-secret")
        self.assertEqual(optimizer[0]["reasoning"], {"effort": "medium"})
        self.assertFalse(optimizer[0]["store"])
        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "seed"):
            self.assertNotIn(name, optimizer[0])
        text = audit.read_text()
        for private in ("target-secret", "optimizer-secret", "private-prompt", "private-instructions", "private-result"):
            self.assertNotIn(private, text)
        self.assertEqual([json.loads(row)["role"] for row in text.splitlines()], ["target", "optimizer"])

    def test_actual_concurrency_is_shared_and_errors_are_safe(self):
        _, config = self.configuration()
        config.update(optimizer_qwen_chat_base_url=config["target_qwen_chat_base_url"],
                      optimizer_api_style="chat_completions", target_endpoint_concurrency=2,
                      optimizer_endpoint_concurrency=3)
        self.assertEqual(list(endpoint_limits(config).values()), [2])
        audit = self.root / "requests.jsonl"
        with patch.dict(os.environ, {"TARGET_API_KEY": "target-secret", "OPTIMIZER_API_KEY": "optimizer-secret"}, clear=True):
            configure_environment(config, "train")
            with real_backend() as backend:
                active = peak = 0
                lock = threading.Lock()
                def transport(payload, timeout, settings):
                    nonlocal active, peak
                    with lock:
                        active += 1
                        peak = max(peak, active)
                    try:
                        time.sleep(0.01)
                        if payload.get("fail"):
                            raise RuntimeError("echoed target-secret private-prompt")
                        return {"choices": [{"message": {"content": "ok"}}]}
                    finally:
                        with lock:
                            active -= 1
                backend._post_chat_completion = transport
                with request_runtime(backend, config, audit) as counts:
                    with ThreadPoolExecutor(max_workers=10) as pool:
                        list(pool.map(lambda i: backend._post_chat_completion({}, None, backend.TARGET_CONFIG if i % 2 else backend.OPTIMIZER_CONFIG), range(20)))
                    with self.assertRaises(RuntimeError) as caught:
                        backend._post_chat_completion({"fail": True}, None, backend.TARGET_CONFIG)
                    self.assertNotIn("target-secret", str(caught.exception))
                    row = next(iter(counts.values()))
                    self.assertEqual(row, {"active": 0, "peak": 2, "calls": 21, "errors": 1})
                self.assertIs(backend._post_chat_completion, transport)
                self.assertEqual(peak, 2)
        self.assertNotIn("target-secret", audit.read_text())
        self.assertTrue(json.loads(audit.read_text().splitlines()[-1])["request_failed"])

    def test_dispatch_forwards_models_and_evaluation_controls(self):
        args, config = self.configuration("eval", "--output-dir", str(self.root / "eval"))
        with patch.dict(os.environ, {"TARGET_API_KEY": "target-secret"}, clear=True), real_backend() as backend:
            model = ModuleType("textualrl.model")
            model.qwen_backend = backend
            model.set_target_backend = lambda value: None
            model.set_optimizer_backend = lambda value: None
            model.target_generation_overrides = backend.target_generation_overrides
            package = ModuleType("textualrl")
            package.model = model
            def launch(path, run_name):
                self.assertEqual(Path(path).name, "eval_only.py")
                self.assertEqual(sys.argv[sys.argv.index("--optimizer_model") + 1], "gpt-5.5")
                self.assertEqual(sys.argv[sys.argv.index("--split") + 1], "test")
                self.assertEqual(backend._target_temperature_override, 0)
                self.assertEqual(backend._target_seed_override, 42)
                self.assertEqual(backend.TARGET_CONFIG.api_key, "target-secret")
                self.assertEqual(backend.OPTIMIZER_CONFIG.api_key, "")
            with patch.dict(sys.modules, {"textualrl": package, "textualrl.model": model}), patch.object(cli.runpy, "run_path", side_effect=launch) as run_path:
                cli.run(args, config)
                run_path.assert_called_once()
        self.assertNotIn("target-secret", (self.root / "eval/effective_config.json").read_text())

    def test_resume_reuses_existing_run_directory(self):
        run = self.root / "prior"
        run.mkdir()
        (run / "runtime_state.json").write_text('{"last_completed_step": 1}')
        _, config = self.configuration("train", "--resume-from", str(run))
        self.assertEqual(config["out_root"], str(run))
        with self.assertRaisesRegex(ValueError, "same run directory"):
            self.configuration("train", "--resume-from", str(run), "--output-dir", str(self.root / "new"))

    def test_public_presets_initialize_real_backend_without_output_caps(self):
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            with self.subTest(preset=path.name):
                args = cli.parser().parse_args(["train", "--config", str(path), "--dry-run"])
                config = cli.resolve_config(args, environ={})
                with patch.dict(os.environ, {"TARGET_API_KEY": "target-secret", "OPTIMIZER_API_KEY": "optimizer-secret"}, clear=True):
                    configure_environment(config, "train")
                    with real_backend() as backend:
                        self.assertFalse(backend.TARGET_CONFIG.enable_thinking)
                        self.assertTrue(backend.OPTIMIZER_CONFIG.omit_output_limit)
                        self.assertEqual(backend.OPTIMIZER_CONFIG.api_style, "responses")
                        self.assertEqual(backend.OPTIMIZER_CONFIG.temperature, None)
                        self.assertEqual(backend.OPTIMIZER_CONFIG.deployment, "gpt-5.5")

    def test_sanitized_overflow_still_triggers_core_context_split(self):
        spec = importlib.util.spec_from_file_location("_release_context_batching", ROOT / "textualrl/gradient/context_batching.py")
        batching = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batching)
        _, config = self.configuration()
        items = [{"id": f"task{task}-rollout{sample}", "rollout_group_id": f"task{task}"}
                 for task in range(4) for sample in range(2)]
        attempts = []
        analyzed = []
        audit = self.root / "overflow_requests.jsonl"

        def http_transport(request, timeout):
            payload = json.loads(request.data)
            rows = json.loads(payload["input"][0]["content"])
            attempts.append(len(rows))
            if len(rows) > 4:
                body = b'{"error":{"code":"context_length_exceeded","message":"maximum context length; optimizer-secret private-prompt"}}'
                raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))
            text = json.dumps({"patch": {"edits": []}})
            response = {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}
            return io.BytesIO(json.dumps(response).encode())

        with patch.dict(os.environ, {"TARGET_API_KEY": "target-secret", "OPTIMIZER_API_KEY": "optimizer-secret"}, clear=True):
            configure_environment(config, "train")
            with real_backend() as backend, patch.object(backend.urllib.request, "urlopen", side_effect=http_transport), patch.object(backend.time, "sleep"):
                def analyze(rows, budget):
                    analyzed.append((list(rows), budget))
                    text, _ = backend._chat_messages_impl(
                        [{"role": "user", "content": json.dumps(rows)}], 99, 2, "analyst", role="optimizer")
                    return json.loads(text)
                with request_runtime(backend, config, audit):
                    leaves = batching.run_context_bounded_analyst(items, 4, "stable_failure", analyze, self.root / "patches", "group")
        self.assertEqual(attempts, [8, 8, 4, 4])  # The core's original retry count is preserved.
        self.assertEqual(len(leaves), 2)
        self.assertEqual([entry[1] for entry in analyzed], [4, 2, 2])
        self.assertEqual([len({row["rollout_group_id"] for row in rows}) for rows, _ in analyzed[1:]], [2, 2])
        self.assertEqual([row for rows, _ in analyzed[1:] for row in rows], items)
        plan_text = (self.root / "patches/group_context_split.json").read_text()
        plan = json.loads(plan_text)
        self.assertTrue(batching.is_context_length_error(RuntimeError(plan["api_error"])))
        self.assertEqual(plan["child_edit_budgets"], [2, 2])
        for secret in ("optimizer-secret", "private-prompt"):
            self.assertNotIn(secret, plan_text + audit.read_text())
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        self.assertEqual(records[0]["http_status"], 400)
        self.assertEqual(records[0]["error_reason"], "context_length_exceeded")
        for status in (400, 401, 403, 429, 500, 504):
            error = _safe_failure("optimizer", HTTPError("https://example.invalid", status, "private-prompt", {}, None))
            self.assertEqual(error.code, status)
            self.assertEqual(error.status_code, status)
            self.assertFalse(batching.is_context_length_error(error))
            self.assertNotIn("private-prompt", str(error))

    @unittest.skipUnless(importlib.util.find_spec("openai"), "full engine integration requires the OpenAI SDK")
    def test_actual_eval_script_runs_offline_on_one_synthetic_item(self):
        for split in ("train", "val", "test"):
            directory = self.root / "splits" / split
            directory.mkdir(parents=True)
            rows = [{"id": "tiny-qa", "question": "What is the capital of France?", "context": "Paris is the capital of France.", "answers": ["Paris"]}] if split == "test" else []
            (directory / "items.json").write_text(json.dumps(rows))
        config = {
            "env": "searchqa", "split_mode": "split_dir", "split_dir": str(self.root / "splits"),
            "skill_init": str(ROOT / "textualrl/envs/searchqa/skills/initial.md"),
            "workers": 1, "max_turns": 1, "max_completion_tokens": 32,
        }
        self.config_path.write_text(yaml.safe_dump(config))
        output = self.root / "actual-eval"
        script = textwrap.dedent('''
            import io, json, socket, sys
            from pathlib import Path
            from unittest.mock import patch
            from textualrl.cli import main
            seen = []
            def transport(request, timeout):
                payload = json.loads(request.data)
                assert request.get_header("Authorization") == "Bearer target-secret"
                assert payload["model"] == "Qwen3.8-27B"
                assert payload["chat_template_kwargs"] == {"enable_thinking": False}
                assert payload["temperature"] == 0 and payload["seed"] == 42
                seen.append(payload)
                return io.BytesIO(json.dumps({"choices": [{"message": {"content": "<answer>Paris</answer>"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 5}}).encode())
            with patch("urllib.request.urlopen", side_effect=transport), patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                assert main(["eval", "--config", sys.argv[1], "--output-dir", sys.argv[2]]) == 0
            output = Path(sys.argv[2])
            summary = json.loads((output / "eval_summary.json").read_text())
            assert summary["n_items"] == 1 and summary["hard"] == 1.0
            assert len(seen) == 1
            evidence = (output / "effective_config.json").read_text() + (output / "runtime_requests.jsonl").read_text()
            assert "target-secret" not in evidence
            print("EVAL_INTEGRATION_OK")
        ''')
        completed = subprocess.run([sys.executable, "-c", script, str(self.config_path), str(output)],
                                   cwd=ROOT, env={"PATH": os.environ.get("PATH", ""), "TARGET_API_KEY": "target-secret"},
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("EVAL_INTEGRATION_OK", completed.stdout)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "full engine integration requires the OpenAI SDK")
    def test_actual_train_script_dispatch_preserves_explicit_model_names(self):
        script = textwrap.dedent('''
            import socket, sys
            from unittest.mock import patch
            from textualrl.cli import main
            from textualrl.engine.trainer import TextualRLTrainer
            from textualrl.model import qwen_backend
            seen = []
            def inspect_dispatch(trainer):
                assert trainer.cfg["target_model"] == "Qwen3.8-27B"
                assert trainer.cfg["optimizer_model"] == "gpt-5.5"
                assert type(trainer.adapter).__name__ == "SearchQAAdapter"
                assert qwen_backend.TARGET_CONFIG.api_key == "target-secret"
                assert qwen_backend.OPTIMIZER_CONFIG.api_key == "optimizer-secret"
                assert qwen_backend.OPTIMIZER_CONFIG.api_style == "responses"
                assert qwen_backend.OPTIMIZER_CONFIG.omit_output_limit
                seen.append(trainer.cfg)
                return {}
            with patch.object(TextualRLTrainer, "train", inspect_dispatch), patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                assert main(["train", "--config", "configs/searchqa.yaml", "--output-dir", sys.argv[1]]) == 0
            assert len(seen) == 1
            print("TRAIN_DISPATCH_OK")
        ''')
        completed = subprocess.run([sys.executable, "-c", script, str(self.root / "train-dispatch")],
                                   cwd=ROOT, env={"PATH": os.environ.get("PATH", ""), "TARGET_API_KEY": "target-secret", "OPTIMIZER_API_KEY": "optimizer-secret"},
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("TRAIN_DISPATCH_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
