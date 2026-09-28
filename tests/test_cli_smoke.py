"""Run real trainer/evaluator code against a local, deterministic HTTP fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml


def test_training_evaluation_and_completed_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            if self.path.endswith("/responses"):
                text = json.dumps({"reasoning": "Fixture: no change needed", "edits": [], "evidence_cards": []})
                result = {"id": "fixture", "object": "response", "status": "completed",
                          "output": [{"type": "message", "role": "assistant", "content": [
                              {"type": "output_text", "text": text}]}],
                          "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
            else:
                result = {"choices": [{"message": {"role": "assistant", "content": "<answer>42</answer>"},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
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
        data = tmp_path / "data" / "searchqa" / "splits"
        for split, size in (("train", 2), ("val", 1), ("test", 1)):
            folder = data / split
            folder.mkdir(parents=True)
            items = [{"id": f"{split}-{i}", "question": "What number?", "context": "The number is 42.",
                      "answers": ["42"]} for i in range(size)]
            (folder / "items.json").write_text(json.dumps(items))
        cfg = yaml.safe_load((root / "configs/searchqa.yaml").read_text())
        cfg.update(num_epochs=1, train_size=2, batch_size=2, same_task_rollouts=2,
                   use_meta_skill=False, use_slow_update=False, workers=2, analyst_workers=1,
                   exec_timeout=5, target_qwen_chat_timeout_seconds=5,
                   optimizer_qwen_chat_timeout_seconds=5)
        config = tmp_path / "smoke.yaml"
        config.write_text(yaml.safe_dump(cfg))
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        env = dict(os.environ, TARGET_API_KEY="local-fixture", OPTIMIZER_API_KEY="local-fixture",
                   TARGET_BASE_URL=endpoint, OPTIMIZER_BASE_URL=endpoint, NO_PROXY="127.0.0.1,localhost")
        out = tmp_path / "train"
        common = ["--config", str(config), "--data-dir", str(tmp_path / "data")]

        def execute(*args):
            result = subprocess.run([sys.executable, "-m", "textualrl", *args], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=45)
            assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
            return result

        execute("train", *common, "--output-dir", str(out))
        assert (out / "best_skill.md").is_file()
        assert (out / "runtime_state.json").is_file()
        target_calls = [body for path, body in calls if path.endswith("chat/completions")]
        optimizer_calls = [body for path, body in calls if path.endswith("responses")]
        assert target_calls and optimizer_calls
        assert {body["temperature"] for body in target_calls} == {0.0, 0.7}
        assert all(body["chat_template_kwargs"]["enable_thinking"] is False for body in target_calls)
        assert all(body["reasoning"]["effort"] == "medium" for body in optimizer_calls)
        assert all("max_output_tokens" not in body for body in optimizer_calls)
        before_resume = len(calls)
        execute("train", *common, "--resume-from", str(out))
        # A completed run may refresh final evaluation, but cannot replay training.
        assert all(body.get("temperature") != 0.7 for _, body in calls[before_resume:])
        before_eval = len(calls)
        execute("eval", *common, "--output-dir", str(tmp_path / "eval"),
                "--skill", str(out / "best_skill.md"), "--split", "test")
        assert len(calls) > before_eval
        assert all(body["temperature"] == 0.0 for _, body in calls[before_eval:])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
