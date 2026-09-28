from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from scripts.eval_only import parse_args as parse_eval_args
from scripts.train import _LEGACY_TO_STRUCTURED, parse_args
from textualrl.config import flatten_config
from textualrl.envs.docvqa.adapter import DocVQAAdapter
from textualrl.envs.livemathematicianbench.adapter import (
    LiveMathematicianBenchAdapter,
)
from textualrl.envs.officeqa.adapter import OfficeQAAdapter
from textualrl.envs.searchqa.adapter import SearchQAAdapter
from textualrl.model import qwen_backend


class EvaluationGenerationSeedTest(unittest.TestCase):
    def test_cli_and_structured_config_expose_evaluation_seed(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                "configs/_base_/default.yaml",
                "--evaluation_target_seed",
                "42",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.evaluation_target_seed, 42)
        self.assertEqual(
            _LEGACY_TO_STRUCTURED["evaluation_target_seed"],
            "evaluation.target_seed",
        )
        self.assertEqual(
            flatten_config({"evaluation": {"target_seed": 42}})[
                "evaluation_target_seed"
            ],
            42,
        )

    def test_eval_only_exposes_evaluation_generation_controls(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "eval_only.py",
                "--config",
                "run/config.json",
                "--skill",
                "run/best_skill.md",
                "--evaluation_target_temperature",
                "0.0",
                "--evaluation_target_seed",
                "42",
            ],
        ):
            args = parse_eval_args()

        self.assertEqual(args.evaluation_target_temperature, 0.0)
        self.assertEqual(args.evaluation_target_seed, 42)

    def test_qwen_target_override_reaches_payload_and_restores(self) -> None:
        payloads: list[dict] = []

        def fake_post(payload, timeout, config):
            del timeout, config
            payloads.append(dict(payload))
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }

        with patch.object(qwen_backend, "_post_chat_completion", fake_post):
            with qwen_backend.target_generation_overrides(
                temperature=0.0,
                seed=42,
            ):
                qwen_backend.chat_target("system", "user", retries=1)
            qwen_backend.chat_target(
                "system",
                "user",
                retries=1,
                temperature=0.7,
            )

        self.assertEqual(payloads[0]["temperature"], 0.0)
        self.assertEqual(payloads[0]["seed"], 42)
        self.assertEqual(payloads[1]["temperature"], 0.7)
        self.assertNotIn("seed", payloads[1])

    def _assert_adapter_scopes_seed(self, adapter_cls) -> None:
        adapter = object.__new__(adapter_cls)
        adapter.same_task_rollout_temperature = 0.7
        adapter.evaluation_target_temperature = 0.0
        adapter.evaluation_target_seed = 42
        adapter.workers = 1
        adapter.max_completion_tokens = 32
        if adapter_cls in (SearchQAAdapter, DocVQAAdapter):
            adapter.max_turns = 1
            adapter.exec_timeout = 30
            adapter.image_detail = "low"
        elif adapter_cls is LiveMathematicianBenchAdapter:
            adapter.max_turns = 1
            adapter.exec_timeout = 30
            adapter.use_theorem = False
            adapter.use_sketch = False
        else:
            adapter.max_tool_turns = 1
            adapter.search_mode = "offline"
            adapter.max_queries_per_turn = 1
            adapter.search_api_url = ""
            adapter.search_auth_env = "AUTH"
            adapter.search_provider = "offline"
            adapter.search_max_num_results = 1
            adapter.search_timeout_seconds = 1
            adapter.use_local_tools = True
            adapter.data_dirs = []

        observed: list[tuple[float | None, int | None]] = []

        @contextmanager
        def capture(*, temperature=None, seed=None):
            observed.append((temperature, seed))
            yield

        module_name = adapter_cls.__module__
        with tempfile.TemporaryDirectory() as tmp, patch(
            f"{module_name}.target_generation_overrides",
            capture,
        ), patch(f"{module_name}.run_batch", return_value=[]):
            adapter.rollout([{"id": "eval"}], "skill", tmp)
            adapter.rollout(
                [{"id": "train", "rollout_count": 4}],
                "skill",
                tmp,
            )

        self.assertEqual(observed, [(0.0, 42), (0.7, None)])

    def test_searchqa_scopes_seed_only_to_evaluation(self) -> None:
        self._assert_adapter_scopes_seed(SearchQAAdapter)

    def test_officeqa_scopes_seed_only_to_evaluation(self) -> None:
        self._assert_adapter_scopes_seed(OfficeQAAdapter)

    def test_docvqa_scopes_seed_only_to_evaluation(self) -> None:
        self._assert_adapter_scopes_seed(DocVQAAdapter)

    def test_livemath_scopes_seed_only_to_evaluation(self) -> None:
        self._assert_adapter_scopes_seed(LiveMathematicianBenchAdapter)


if __name__ == "__main__":
    unittest.main()
