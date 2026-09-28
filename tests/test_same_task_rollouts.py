from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textualrl.envs.alfworld.adapter import (
    ALFWorldAdapter,
    ALFWorldBatchRun,
)
from textualrl.envs.alfworld.rollout import (
    annotate_same_task_groups as annotate_alfworld_groups,
)
from textualrl.envs.docvqa.adapter import DocVQAAdapter
from textualrl.envs.docvqa.rollout import annotate_same_task_groups as annotate_docvqa_groups
from textualrl.envs.livemathematicianbench.adapter import LiveMathematicianBenchAdapter
from textualrl.envs.livemathematicianbench.rollout import (
    annotate_same_task_groups as annotate_livemath_groups,
)
from textualrl.envs.searchqa.adapter import SearchQAAdapter
from textualrl.envs.searchqa.rollout import annotate_same_task_groups, run_batch
from textualrl.gradient.reflect import _select_same_task_group_representatives


class SameTaskRolloutTest(unittest.TestCase):
    def test_searchqa_expansion_uses_unique_execution_ids(self) -> None:
        adapter = object.__new__(SearchQAAdapter)
        items = [
            {
                "id": "task-a",
                "question": "Question?",
                "context": "Context",
                "answers": ["Answer"],
            },
            {
                "id": "task-b",
                "question": "Other?",
                "context": "Other context",
                "answers": ["Other answer"],
            },
        ]

        expanded = adapter.expand_same_task_rollouts(items, 4)

        self.assertEqual(len(expanded), 8)
        self.assertEqual(len({row["id"] for row in expanded}), 8)
        self.assertEqual(
            [row["rollout_group_id"] for row in expanded[:4]],
            ["task-a"] * 4,
        )
        self.assertEqual(
            [row["rollout_index"] for row in expanded[:4]],
            [1, 2, 3, 4],
        )
        self.assertTrue(all(row["rollout_count"] == 4 for row in expanded))

    def test_docvqa_expansion_uses_unique_execution_ids(self) -> None:
        adapter = object.__new__(DocVQAAdapter)
        item = {
            "id": "doc-question",
            "question": "What is the total?",
            "image_path": "/tmp/document.png",
            "answers": ["42"],
        }

        expanded = adapter.expand_same_task_rollouts([item], 4)

        self.assertEqual(len(expanded), 4)
        self.assertEqual(len({row["id"] for row in expanded}), 4)
        self.assertEqual(
            [row["rollout_group_id"] for row in expanded],
            ["doc-question"] * 4,
        )
        self.assertEqual(
            [row["rollout_index"] for row in expanded],
            [1, 2, 3, 4],
        )

    def test_docvqa_group_annotation_computes_advantage(self) -> None:
        results = [
            {
                "id": f"doc-question__sample_{idx:02d}_of_04",
                "rollout_group_id": "doc-question",
                "rollout_index": idx,
                "rollout_count": 4,
                "hard": hard,
                "soft": float(hard),
            }
            for idx, hard in enumerate([1, 0, 1, 0], 1)
        ]

        annotate_docvqa_groups(results)

        self.assertEqual(results[0]["same_task_group"]["status"], "mixed")
        self.assertEqual(results[0]["same_task_group"]["hard_mean"], 0.5)
        self.assertEqual(results[0]["group_relative_hard"], 0.5)
        self.assertEqual(results[1]["group_relative_hard"], -0.5)

    def test_livemath_expansion_and_group_advantage(self) -> None:
        adapter = object.__new__(LiveMathematicianBenchAdapter)
        item = {
            "id": "math-question",
            "question": "Which option is correct?",
            "choices": [
                {"label": "A", "text": "First"},
                {"label": "B", "text": "Second"},
            ],
            "correct_choice": {"label": "A", "text": "First"},
        }

        expanded = adapter.expand_same_task_rollouts([item], 4)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(len({row["id"] for row in expanded}), 4)
        self.assertEqual(
            [row["rollout_group_id"] for row in expanded],
            ["math-question"] * 4,
        )

        results = [
            {
                **row,
                "hard": hard,
                "soft": float(hard),
                "response": f"response-{idx}",
            }
            for idx, (row, hard) in enumerate(
                zip(expanded, [1, 0, 1, 0]),
                1,
            )
        ]
        annotate_livemath_groups(results)

        self.assertEqual(results[0]["same_task_group"]["status"], "mixed")
        self.assertEqual(results[0]["group_relative_hard"], 0.5)
        self.assertEqual(results[1]["group_relative_hard"], -0.5)

    def test_alfworld_expansion_and_group_advantage(self) -> None:
        adapter = object.__new__(ALFWorldAdapter)
        batch = ALFWorldBatchRun(
            env_num=1,
            eval_dataset="train",
            seed=42,
            is_train=True,
            workers=4,
            specific_gamefiles=["json_2.1.1/train/task/game.tw-pddl"],
            result_ids=["train:0001"],
            items=[
                {
                    "id": "train:0001",
                    "gamefile": "json_2.1.1/train/task/game.tw-pddl",
                    "task_type": "pick_and_place_simple",
                }
            ],
        )

        expanded = adapter.expand_same_task_rollouts(batch, 4)

        self.assertIsInstance(expanded, ALFWorldBatchRun)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(len(set(expanded.result_ids or [])), 4)
        self.assertEqual(
            [row["rollout_group_id"] for row in (expanded.items or [])],
            ["train:0001"] * 4,
        )
        self.assertEqual(
            [row["rollout_index"] for row in (expanded.items or [])],
            [1, 2, 3, 4],
        )
        self.assertEqual(len(expanded.specific_gamefiles or []), 4)

        results = [
            {
                "id": row["id"],
                "rollout_group_id": row["rollout_group_id"],
                "rollout_index": row["rollout_index"],
                "rollout_count": row["rollout_count"],
                "hard": hard,
                "soft": float(hard),
                "n_turns": idx,
                "fail_reason": "" if hard else "incomplete",
            }
            for idx, (row, hard) in enumerate(
                zip(expanded.items or [], [1, 0, 1, 0]),
                1,
            )
        ]
        annotate_alfworld_groups(results)
        self.assertEqual(results[0]["same_task_group"]["status"], "mixed")
        self.assertEqual(results[0]["group_relative_hard"], 0.5)
        self.assertEqual(results[1]["group_relative_hard"], -0.5)

    def test_alfworld_adapter_uses_train_and_eval_temperatures(self) -> None:
        adapter = object.__new__(ALFWorldAdapter)
        adapter.max_steps = 8
        adapter.workers = 4
        adapter.max_api_workers = 4
        adapter.max_completion_tokens = 128
        adapter.same_task_rollout_temperature = 0.7
        adapter.evaluation_target_temperature = 0.0
        grouped = ALFWorldBatchRun(
            env_num=1,
            eval_dataset="train",
            seed=42,
            is_train=True,
            workers=1,
            items=[
                {
                    "id": "sample",
                    "rollout_group_id": "task",
                    "rollout_index": 1,
                    "rollout_count": 4,
                }
            ],
        )
        evaluation = ALFWorldBatchRun(
            env_num=1,
            eval_dataset="eval_in_distribution",
            seed=42,
            is_train=False,
            workers=1,
            items=[{"id": "task"}],
        )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            adapter,
            "_run_batch",
            return_value=[],
        ) as run:
            adapter.rollout(grouped, "skill", str(Path(tmp) / "grouped"))
            self.assertEqual(run.call_args.kwargs["target_temperature"], 0.7)
            adapter.rollout(evaluation, "skill", str(Path(tmp) / "eval"))
            self.assertEqual(run.call_args.kwargs["target_temperature"], 0.0)

    def test_group_annotation_records_mixed_outcomes_and_advantage(self) -> None:
        results = [
            {
                "id": f"task-a__sample_{idx:02d}_of_04",
                "rollout_group_id": "task-a",
                "rollout_index": idx,
                "rollout_count": 4,
                "hard": hard,
                "soft": float(hard),
                "predicted_answer": f"answer-{idx}",
                "response": f"response-{idx}",
            }
            for idx, hard in enumerate([1, 0, 1, 0], 1)
        ]

        annotate_same_task_groups(results)

        summary = results[0]["same_task_group"]
        self.assertEqual(summary["status"], "mixed")
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 2)
        self.assertEqual(summary["hard_mean"], 0.5)
        self.assertEqual(results[0]["group_relative_hard"], 0.5)
        self.assertEqual(results[1]["group_relative_hard"], -0.5)

    def test_reflection_counts_each_task_group_once_per_outcome(self) -> None:
        results = []
        for group_id, outcomes in (("task-a", [1, 0, 1, 0]), ("task-b", [0, 0, 0, 0])):
            for idx, hard in enumerate(outcomes, 1):
                results.append(
                    {
                        "id": f"{group_id}__sample_{idx:02d}_of_04",
                        "rollout_group_id": group_id,
                        "rollout_index": idx,
                        "rollout_count": 4,
                        "hard": hard,
                        "soft": float(hard),
                    }
                )
        annotate_same_task_groups(results)

        failures, grouped = _select_same_task_group_representatives(
            results,
            successful=False,
        )
        successes, _ = _select_same_task_group_representatives(
            results,
            successful=True,
        )

        self.assertTrue(grouped)
        self.assertEqual(
            {row["rollout_group_id"] for row in failures},
            {"task-a", "task-b"},
        )
        self.assertEqual(
            {row["rollout_group_id"] for row in successes},
            {"task-a"},
        )

    def test_batch_execution_persists_group_audit(self) -> None:
        adapter = object.__new__(SearchQAAdapter)
        expanded = adapter.expand_same_task_rollouts(
            [{"id": "task-a", "question": "Q", "answers": ["A"]}],
            4,
        )

        def fake_process(item, *_args, **_kwargs):
            hard = int(item["rollout_index"] in {1, 3})
            return {
                "id": item["id"],
                "rollout_group_id": item["rollout_group_id"],
                "rollout_index": item["rollout_index"],
                "rollout_count": item["rollout_count"],
                "hard": hard,
                "soft": float(hard),
                "predicted_answer": f"answer-{item['rollout_index']}",
                "response": f"response-{item['rollout_index']}",
                "agent_ok": True,
                "fail_reason": "" if hard else "EM=0",
            }

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "textualrl.envs.searchqa.rollout.process_one",
                side_effect=fake_process,
            ):
                results = run_batch(expanded, tmp, "skill", workers=4)

            audit = json.loads(
                (Path(tmp) / "same_task_groups.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(results), 4)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["status"], "mixed")
        self.assertEqual(audit[0]["success_count"], 2)

    def test_searchqa_adapter_uses_train_and_eval_temperatures(self) -> None:
        adapter = object.__new__(SearchQAAdapter)
        adapter.max_turns = 1
        adapter.exec_timeout = 120
        adapter.workers = 4
        adapter.max_completion_tokens = 128
        adapter.same_task_rollout_temperature = 0.7
        adapter.evaluation_target_temperature = 0.0
        grouped = [
            {
                "id": "task-a__sample_01_of_04",
                "rollout_group_id": "task-a",
                "rollout_index": 1,
                "rollout_count": 4,
            }
        ]
        evaluation = [{"id": "task-a"}]

        with patch(
            "textualrl.envs.searchqa.adapter.run_batch",
            return_value=[],
        ) as batch:
            adapter.rollout(grouped, "skill", "/tmp/grouped")
            self.assertEqual(batch.call_args.kwargs["target_temperature"], 0.7)

            adapter.rollout(evaluation, "skill", "/tmp/eval")
            self.assertEqual(batch.call_args.kwargs["target_temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
