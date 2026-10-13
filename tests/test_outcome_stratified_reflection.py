from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from textualrl.config import flatten_config
from textualrl.gradient.reflect import (
    _outcome_stratified_routing_audit,
    _split_task_block_minibatches,
    build_outcome_stratified_task_blocks,
    fmt_minibatch_trajectories,
    run_outcome_stratified_analyst_group,
    run_minibatch_reflect,
)
from textualrl.optimizer.group_relative import (
    annotate_analyst_patch,
    configure_group_relative_edit_credit,
    is_group_relative_edit_credit_enabled,
)


def _row(task_id: str, index: int, hard: int, soft: float = 0.0) -> dict:
    return {
        "id": f"{task_id}__sample_{index:02d}_of_04",
        "rollout_group_id": task_id,
        "rollout_index": index,
        "rollout_count": 4,
        "hard": hard,
        "soft": soft,
        "task_description": f"Answer task {task_id} from visible evidence.",
        "task_type": "qa",
        "same_task_group": {
            "status": "unknown",
            "hard_mean": 0.0,
            "success_count": 0,
            "completed_count": 4,
            "expected_count": 4,
            "outcomes": [],
        },
    }


def _write_conversation(
    prediction_dir: Path,
    row: dict,
    behavior: str,
    *,
    latency: float = 0.0,
) -> None:
    task_dir = prediction_dir / row["id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    conversation = [
        {
            "role": "assistant",
            "content": behavior,
            "latency": latency,
            "usage": {"completion_tokens": int(latency) + 1},
        }
    ]
    (task_dir / "conversation.json").write_text(
        json.dumps(conversation),
        encoding="utf-8",
    )


def _add_task(
    rows: list[dict],
    prediction_dir: Path,
    task_id: str,
    outcomes: list[int],
    behaviors: list[str],
) -> None:
    hard_mean = sum(outcomes) / len(outcomes)
    status = (
        "stable_failure"
        if hard_mean == 0.0
        else "stable_success"
        if hard_mean == 1.0
        else "mixed"
    )
    task_rows = [
        _row(task_id, index, hard, float(hard))
        for index, hard in enumerate(outcomes, 1)
    ]
    sibling_outcomes = [
        {
            "id": row["id"],
            "rollout_index": row["rollout_index"],
            "hard": row["hard"],
            "soft": row["soft"],
        }
        for row in task_rows
    ]
    for row, behavior in zip(task_rows, behaviors, strict=True):
        row["group_relative_hard"] = row["hard"] - hard_mean
        row["same_task_group"] = {
            "status": status,
            "hard_mean": hard_mean,
            "success_count": sum(outcomes),
            "completed_count": len(outcomes),
            "expected_count": len(outcomes),
            "outcomes": sibling_outcomes,
        }
        _write_conversation(
            prediction_dir,
            row,
            behavior,
            latency=float(row["rollout_index"]),
        )
    rows.extend(task_rows)


class OutcomeStratifiedReflectionTest(unittest.TestCase):
    def tearDown(self) -> None:
        configure_group_relative_edit_credit(False)

    def test_routes_and_deduplicates_each_task_without_timing_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            rows: list[dict] = []
            _add_task(
                rows,
                prediction_dir,
                "mixed-task",
                [1, 1, 0, 0],
                ["answer A", "answer A", "answer B", "answer B"],
            )
            _add_task(
                rows,
                prediction_dir,
                "success-task",
                [1, 1, 1, 1],
                ["path A", "path A", "path B", "path B"],
            )
            _add_task(
                rows,
                prediction_dir,
                "failure-task",
                [0, 0, 0, 0],
                ["wrong A", "wrong A", "wrong A", "wrong B"],
            )

            routed = build_outcome_stratified_task_blocks(
                rows,
                str(prediction_dir),
            )

        self.assertEqual(len(routed["mixed"]), 1)
        self.assertEqual(len(routed["stable_success"]), 1)
        self.assertEqual(len(routed["stable_failure"]), 1)
        self.assertEqual(
            [row["_outcome_stratified_multiplicity"] for row in routed["mixed"][0]],
            [2, 2],
        )
        self.assertEqual(
            [
                row["_outcome_stratified_multiplicity"]
                for row in routed["stable_success"][0]
            ],
            [2, 2],
        )
        self.assertEqual(
            [
                row["_outcome_stratified_multiplicity"]
                for row in routed["stable_failure"][0]
            ],
            [3, 1],
        )
        audit = _outcome_stratified_routing_audit(
            routed,
            {"stable_success": [], "stable_failure": []},
        )
        self.assertEqual(audit["route_task_counts"], {
            "mixed": 1,
            "stable_success": 1,
            "stable_failure": 1,
        })
        self.assertEqual(audit["support_unit"], "distinct_task_id")
        self.assertEqual(
            audit["routes"]["mixed"][0]["original_rollout_count"],
            4,
        )

    def test_homogeneous_batches_never_analyze_one_task_alone(self) -> None:
        blocks = [[{"rollout_group_id": f"task-{index}"}] for index in range(9)]

        batches, skipped = _split_task_block_minibatches(blocks, 4)
        singleton_batches, singleton_skipped = _split_task_block_minibatches(
            blocks[:1],
            4,
        )

        self.assertEqual([len(batch) for batch in batches], [4, 5])
        self.assertEqual(skipped, [])
        self.assertEqual(singleton_batches, [])
        self.assertEqual(singleton_skipped, blocks[:1])
        self.assertTrue(all(len(batch) >= 2 for batch in batches))

    def test_dedup_keeps_all_source_rollouts_in_patch_provenance(self) -> None:
        configure_group_relative_edit_credit(True)
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            rows: list[dict] = []
            _add_task(
                rows,
                prediction_dir,
                "success-task",
                [1, 1, 1, 1],
                ["path A", "path A", "path B", "path B"],
            )
            block = build_outcome_stratified_task_blocks(
                rows,
                str(prediction_dir),
            )["stable_success"][0]
            result = {
                "patch": {
                    "edits": [
                        {
                            "op": "append",
                            "content": "Use the visible evidence before answering.",
                            "evidence_task_ids": ["success-task"],
                        }
                    ]
                }
            }

            annotate_analyst_patch(result, block)

        edit = result["patch"]["edits"][0]
        self.assertEqual(len(block), 2)
        self.assertEqual(
            edit["provenance_rollout_ids"],
            sorted(row["id"] for row in rows),
        )
        self.assertEqual(edit["task_support_count"], 1)

    def test_standalone_dispatches_without_group_relative_credit(self) -> None:
        configure_group_relative_edit_credit(
            False,
            outcome_stratified_reflection=True,
            taskwise_edit_budget=1,
        )
        self.assertFalse(is_group_relative_edit_credit_enabled())
        calls: list[tuple[str, tuple[str, ...], int]] = []

        def fake_analyst(
            _skill,
            items,
            _prediction_dir,
            *,
            route,
            edit_budget,
            **_kwargs,
        ):
            task_ids = tuple(sorted({row["rollout_group_id"] for row in items}))
            calls.append((route, task_ids, edit_budget))
            return {
                "source_type": route,
                "patch": {"edits": []},
                "reflection_route": route,
            }

        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            patches_dir = Path(tmp) / "patches"
            rows: list[dict] = []
            _add_task(
                rows,
                prediction_dir,
                "mixed-task",
                [1, 0, 1, 0],
                ["good", "bad", "good alt", "bad alt"],
            )
            for task_id in ("success-a", "success-b"):
                _add_task(
                    rows,
                    prediction_dir,
                    task_id,
                    [1, 1, 1, 1],
                    ["good", "good", "good alt", "good alt"],
                )
            for task_id in ("failure-a", "failure-b"):
                _add_task(
                    rows,
                    prediction_dir,
                    task_id,
                    [0, 0, 0, 0],
                    ["bad", "bad", "bad alt", "bad alt"],
                )

            with patch(
                "textualrl.gradient.reflect.run_outcome_stratified_analyst_group",
                side_effect=fake_analyst,
            ):
                run_minibatch_reflect(
                    rows,
                    "current skill",
                    str(prediction_dir),
                    str(patches_dir),
                    workers=3,
                    failure_only=False,
                    minibatch_size=8,
                    edit_budget=4,
                    random_seed=42,
                )
            routing = json.loads(
                (patches_dir / "outcome_stratified_routing.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertCountEqual(
            calls,
            [
                ("mixed", ("mixed-task",), 1),
                ("stable_success", ("success-a", "success-b"), 4),
                ("stable_failure", ("failure-a", "failure-b"), 4),
            ],
        )
        self.assertEqual(
            set(routing["minibatches"]["stable_success"][0]),
            {"success-a", "success-b"},
        )
        self.assertEqual(
            set(routing["minibatches"]["stable_failure"][0]),
            {"failure-a", "failure-b"},
        )

    def test_standalone_patch_does_not_attach_group_relative_credit(self) -> None:
        configure_group_relative_edit_credit(
            False,
            outcome_stratified_reflection=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            rows: list[dict] = []
            _add_task(
                rows,
                prediction_dir,
                "mixed-task",
                [1, 0, 1, 0],
                ["good", "bad", "good alt", "bad alt"],
            )
            block = build_outcome_stratified_task_blocks(
                rows,
                str(prediction_dir),
            )["mixed"][0]
            with patch(
                "textualrl.gradient.reflect.chat_optimizer",
                return_value=("{}", {}),
            ), patch(
                "textualrl.gradient.reflect.extract_json",
                return_value={
                    "patch": {
                        "edits": [
                            {
                                "op": "append",
                                "content": "Check visible evidence before answering.",
                            }
                        ]
                    }
                },
            ):
                result = run_outcome_stratified_analyst_group(
                    "current skill",
                    block,
                    str(prediction_dir),
                    route="mixed",
                    edit_budget=1,
                    system_prompt="BASE ANALYST",
                )

        self.assertIsNotNone(result)
        edit = result["patch"]["edits"][0]
        self.assertNotIn("_group_relative_credit", edit)
        self.assertEqual(edit["evidence_task_ids"], ["mixed-task"])
        self.assertEqual(
            result["outcome_stratified_source_task_ids"],
            ["mixed-task"],
        )

    def test_formatter_repeats_shared_context_once_per_task_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            rows: list[dict] = []
            _add_task(
                rows,
                prediction_dir,
                "mixed-task",
                [1, 1, 0, 0],
                ["same answer", "same answer", "other answer", "other answer"],
            )
            for row in rows:
                row["target_system_prompt"] = "SHARED SYSTEM PROMPT"
                row["target_user_prompt"] = "SHARED USER PROMPT"
            block = build_outcome_stratified_task_blocks(
                rows,
                str(prediction_dir),
            )["mixed"][0]

            rendered = fmt_minibatch_trajectories(block, str(prediction_dir))

        self.assertEqual(rendered.count("SHARED SYSTEM PROMPT"), 1)
        self.assertEqual(rendered.count("SHARED USER PROMPT"), 1)
        self.assertIn("Unique Trajectory 1/2", rendered)
        self.assertIn("Unique Trajectory 2/2", rendered)
        self.assertEqual(rendered.count("multiplicity=2"), 2)

    def test_mixed_route_is_not_labeled_as_failure_only(self) -> None:
        configure_group_relative_edit_credit(True)
        captured: dict[str, str] = {}

        def fake_chat_optimizer(**kwargs):
            captured["system"] = kwargs["system"]
            captured["user"] = kwargs["user"]
            return "{}", {}

        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            rows: list[dict] = []
            _add_task(
                rows,
                prediction_dir,
                "mixed-task",
                [1, 0, 1, 0],
                ["good", "bad", "good alt", "bad alt"],
            )
            block = build_outcome_stratified_task_blocks(
                rows,
                str(prediction_dir),
            )["mixed"][0]

            with patch(
                "textualrl.gradient.reflect.chat_optimizer",
                side_effect=fake_chat_optimizer,
            ), patch(
                "textualrl.gradient.reflect.extract_json",
                return_value={"patch": {"edits": []}},
            ):
                result = run_outcome_stratified_analyst_group(
                    "current skill",
                    block,
                    str(prediction_dir),
                    route="mixed",
                    edit_budget=4,
                    system_prompt="BASE ANALYST",
                )

        self.assertIsNotNone(result)
        self.assertIn("Outcome-Stratified Route A", captured["system"])
        self.assertIn("## Mixed Sibling Trajectories (4 total)", captured["user"])
        self.assertNotIn("## Failed Trajectories", captured["user"])

    def test_config_flattening_exposes_outcome_stratified_switch(self) -> None:
        flat = flatten_config(
            {
                "optimizer": {
                    "group_relative_outcome_stratified_reflection": True,
                }
            }
        )

        self.assertTrue(flat["group_relative_outcome_stratified_reflection"])


if __name__ == "__main__":
    unittest.main()
