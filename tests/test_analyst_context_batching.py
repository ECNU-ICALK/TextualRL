import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_outcome_stratified_reflection import _add_task
from textualrl.gradient.context_batching import (
    is_context_length_error,
    run_context_bounded_analyst,
)
from textualrl.gradient.reflect import (
    run_error_analyst_minibatch,
    run_success_analyst_minibatch,
    run_minibatch_reflect,
)
from textualrl.optimizer.group_relative import configure_group_relative_edit_credit


OVERFLOW = "Qwen chat API returned HTTP 400: This model's maximum context length is 1010000 tokens"


def rows(n):
    return [
        {"id": f"task-{task}-{sample}", "rollout_group_id": f"task-{task}"}
        for task in range(n) for sample in range(4)
    ]


class AnalystContextBatchingTest(unittest.TestCase):
    def tearDown(self):
        configure_group_relative_edit_credit(False)

    def test_only_explicit_context_400_triggers_split(self):
        self.assertTrue(is_context_length_error(RuntimeError(OVERFLOW)))
        for message in ["HTTP 504 stream timeout", "HTTP 400 invalid temperature",
                        "context note", "HTTP 500 maximum context length"]:
            self.assertFalse(is_context_length_error(RuntimeError(message)))

    def test_fitting_batch_is_unchanged_and_cached(self):
        evidence = rows(8)
        result = {"patch": {"edits": []}, "cross_group_cards": [{"quote": "kept"}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch(__name__ + ".dummy_call", return_value=result) as call:
                first = run_context_bounded_analyst(evidence, 4, "stable_failure", call, tmp, "batch")
                call.assert_called_once_with(evidence, 4)
                second = run_context_bounded_analyst(evidence, 4, "stable_failure", call, tmp, "batch")
                self.assertEqual(call.call_count, 1)
                self.assertEqual(first, second)
                self.assertEqual(first[0][1], result)
                self.assertFalse(list(Path(tmp).glob("*_context_split.json")))

    def test_overflow_preserves_siblings_and_divides_parent_budget(self):
        evidence = rows(8)
        calls = []

        def call(items, budget):
            calls.append((items, budget))
            if len(items) > 16:
                raise RuntimeError(OVERFLOW)
            return {"patch": {"edits": [{"content": "rule"}] * budget},
                    "cross_group_cards": [{"rollout_id": item["id"]} for item in items]}

        with tempfile.TemporaryDirectory() as tmp:
            result = run_context_bounded_analyst(evidence, 4, "stable_failure", call, tmp, "batch")
            self.assertEqual([budget for _, budget in calls], [4, 2, 2])
            self.assertEqual([len(items) for items, _ in calls], [32, 16, 16])
            self.assertEqual([row for items, _ in calls[1:] for row in items], evidence)
            self.assertTrue(all(len({x["rollout_group_id"] for x in part}) == 4 for part, _ in calls[1:]))
            self.assertEqual(sum(len(record[1]["patch"]["edits"]) for record in result), 4)
            self.assertEqual(sum(len(record[1]["cross_group_cards"]) for record in result), 32)
            self.assertTrue((Path(tmp) / "batch_context_split.json").exists())

    def test_recursive_split_preserves_total_budget(self):
        calls = []

        def call(items, budget):
            calls.append((len(items), budget))
            if len(items) > 8:
                raise RuntimeError(OVERFLOW)
            return {"patch": {"edits": [{"content": "rule"}]}}

        with tempfile.TemporaryDirectory() as tmp:
            result = run_context_bounded_analyst(rows(8), 4, "stable_success", call, tmp, "batch")
        self.assertEqual([record[2:] for record in result], [(2, 8)] * 4)
        self.assertEqual(sum(budget for size, budget in calls if size == 8), 4)

    def test_interrupted_child_resumes_without_repeating_parent_or_sibling(self):
        calls = []

        def interrupted(items, budget):
            calls.append(len(items))
            if len(items) > 16:
                raise RuntimeError(OVERFLOW)
            if items[0]["rollout_group_id"] == "task-4":
                raise TimeoutError("temporary transport failure")
            return {"patch": {"edits": []}}

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):
                run_context_bounded_analyst(rows(8), 4, "stable_failure", interrupted, tmp, "batch")
            with patch(__name__ + ".dummy_call", return_value={"patch": {"edits": []}}) as retry:
                result = run_context_bounded_analyst(rows(8), 4, "stable_failure", retry, tmp, "batch")
                retry.assert_called_once_with(rows(8)[16:], 2)
                self.assertEqual(len(result), 2)

    def test_cannot_weaken_two_task_rule_or_split_a_mixed_task(self):
        for route, n, budget in [("mixed", 1, 1), ("stable_failure", 2, 4),
                                 ("stable_success", 3, 4), ("stable_failure", 8, 1)]:
            with self.subTest(route=route, n=n, budget=budget), tempfile.TemporaryDirectory() as tmp:
                with patch(__name__ + ".dummy_call", side_effect=RuntimeError(OVERFLOW)) as call:
                    with self.assertRaisesRegex(RuntimeError, "Cannot split"):
                        run_context_bounded_analyst(rows(n), budget, route, call, tmp, "batch")
                    self.assertEqual(call.call_count, 1)

    def test_other_errors_are_not_converted_to_empty_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(__name__ + ".dummy_call", side_effect=TimeoutError("timeout")) as call:
                with self.assertRaises(TimeoutError):
                    run_context_bounded_analyst(rows(8), 4, "stable_failure", call, tmp, "batch")
                self.assertFalse(list(Path(tmp).iterdir()))

    def test_analysts_can_propagate_errors_to_runtime_handler(self):
        for analyst in [run_error_analyst_minibatch, run_success_analyst_minibatch]:
            with patch("textualrl.gradient.reflect.fmt_minibatch_trajectories", return_value="trace"), \
                    patch("textualrl.gradient.reflect.chat_optimizer", side_effect=RuntimeError(OVERFLOW)):
                with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                    analyst("skill", rows(2), "unused", raise_errors=True)

    def test_dispatcher_keeps_leaf_patches_for_original_aggregation(self):
        configure_group_relative_edit_credit(False, outcome_stratified_reflection=True)
        calls = []

        def analyst(_skill, items, _pred, *, edit_budget, **kwargs):
            calls.append((len(items), edit_budget))
            if len(items) > 16:
                raise RuntimeError(OVERFLOW)
            return {"source_type": "failure", "patch": {"edits": [{"content": "rule"}]},
                    "outcome_stratified_source_task_ids": list({x["rollout_group_id"] for x in items})}

        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions"
            patches = Path(tmp) / "patches"
            evidence = []
            for task in range(8):
                _add_task(evidence, pred, f"task-{task}", [0] * 4,
                          [f"action-{sample}" for sample in range(4)])
            with patch("textualrl.gradient.reflect.run_outcome_stratified_analyst_group", side_effect=analyst):
                result = run_minibatch_reflect(evidence, "skill", str(pred), str(patches),
                                              workers=1, failure_only=False, minibatch_size=8, edit_budget=4,
                                              random_seed=42)
                resumed = run_minibatch_reflect(evidence, "skill", str(pred), str(patches),
                                               workers=1, failure_only=False, minibatch_size=8, edit_budget=4,
                                               random_seed=42)
            self.assertEqual(calls, [(32, 4), (16, 2), (16, 2)])
            self.assertEqual(len(result), 2)
            self.assertEqual(result, resumed)

    def test_resume_rejects_a_split_plan_for_different_evidence(self):
        def call(items, budget):
            if len(items) > 16:
                raise RuntimeError(OVERFLOW)
            return {"patch": {"edits": []}}

        with tempfile.TemporaryDirectory() as tmp:
            run_context_bounded_analyst(rows(8), 4, "stable_failure", call, tmp, "batch")
            plan_path = Path(tmp) / "batch_context_split.json"
            plan = json.loads(plan_path.read_text())
            plan["rollout_ids"][0] = "different-trajectory"
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "does not match evidence"):
                run_context_bounded_analyst(rows(8), 4, "stable_failure", call, tmp, "batch")


def dummy_call(items, budget):
    raise AssertionError("must be mocked")


if __name__ == "__main__":
    unittest.main()
