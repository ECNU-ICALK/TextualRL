from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_outcome_stratified_reflection import _add_task
from textualrl.gradient.aggregate import merge_patches
from textualrl.gradient.reflect import run_minibatch_reflect
from textualrl.optimizer.cross_group import (
    collect_cross_group_evidence,
    configure_cross_group_evidence,
)
from textualrl.optimizer.group_relative import configure_group_relative_edit_credit
from textualrl.prompts import load_prompt


def _displayed(rows, user):
    """Find rollout references actually exposed to the mocked analyst."""
    return [row for row in rows if row["id"] in user]


def _card(row, quote="Observe the visible state before retrying."):
    return {
        "task_id": row["rollout_group_id"],
        "rollout_id": row["id"],
        "quote": quote,
        "condition": "A retry is being considered",
        "behavior": "Observe the visible state before retrying",
        "observed_effect": "unknown",
    }


class UnroutedReflectionTest(unittest.TestCase):
    def setUp(self):
        configure_cross_group_evidence(True)
        configure_group_relative_edit_credit(False, outcome_stratified_reflection=False)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.predictions = self.root / "predictions"
        self.rows = []

    def tearDown(self):
        configure_cross_group_evidence(False)
        configure_group_relative_edit_credit(False)

    def add_task(self, task, outcomes):
        start = len(self.rows)
        _add_task(
            self.rows, self.predictions, task, outcomes,
            ["Observe the visible state before retrying."] * len(outcomes),
        )
        for row in self.rows[start:]:
            row["rollout_count"] = len(outcomes)

    def reflect(self, *, minibatch_size=8, cache="patches"):
        return run_minibatch_reflect(
            self.rows, "Current skill", str(self.predictions), str(self.root / cache),
            workers=1, failure_only=False, minibatch_size=minibatch_size,
            edit_budget=4, random_seed=42,
        )

    def test_all_original_rollouts_are_batched_together_with_outcomes_and_cards(self):
        self.add_task("mixed", [1, 0, 1, 0])
        self.add_task("success", [1, 1, 1, 1])
        self.add_task("failure", [0, 0, 0, 0])
        calls = []

        def optimizer(**kwargs):
            displayed = _displayed(self.rows, kwargs["user"])
            calls.append(displayed)
            self.assertIn("cross_group_cards", kwargs["system"])
            self.assertNotIn("exactly one task", kwargs["system"])
            self.assertNotIn("## Failed Trajectories", kwargs["user"])
            self.assertNotIn("## Successful Trajectories", kwargs["user"])
            self.assertGreaterEqual(
                kwargs["user"].count("Observe the visible state before retrying."), len(displayed),
            )
            for row in displayed:
                self.assertIn(row["rollout_group_id"], kwargs["user"])
                self.assertIn(f"hard={row['hard']}", kwargs["user"])
            return json.dumps({
                "patch": {"edits": []},
                "cross_group_cards": [_card(row) for row in displayed],
            }), {}

        with patch("textualrl.gradient.reflect.chat_optimizer", side_effect=optimizer):
            results = self.reflect()

        self.assertEqual(sorted(map(len, calls)), [4, 8])
        self.assertEqual(
            Counter(row["id"] for batch in calls for row in batch),
            Counter(row["id"] for row in self.rows),
        )
        pool = collect_cross_group_evidence(results)
        self.assertEqual(pool["card_count"], 12)
        self.assertEqual(pool["route_counts"], {"unified": 12})
        self.assertEqual(pool["distinct_task_count"], 3)
        actual = {(card["task_id"], card["rollout_id"], card["hard"]) for card in pool["cards"]}
        expected = {(row["rollout_group_id"], row["id"], row["hard"]) for row in self.rows}
        self.assertEqual(actual, expected)

    def test_no_edit_cards_reach_the_final_proposal_review(self):
        self.add_task("mixed-a", [1, 0, 1, 0])
        self.add_task("mixed-b", [1, 0, 1, 0])
        calls = []

        def analyst(**kwargs):
            displayed = _displayed(self.rows, kwargs["user"])
            calls.append(displayed)
            edits = ([{"op": "append", "content": "Always retry."}] if len(calls) == 1 else [])
            return json.dumps({"patch": {"edits": edits},
                               "cross_group_cards": [_card(row) for row in displayed]}), {}

        with patch("textualrl.gradient.reflect.chat_optimizer", side_effect=analyst):
            results = self.reflect(minibatch_size=4)
        pool = collect_cross_group_evidence(results)
        self.assertEqual(pool["card_count"], 8)
        self.assertEqual(sum(bool(result["patch"]["edits"]) for result in results), 1)
        no_edit_result = next(result for result in results if not result["patch"]["edits"])
        evidence = no_edit_result["cross_group_cards"][0]
        reviewed = {
            "edits": [{"op": "append", "content": "Observe the state before retrying.",
                       "cross_group_review": {
                           "decision": "narrow", "reason": "Account for the observed state.",
                           "evidence": [{"task_id": evidence["task_id"],
                                         "rollout_id": evidence["rollout_id"],
                                         "relation": "limits", "reason": "Checks precede the retry."}],
                       }}],
        }

        def reviewer(**kwargs):
            for card in pool["cards"]:
                self.assertIn(card["rollout_id"], kwargs["user"])
            return json.dumps(reviewed), {}

        proposals = [result["patch"] for result in results if result["patch"]["edits"]]
        with patch("textualrl.gradient.aggregate.chat_optimizer", side_effect=reviewer) as review:
            merged = merge_patches("Current skill", proposals, [], verbose=False,
                                   cross_group_evidence=pool)
        review.assert_called_once()
        self.assertEqual(merged["cross_group_audit"]["status"], "model_reviewed")
        valid = merged["edits"][0]["cross_group_review_audit"]["valid_evidence"]
        self.assertEqual([card["rollout_id"] for card in valid], [evidence["rollout_id"]])
        self.assertEqual(valid[0]["route"], "unified")

    def test_hierarchical_merge_uses_unified_instructions_even_without_valid_cards(self):
        for has_cards in (True, False):
            with self.subTest(has_cards=has_cards):
                proposals = [{"edits": [{"op": "append", "content": f"Guidance {index}",
                                         "source_type": "unified"}]} for index in range(3)]
                evidence = {"cards": ([{
                    **_card({"rollout_group_id": "task", "id": "rollout"}),
                    "route": "unified", "hard": 1,
                }] if has_cards else [])}
                calls = []

                def optimizer(**kwargs):
                    calls.append(kwargs)
                    return json.dumps({"edits": [{"op": "append", "content": "Merged guidance",
                                                  "source_type": "unified"}]}), {}

                with patch("textualrl.gradient.aggregate.chat_optimizer", side_effect=optimizer):
                    result = merge_patches(
                        "Current skill", proposals, [], batch_size=2, workers=1, verbose=False,
                        cross_group_evidence=evidence,
                    )
                hierarchical = [call for call in calls if "## Patches to merge" in call["user"]]
                review = [call for call in calls if "## Current-training-step" in call["user"]]
                self.assertEqual(len(hierarchical), 2)
                self.assertEqual(len(review), int(has_cards))
                for call in calls:
                    self.assertIn("Unified Proposal Consolidation", call["system"])
                    self.assertIn('source_type="unified"', call["system"])
                    self.assertIn("relative priority", call["system"])
                for call in hierarchical:
                    self.assertTrue(call["system"].startswith(load_prompt("merge_failure").rstrip()))
                if review:
                    self.assertIn("unified trajectory minibatches", review[0]["system"])
                    self.assertTrue(review[0]["system"].startswith(load_prompt("merge_final").rstrip()))
                self.assertEqual(result["edits"][0]["source_type"], "unified")

    def test_legacy_hierarchical_merge_keeps_the_original_prompt(self):
        proposals = [{"edits": [{"op": "append", "content": f"Guidance {index}",
                                 "source_type": "failure"}]} for index in range(3)]
        with patch("textualrl.gradient.aggregate.chat_optimizer", return_value=(
            json.dumps({"edits": [{"op": "append", "content": "Merged guidance"}]}), {},
        )) as chat:
            merge_patches("Current skill", proposals, [], batch_size=2, workers=1, verbose=False,
                          cross_group_evidence={"cards": []})
        self.assertEqual(chat.call_count, 2)
        for call in chat.call_args_list:
            self.assertEqual(call.kwargs["system"], load_prompt("merge_failure"))

    def test_unified_cards_still_reject_hidden_siblings_and_non_runtime_quotes(self):
        self.add_task("mixed", [1, 0, 1, 0])
        for row in self.rows:
            path = self.predictions / row["id"] / "conversation.json"
            conversation = json.loads(path.read_text())
            conversation.extend([
                {"role": "system", "content": "SECRET GOLD EVALUATOR"},
                {"type": "verification", "content": "POST EXECUTION VERIFICATION"},
                {"type": "evaluation", "content": "POST EXECUTION SCORE"},
            ])
            path.write_text(json.dumps(conversation))

        def optimizer(**kwargs):
            displayed = _displayed(self.rows, kwargs["user"])
            visible = displayed[0]
            hidden = next(row for row in self.rows if row not in displayed)
            return json.dumps({"patch": {"edits": []}, "cross_group_cards": [
                _card(visible), _card(visible), _card(hidden),
                {**_card(visible), "task_id": "unrelated-task"},
                _card(visible, quote="SECRET GOLD EVALUATOR"),
                _card(visible, quote="POST EXECUTION VERIFICATION"),
                _card(visible, quote="POST EXECUTION SCORE"),
                _card(visible, quote="Invented runtime action"),
            ]}), {}

        with patch("textualrl.gradient.reflect.chat_optimizer", side_effect=optimizer):
            results = self.reflect(minibatch_size=3)
        self.assertEqual(len(results), 2)
        self.assertEqual(collect_cross_group_evidence(results)["card_count"], 2)
        for result in results:
            audit = result["cross_group_card_audit"]
            self.assertEqual(audit["valid_card_count"], 1)
            reasons = Counter(row["reason"] for row in audit["excluded_cards"])
            self.assertEqual(reasons["duplicate_trajectory_card"], 1)
            self.assertEqual(reasons["unknown_or_undisplayed_trajectory"], 2)
            self.assertEqual(reasons["quote_not_in_runtime_trajectory"], 4)

    def test_single_trajectory_tail_and_no_edit_cards_survive_resume(self):
        self.add_task("mixed-a", [1, 0, 1, 0])
        self.add_task("mixed-b", [1, 0, 1, 0])
        batch_sizes = []

        def optimizer(**kwargs):
            displayed = _displayed(self.rows, kwargs["user"])
            batch_sizes.append(len(displayed))
            return json.dumps({"patch": {"edits": []},
                               "cross_group_cards": [_card(row) for row in displayed]}), {}

        with patch("textualrl.gradient.reflect.chat_optimizer", side_effect=optimizer) as chat:
            first = self.reflect(minibatch_size=7)
            second = self.reflect(minibatch_size=7)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(sorted(batch_sizes), [1, 7])
        self.assertEqual(collect_cross_group_evidence(first), collect_cross_group_evidence(second))
        self.assertEqual(collect_cross_group_evidence(second)["card_count"], 8)

    def test_single_rollout_keeps_existing_analysis_for_both_routing_settings(self):
        for task, outcome in [("success-a", 1), ("success-b", 1), ("failure-a", 0), ("failure-b", 0)]:
            self.add_task(task, [outcome])
        for routing in (False, True):
            with self.subTest(outcome_stratified=routing):
                configure_group_relative_edit_credit(False, outcome_stratified_reflection=routing)
                calls = []

                def optimizer(**kwargs):
                    calls.append(_displayed(self.rows, kwargs["user"]))
                    self.assertNotIn("cross_group_cards", kwargs["system"])
                    return json.dumps({"patch": {"edits": []}}), {}

                with patch("textualrl.gradient.reflect.chat_optimizer", side_effect=optimizer):
                    results = self.reflect(cache=f"single-{routing}")
                self.assertEqual(sorted(map(len, calls)), [2, 2])
                self.assertTrue(all(len({row["hard"] for row in batch}) == 1 for batch in calls))
                self.assertEqual(collect_cross_group_evidence(results)["card_count"], 0)

    def test_disabling_cross_group_evidence_preserves_legacy_representatives(self):
        configure_cross_group_evidence(False)
        self.add_task("mixed-a", [1, 0, 1, 0])
        self.add_task("mixed-b", [1, 0, 1, 0])
        calls = []

        def optimizer(**kwargs):
            calls.append(_displayed(self.rows, kwargs["user"]))
            self.assertNotIn("cross_group_cards", kwargs["system"])
            return json.dumps({"patch": {"edits": []}}), {}

        with patch("textualrl.gradient.reflect.chat_optimizer", side_effect=optimizer):
            results = self.reflect()
        self.assertEqual(sorted(map(len, calls)), [2, 2])
        self.assertEqual(sum(map(len, calls)), 4)
        self.assertTrue(all(len({row["hard"] for row in batch}) == 1 for batch in calls))
        self.assertEqual(collect_cross_group_evidence(results)["card_count"], 0)


if __name__ == "__main__":
    unittest.main()
