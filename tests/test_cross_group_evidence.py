from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_outcome_stratified_reflection import _add_task
from textualrl.config import flatten_config
from textualrl.gradient.aggregate import merge_patches
from textualrl.gradient.reflect import (
    build_outcome_stratified_task_blocks, fmt_trajectory,
    run_minibatch_reflect, run_outcome_stratified_analyst_group,
)
from textualrl.optimizer.cross_group import (
    ANALYST_SUFFIX, MERGE_SUFFIX, annotate_cross_group_review,
    attach_cross_group_cards, collect_cross_group_evidence,
    configure_cross_group_evidence, cross_group_evidence_enabled,
    validate_cross_group_config,
)
from textualrl.optimizer.group_relative import configure_group_relative_edit_credit


def _card(task="success-a", rollout="success-a__sample_01_of_04", quote="wait for updated state"):
    return {"task_id": task, "rollout_id": rollout, "quote": quote,
            "condition": "State can change asynchronously", "behavior": "Wait and check",
            "observed_effect": "The state changes"}


def _pool():
    return {"cards": [
        {**_card(), "route": "stable_success", "hard": 1},
        {**_card("mixed", "mixed__sample_02_of_04", "repeat with no change"),
         "route": "mixed", "hard": 0},
        {**_card("failure", "failure__sample_01_of_04", "repeat with no change"),
         "route": "stable_failure", "hard": 0},
    ]}


class CrossGroupEvidenceTest(unittest.TestCase):
    def setUp(self):
        configure_cross_group_evidence(False)
        configure_group_relative_edit_credit(False, outcome_stratified_reflection=True)

    def tearDown(self):
        configure_cross_group_evidence(False)
        configure_group_relative_edit_credit(False)

    def test_switch_defaults_off_and_does_not_enable_other_methods(self):
        self.assertFalse(cross_group_evidence_enabled())
        flat = flatten_config({"optimizer": {"use_cross_group_evidence": True}})
        self.assertEqual(flat, {"use_cross_group_evidence": True})
        cfg = {"use_cross_group_evidence": True,
               "group_relative_outcome_stratified_reflection": True,
               "use_meta_skill": False, "use_slow_update": False,
               "same_task_rollouts": 4, "skill_update_mode": "patch"}
        original = deepcopy(cfg)
        validate_cross_group_config(cfg)
        self.assertEqual(cfg, original)
        with self.assertRaises(ValueError):
            validate_cross_group_config({"use_cross_group_evidence": True})
        with self.assertRaises(TypeError):
            validate_cross_group_config({"use_cross_group_evidence": "false"})
        with self.assertRaises(ValueError):
            validate_cross_group_config({**cfg, "use_group_relative_edit_credit": True})
        with self.assertRaises(ValueError):
            validate_cross_group_config({**cfg, "skill_update_mode": "rewrite"})

    def test_sources_exclude_evaluator_and_undisplayed_siblings_without_dropping_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            _add_task(rows, Path(tmp), "success-a", [1, 1, 1, 1],
                      ["wait for updated state"] * 4)
            displayed = build_outcome_stratified_task_blocks(rows, tmp)["stable_success"][0]
            path = Path(tmp) / rows[0]["id"] / "conversation.json"
            conv = json.loads(path.read_text())
            conv.append({"role": "system", "content": "SECRET GOLD EVALUATOR"})
            path.write_text(json.dumps(conv))
            source = {"patch": {"edits": [{"op": "append", "content": "Keep valid retries."}]},
                      "cross_group_cards": [_card(), _card(quote="SECRET GOLD EVALUATOR"),
                                            _card(rollout=rows[1]["id"]), _card()]}
            result = attach_cross_group_cards(source, displayed, tmp, "stable_success", fmt_trajectory)
        self.assertEqual(result["patch"], source["patch"])
        self.assertEqual(len(result["cross_group_cards"]), 1)
        self.assertEqual(result["cross_group_cards"][0]["hard"], 1)
        self.assertEqual(len(source["cross_group_cards"]), 4)
        reasons = {row["reason"] for row in result["cross_group_card_audit"]["excluded_cards"]}
        self.assertEqual(reasons, {"quote_not_in_runtime_trajectory",
                                  "unknown_or_undisplayed_trajectory", "duplicate_trajectory_card"})

    def test_no_edit_successes_still_provide_evidence_in_same_analyst_call(self):
        configure_cross_group_evidence(True)
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for task in ["success-a", "success-b"]:
                _add_task(rows, Path(tmp), task, [1] * 4, ["wait for updated state"] * 4)
            blocks = build_outcome_stratified_task_blocks(rows, tmp)["stable_success"]
            items = [item for block in blocks for item in block]
            output = {"patch": {"edits": []}, "cross_group_cards": [
                _card(task=item["rollout_group_id"], rollout=item["id"]) for item in items]}
            with patch("textualrl.gradient.reflect.chat_optimizer", return_value=("response", {})) as chat, \
                    patch("textualrl.gradient.reflect.extract_json", return_value=output):
                result = run_outcome_stratified_analyst_group(
                    "old skill", items, tmp, route="stable_success", edit_budget=4,
                    system_prompt="original system", meta_skill_context="meta unchanged")
        self.assertEqual(chat.call_count, 1)
        self.assertIn(ANALYST_SUFFIX, chat.call_args.kwargs["system"])
        self.assertEqual(result["patch"]["edits"], [])
        pool = collect_cross_group_evidence([None, result])
        self.assertEqual(pool["card_count"], 2)
        self.assertEqual(pool["distinct_task_count"], 2)
        self.assertEqual(pool["route_counts"], {"stable_success": 2})

    def test_disabled_analyst_prompt_and_payload_stay_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            _add_task(rows, Path(tmp), "mixed", [1, 0, 1, 0], ["good", "bad", "good", "bad"])
            block = build_outcome_stratified_task_blocks(rows, tmp)["mixed"][0]
            output = {"patch": {"edits": [{"op": "append", "content": "Original rule"}]}}
            with patch("textualrl.gradient.reflect.chat_optimizer", return_value=("response", {})) as chat, \
                    patch("textualrl.gradient.reflect.extract_json", return_value=output):
                result = run_outcome_stratified_analyst_group(
                    "skill", block, tmp, route="mixed", edit_budget=4, system_prompt="original")
        self.assertNotIn("Cross-Group", chat.call_args.kwargs["system"])
        self.assertNotIn("cross_group_card_audit", result)
        self.assertNotIn("cross_group_cards", result)
        self.assertEqual(result["patch"]["edits"][0]["content"], "Original rule")
        self.assertEqual(chat.call_count, 1)

    def test_final_merge_uses_success_evidence_without_success_edits(self):
        configure_cross_group_evidence(True)
        original = {"edits": [{"op": "append", "content": "Never repeat.", "support_count": 1}]}
        response = {"edits": [{"op": "append", "content": "Retry only after state changes.",
                     "support_count": 1, "cross_group_review": {
                         "decision": "narrow", "reason": "Successful retries have changed state.",
                         "evidence": [{"task_id": "success-a", "rollout_id": "success-a__sample_01_of_04",
                                       "relation": "limits", "reason": "State changes before the retry."}]}}]}
        with patch("textualrl.gradient.aggregate.chat_optimizer", return_value=("response", {})) as chat, \
                patch("textualrl.gradient.aggregate.extract_json", return_value=response), \
                patch("textualrl.gradient.aggregate.load_prompt", return_value="normal merge"):
            result = merge_patches("old", [original], [], verbose=False, cross_group_evidence=_pool())
        self.assertEqual(chat.call_count, 1)
        self.assertIn("wait for updated state", chat.call_args.kwargs["user"])
        self.assertIn(MERGE_SUFFIX, chat.call_args.kwargs["system"])
        self.assertEqual(result["edits"][0]["support_count"], 1)
        self.assertEqual(result["cross_group_audit"]["edits_with_traceable_review"], 1)
        self.assertEqual(original["edits"][0]["content"], "Never repeat.")

    def test_disabled_merge_ignores_cards_and_preserves_zero_call_shortcut(self):
        original = {"edits": [{"op": "append", "content": "Original"}]}
        with patch("textualrl.gradient.aggregate.chat_optimizer") as chat:
            result = merge_patches("old", [original], [], verbose=False, cross_group_evidence=_pool())
        self.assertEqual(result, original)
        chat.assert_not_called()

    def test_no_cards_or_no_edits_does_not_create_a_review(self):
        configure_cross_group_evidence(True)
        original = {"edits": [{"op": "append", "content": "Original"}]}
        with patch("textualrl.gradient.aggregate.chat_optimizer") as chat:
            result = merge_patches("old", [original], [], verbose=False, cross_group_evidence={"cards": []})
            empty = merge_patches("old", [], [], verbose=False, cross_group_evidence=_pool())
        self.assertEqual(result, original)
        self.assertEqual(empty["edits"], [])
        chat.assert_not_called()

    def test_normal_two_group_merge_does_not_add_an_extra_model_call(self):
        configure_cross_group_evidence(True)
        source = {"edits": [{"op": "append", "content": "Original"}]}
        with patch("textualrl.gradient.aggregate.chat_optimizer", return_value=("response", {})) as chat, \
                patch("textualrl.gradient.aggregate.extract_json", return_value=source):
            result = merge_patches("old", [source], [source], verbose=False, cross_group_evidence=_pool())
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["cross_group_audit"]["status"], "model_reviewed")

    def test_failed_review_preserves_existing_fallback_without_gate(self):
        configure_cross_group_evidence(True)
        source = {"edits": [{"op": "append", "content": "Original"}]}
        with patch("textualrl.gradient.aggregate.chat_optimizer", side_effect=TimeoutError):
            result = merge_patches("old", [source], [], verbose=False, cross_group_evidence=_pool())
        self.assertEqual(result["edits"][0]["content"], "Original")
        self.assertEqual(result["cross_group_audit"]["status"], "merge_fallback_not_reviewed")

    def test_unknown_references_do_not_inflate_credit_or_remove_edit(self):
        source = {"edits": [{"content": "Original", "support_count": 7,
                  "cross_group_review": {"decision": "retain", "evidence": [
                      {"task_id": "unseen-test-task", "rollout_id": "invented",
                       "relation": "supports", "reason": "invented"}]}}]}
        result = annotate_cross_group_review(source, _pool(), status="model_reviewed")
        edit = result["edits"][0]
        self.assertEqual(edit["support_count"], 7)
        self.assertEqual(edit["content"], "Original")
        self.assertEqual(edit["cross_group_review_audit"]["distinct_task_count"], 0)
        self.assertEqual(len(edit["cross_group_review_audit"]["excluded_references"]), 1)

    def test_cards_do_not_cross_steps(self):
        self.assertEqual(collect_cross_group_evidence([])["cards"], [])
        first = {"cross_group_cards": _pool()["cards"], "cross_group_card_audit": {}}
        self.assertEqual(collect_cross_group_evidence([first])["card_count"], 3)
        self.assertEqual(collect_cross_group_evidence([])["cards"], [])

    def test_enabled_cache_is_separate_from_legacy_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred, patches = Path(tmp) / "predictions", Path(tmp) / "patches"
            rows = []
            _add_task(rows, pred, "mixed", [1, 0, 1, 0], ["good", "bad", "good", "bad"])
            def run():
                return run_minibatch_reflect(rows, "skill", str(pred), str(patches),
                    workers=1, failure_only=False, minibatch_size=8, edit_budget=4, random_seed=42)
            fake = {"patch": {"edits": []}, "source_type": "contrastive"}
            with patch("textualrl.gradient.reflect.run_outcome_stratified_analyst_group", return_value=fake) as analyst:
                run()
                run()
                self.assertEqual(analyst.call_count, 1)
                configure_cross_group_evidence(True)
                run()
                run()
                self.assertEqual(analyst.call_count, 2)
            self.assertTrue((patches / "cross_group_evidence" / "route_a_mixed_000.json").exists())


if __name__ == "__main__":
    unittest.main()
