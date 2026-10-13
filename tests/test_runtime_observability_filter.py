from __future__ import annotations

import unittest

from textualrl.optimizer.quarantine import (
    filter_observable_runtime_edits,
    filter_observable_runtime_patches,
    format_runtime_observability_context,
    partition_runtime_observable_edits,
)

class RuntimeObservabilityFilterTest(unittest.TestCase):
    def test_keeps_observable_subset(self) -> None:
        good = {
            "op": "append",
            "content": "Copy the answer span supported by the supplied context.",
        }
        bad = {
            "op": "append",
            "content": "Normalize the response until it matches the gold answer.",
        }

        kept, hits = filter_observable_runtime_edits([good, bad])

        self.assertEqual(kept, [good])
        self.assertEqual(len(hits), 1)
        self.assertIn("gold_target", hits[0]["matched_reasons"])

    def test_allows_deleting_a_non_deployable_rule(self) -> None:
        deletion = {
            "op": "delete",
            "target": "Compare the response with the gold answer.",
        }

        kept, hits = filter_observable_runtime_edits([deletion])

        self.assertEqual(kept, [deletion])
        self.assertEqual(hits, [])

    def test_allows_runtime_observable_normalization_language(self) -> None:
        edits = [
            {
                "op": "append",
                "content": "Return the canonical form supported by the context.",
            },
            {
                "op": "append",
                "content": "Preserve the official name and proper casing from the source.",
            },
            {
                "op": "append",
                "content": "Keep the core identifier that distinguishes the entity.",
            },
        ]

        kept, hits = filter_observable_runtime_edits(edits)

        self.assertEqual(kept, edits)
        self.assertEqual(hits, [])

    def test_blocks_only_explicit_hidden_training_dependencies(self) -> None:
        edits = [
            {"op": "append", "content": "Match the gold standard."},
            {
                "op": "append",
                "content": "Prefer the form expected by the evaluator.",
            },
            {
                "op": "append",
                "content": "Select the answer with the highest validation score.",
            },
        ]

        kept, hits = filter_observable_runtime_edits(edits)

        self.assertEqual(kept, [])
        self.assertEqual(len(hits), 3)
        reasons = {reason for hit in hits for reason in hit["matched_reasons"]}
        self.assertIn("gold_standard", reasons)
        self.assertIn("evaluation_expectation", reasons)
        self.assertIn("evaluation_signal", reasons)

    def test_allows_rules_that_prohibit_hidden_signal_use(self) -> None:
        edits = [
            {
                "op": "append",
                "content": "Never use the gold answer; rely only on supplied evidence.",
            },
            {
                "op": "append",
                "content": "Do not consult evaluator feedback when producing an answer.",
            },
        ]

        kept, hits = filter_observable_runtime_edits(edits)

        self.assertEqual(kept, edits)
        self.assertEqual(hits, [])

    def test_partition_preserves_valid_siblings_in_original_order(self) -> None:
        first = {
            "op": "append",
            "content": "Resolve aliases using the supplied passage.",
        }
        hidden = {
            "op": "append",
            "content": "Revise the answer after reading grader feedback.",
        }
        second = {
            "op": "append",
            "content": "Return only the shortest supported answer span.",
        }

        kept, hits = partition_runtime_observable_edits(
            [first, hidden, second]
        )

        self.assertEqual(kept, [first, second])
        self.assertEqual([hit["item"] for hit in hits], [hidden])

    def test_source_patch_filter_removes_training_reasoning_and_bad_edit(self) -> None:
        patches = [
            {
                "reasoning": "The prediction did not match the gold answer.",
                "edits": [
                    {
                        "op": "append",
                        "content": "Use the supplied context to resolve aliases.",
                    },
                    {
                        "op": "append",
                        "content": "Compare the result with the gold answer.",
                    },
                ],
            }
        ]

        filtered, hits = filter_observable_runtime_patches(patches)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0]["edits"]), 1)
        self.assertEqual(len(hits), 1)
        self.assertNotIn("gold", filtered[0]["reasoning"].lower())

    def test_generation_pipeline_enforces_deployability(self) -> None:
        gate_context = format_runtime_observability_context().lower()
        self.assertIn("inference time", gate_context)
        self.assertIn("gold or reference answers", gate_context)
        self.assertIn("evaluator or grader feedback", gate_context)


if __name__ == "__main__":
    unittest.main()
