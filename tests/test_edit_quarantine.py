from __future__ import annotations

from pathlib import Path
import tempfile

from textualrl.optimizer.quarantine import (
    add_quarantined_candidate,
    append_edit_history_event,
    behavior_similarity,
    build_edit_history_event,
    candidates_equivalent,
    edit_similarity,
    find_permanently_quarantined_edits,
    find_quarantined_candidate,
    find_unobservable_runtime_edits,
    fingerprint_edit,
    format_candidate_resample_context,
    format_permanent_edit_context,
    format_runtime_observability_context,
    load_edit_history,
    promote_repeated_edits,
)


def _edit(content: str, *, support_count: int | None = None) -> dict:
    item = {"op": "append", "content": content}
    if support_count is not None:
        item["support_count"] = support_count
    return item


def _patch_edit(op: str, content: str, target: str = "") -> dict:
    item = {"op": op, "content": content}
    if target:
        item["target"] = target
    return item


A1 = _edit("Verify retrieved evidence before selecting the final answer.")
A2 = _edit("Verify the retrieved evidence before selecting the final answer.")
A3 = _edit("Verify retrieved evidence before selecting a final answer.")
B = _edit("Prefer an answer span stated verbatim in the context.")
C = _edit("Resolve aliases before copying an answer span.")
D = _edit("Check whether the question asks for a person or a place.")
E = _edit("Use surrounding sentences to resolve ambiguous references.")


def test_support_count_does_not_change_edit_identity() -> None:
    left = _edit("Verify evidence before answering.", support_count=2)
    right = _edit("Verify evidence before answering.", support_count=7)

    assert fingerprint_edit(left) == fingerprint_edit(right)
    assert edit_similarity(left, right) == 1.0


def test_minor_token_variants_share_an_edit_family() -> None:
    assert edit_similarity(A1, A2) >= 0.88
    assert edit_similarity(A1, A3) >= 0.88


def test_candidate_identity_remains_sensitive_to_operation_and_target() -> None:
    append = _patch_edit("append", "Verify evidence before answering.")
    replace = _patch_edit(
        "replace",
        "Verify evidence before answering.",
        target="Old answer selection rule.",
    )

    assert edit_similarity(append, replace) == 0.0
    assert not candidates_equivalent([append], [replace])
    assert behavior_similarity(append, replace) == 1.0


def test_behavior_family_ignores_patch_operation_and_target() -> None:
    append = _patch_edit(
        "append",
        "Verify retrieved evidence before selecting the final answer.",
    )
    insert_after = _patch_edit(
        "insert_after",
        "Verify the retrieved evidence before selecting a final answer.",
        target="## Retrieval",
    )
    replace = _patch_edit(
        "replace",
        "Verify retrieved evidence before selecting the final answer.",
        target="Choose the first plausible answer.",
    )

    assert behavior_similarity(append, insert_after) >= 0.72
    assert behavior_similarity(append, replace) == 1.0


def test_near_equivalent_candidate_is_blocked_order_independently() -> None:
    rejected = [A1, B, C]
    near_duplicate = [C, A2, B]
    records: list[dict] = []
    added = add_quarantined_candidate(records, rejected, step=1)

    assert added is not None
    assert candidates_equivalent(rejected, near_duplicate)
    assert find_quarantined_candidate(near_duplicate, records) == added


def test_different_edit_combination_remains_available() -> None:
    records: list[dict] = []
    add_quarantined_candidate(records, [A1, B, C, D])

    assert find_quarantined_candidate([A2, C, D, E], records) is None


def test_edit_family_is_blocked_after_three_distinct_rejected_candidates() -> None:
    records: list[dict] = []
    add_quarantined_candidate(records, [A1, B], step=1)
    add_quarantined_candidate(records, [A2, C], step=2)
    add_quarantined_candidate(records, [A3, D], step=3)

    added = promote_repeated_edits(records, threshold=3)

    assert len(added) == 1
    assert added[0]["rejected_candidate_count"] == 3
    assert added[0]["accepted_candidate_count"] == 0
    blocked = find_permanently_quarantined_edits([A2, E], records)
    assert len(blocked) == 1
    assert blocked[0]["similarity"] >= 0.88


def test_behavior_family_promotes_across_patch_operations() -> None:
    shared_texts = [
        "Verify retrieved evidence before selecting the final answer.",
        "Verify the retrieved evidence before selecting a final answer.",
        "Verify retrieved evidence before selecting the final answer.",
    ]
    shared_edits = [
        _patch_edit("append", shared_texts[0]),
        _patch_edit("insert_after", shared_texts[1], target="## Retrieval"),
        _patch_edit("replace", shared_texts[2], target="Old selection rule"),
    ]
    records: list[dict] = []
    add_quarantined_candidate(records, [shared_edits[0], B], step=1)
    add_quarantined_candidate(records, [shared_edits[1], C], step=2)
    add_quarantined_candidate(records, [shared_edits[2], D], step=3)

    added = promote_repeated_edits(
        records,
        threshold=3,
        behavior_similarity_threshold=0.72,
    )

    assert len(added) == 1
    assert added[0]["behavior"]["text"].startswith("verify retrieved evidence")
    probe = _patch_edit(
        "insert_after",
        "Verify retrieved evidence before selecting the final answer.",
        target="A completely different section",
    )
    blocked = find_permanently_quarantined_edits(
        [probe],
        records,
        behavior_similarity_threshold=0.72,
    )
    assert len(blocked) == 1
    assert blocked[0]["behavior"]["kind"] == "instruction"


def test_accepted_candidate_prevents_permanent_quarantine() -> None:
    records: list[dict] = []
    add_quarantined_candidate(records, [A1, B], step=1)
    add_quarantined_candidate(records, [A2, C], step=2)
    add_quarantined_candidate(records, [A3, D], step=3)
    accepted = build_edit_history_event(
        [A2, E],
        validation_ran=True,
        outcome="accept",
    )

    assert promote_repeated_edits(records, [accepted], threshold=3) == []


def test_duplicate_candidate_does_not_add_an_edit_strike() -> None:
    records: list[dict] = []
    assert add_quarantined_candidate(records, [A1, B], step=1) is not None
    assert add_quarantined_candidate(records, [B, A2], step=2) is None

    assert promote_repeated_edits(records, threshold=2) == []


def test_edit_history_records_blocked_and_no_effect_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "edit_history.jsonl")
        blocked = build_edit_history_event(
            [A1, B],
            step=1,
            validation_ran=False,
            outcome="blocked_pre_validation",
        )
        neutral = build_edit_history_event(
            [A2, C],
            step=2,
            validation_ran=True,
            outcome="reject",
            no_effect=True,
        )
        append_edit_history_event(path, blocked)
        append_edit_history_event(path, neutral)

        loaded = load_edit_history(path)
        assert len(loaded) == 2
        assert loaded[0]["outcome"] == "blocked_pre_validation"
        assert loaded[1]["no_effect"] is True
        assert loaded[1]["edits"][0]["semantic"]["content"].startswith(
            "verify the retrieved evidence"
        )
        assert loaded[1]["edits"][0]["behavior"]["text"].startswith(
            "verify the retrieved evidence"
        )


def test_runtime_observability_rejects_evaluator_only_signals() -> None:
    items = [
        _edit("Compare the response character by character with the gold answer."),
        _edit("Use the ground-truth label to resolve ambiguity."),
        _edit("Revise the response when the evaluator score is low."),
    ]

    hits = find_unobservable_runtime_edits(items)

    assert len(hits) == 3
    reasons = {reason for hit in hits for reason in hit["matched_reasons"]}
    assert {"gold_target", "ground_truth", "evaluator_feedback"} <= reasons


def test_runtime_observability_allows_source_grounded_exactness() -> None:
    items = [
        _edit("Select the exact answer span from the provided context."),
        _edit("Preserve the exact spelling of entities visible in the source."),
        _edit("Check the proposed answer against the retrieved passage."),
    ]

    assert find_unobservable_runtime_edits(items) == []
    context = format_runtime_observability_context()
    assert "inference time" in context
    assert "source-grounded procedures" in context


def test_quarantine_prompts_show_candidate_and_permanent_families() -> None:
    records: list[dict] = []
    add_quarantined_candidate(records, [A1, B], step=1)
    add_quarantined_candidate(records, [A2, C], step=2)
    add_quarantined_candidate(records, [A3, D], step=3)
    promote_repeated_edits(records, threshold=3)

    candidate_context = format_candidate_resample_context(records)
    permanent_context = format_permanent_edit_context(records)
    assert "Rejected candidate family" in candidate_context
    assert "Individual edits may be reused" in candidate_context
    assert "Permanent bad-edit quarantine" in permanent_context
    assert "rejected=3" in permanent_context
