"""Per-example promotion audit for validation-gated skill updates."""
from __future__ import annotations

from typing import Any


def _index_results(results: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in results:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            raise ValueError(f"{label} result is missing an id")
        if item_id in indexed:
            raise ValueError(f"duplicate {label} result id: {item_id}")
        indexed[item_id] = row
    return indexed


def build_paired_promotion_audit(
    current_results: list[dict],
    candidate_results: list[dict],
    *,
    require_hard_gain: bool = True,
) -> dict[str, Any]:
    """Compare current and candidate outcomes on the same validation items.

    A candidate is promotion-safe only when every current hard success remains
    successful and, by default, at least one current failure becomes a hard
    success. Missing candidate rows are treated as unsafe.
    """
    current = _index_results(current_results, label="current")
    candidate = _index_results(candidate_results, label="candidate")

    current_ids = set(current)
    candidate_ids = set(candidate)
    matched_ids = sorted(current_ids & candidate_ids)
    missing_candidate_ids = sorted(current_ids - candidate_ids)
    extra_candidate_ids = sorted(candidate_ids - current_ids)

    beneficial: list[dict] = []
    harmful: list[dict] = []
    stable_success: list[str] = []
    stable_failure: list[str] = []
    soft_delta_sum = 0.0

    for item_id in matched_ids:
        before = current[item_id]
        after = candidate[item_id]
        before_hard = int(float(before.get("hard", 0)) > 0.0)
        after_hard = int(float(after.get("hard", 0)) > 0.0)
        before_soft = float(before.get("soft", 0.0) or 0.0)
        after_soft = float(after.get("soft", 0.0) or 0.0)
        soft_delta_sum += after_soft - before_soft

        detail = {
            "id": item_id,
            "current_hard": before_hard,
            "candidate_hard": after_hard,
            "current_soft": before_soft,
            "candidate_soft": after_soft,
            "current_answer": before.get("predicted_answer", ""),
            "candidate_answer": after.get("predicted_answer", ""),
            "gold_answers": after.get("gold_answers", before.get("gold_answers", [])),
        }
        if before_hard == 0 and after_hard == 1:
            beneficial.append(detail)
        elif before_hard == 1 and after_hard == 0:
            harmful.append(detail)
        elif before_hard == 1:
            stable_success.append(item_id)
        else:
            stable_failure.append(item_id)

    has_full_coverage = not missing_candidate_ids and len(matched_ids) == len(current)
    has_required_gain = bool(beneficial) or not require_hard_gain
    accepted = has_full_coverage and not harmful and has_required_gain

    rejection_reasons: list[str] = []
    if missing_candidate_ids:
        rejection_reasons.append("missing_candidate_results")
    if harmful:
        rejection_reasons.append("hard_regression")
    if require_hard_gain and not beneficial:
        rejection_reasons.append("no_hard_gain")

    return {
        "accepted": accepted,
        "require_hard_gain": require_hard_gain,
        "matched_count": len(matched_ids),
        "current_count": len(current),
        "candidate_count": len(candidate),
        "beneficial_count": len(beneficial),
        "harmful_count": len(harmful),
        "stable_success_count": len(stable_success),
        "stable_failure_count": len(stable_failure),
        "mean_soft_delta": soft_delta_sum / max(len(matched_ids), 1),
        "missing_candidate_ids": missing_candidate_ids,
        "extra_candidate_ids": extra_candidate_ids,
        "beneficial": beneficial,
        "harmful": harmful,
        "rejection_reasons": rejection_reasons,
    }
