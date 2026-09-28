"""Offline-style edit value accounting for validated optimizer candidates.

The audit deliberately exposes no prompt formatting or candidate-ranking API.
Its output is correlational: edits in one candidate are evaluated together, so
the candidate delta is shared equally across its distinct semantic families.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from typing import Any

from skillopt.optimizer.quarantine import (
    behavior_similarity,
    edit_behavior_payload,
    edit_semantic_payload,
)


AUDIT_VERSION = "edit_value_audit_v1"
ACCEPTED_OUTCOMES = {"accept", "accept_new_best", "force_accept"}


def _event_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row["raw"]
        for row in event.get("edits", [])
        if isinstance(row, dict) and isinstance(row.get("raw"), dict)
    ]


def _family_id(item: dict[str, Any], update_mode: str) -> str:
    payload = edit_behavior_payload(item, update_mode)
    if not payload:
        payload = edit_semantic_payload(item, update_mode)
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"edit_family::{digest}"


def _find_family(
    families: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    update_mode: str,
    similarity_threshold: float,
) -> dict[str, Any] | None:
    best_family = None
    best_score = -1.0
    for family in families:
        if family["update_mode"] != update_mode:
            continue
        score = max(
            behavior_similarity(item, variant, update_mode=update_mode)
            for variant in family["variant_items"]
        )
        if score >= similarity_threshold and score > best_score:
            best_family = family
            best_score = score
    return best_family


def _new_family(item: dict[str, Any], update_mode: str) -> dict[str, Any]:
    return {
        "family_id": _family_id(item, update_mode),
        "update_mode": update_mode,
        "representative": item,
        "semantic": edit_semantic_payload(item, update_mode),
        "behavior": edit_behavior_payload(item, update_mode),
        "variant_items": [item],
        "support_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "no_effect_count": 0,
        "candidate_delta_sum": 0.0,
        "allocated_delta_sum": 0.0,
        "allocated_delta_sq_sum": 0.0,
        "paired_beneficial_count": 0,
        "paired_harmful_count": 0,
        "candidate_fingerprints": [],
    }


def _candidate_delta(event: dict[str, Any]) -> tuple[float, str]:
    metrics = event.get("metrics") or {}
    hard_delta = metrics.get("hard_delta")
    if hard_delta is not None:
        return float(hard_delta), "hard"
    soft_delta = metrics.get("soft_delta")
    if soft_delta is not None:
        return float(soft_delta), "soft"
    return 0.0, "unavailable"


def build_edit_value_audit(
    edit_history: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    """Aggregate validated candidate outcomes without influencing training."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1]")

    families: list[dict[str, Any]] = []
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    validation_event_count = 0
    delta_source_counts: dict[str, int] = {}

    for event in edit_history:
        if not isinstance(event, dict) or not event.get("validation_ran"):
            continue
        items = _event_items(event)
        if not items:
            continue
        validation_event_count += 1
        update_mode = str(event.get("update_mode") or "patch")
        candidate_fingerprint = str(event.get("candidate_fingerprint") or "")
        delta, delta_source = _candidate_delta(event)
        delta_source_counts[delta_source] = delta_source_counts.get(delta_source, 0) + 1
        paired = event.get("paired_summary") or {}

        event_families: list[dict[str, Any]] = []
        for item in items:
            family = _find_family(
                families,
                item,
                update_mode=update_mode,
                similarity_threshold=similarity_threshold,
            )
            if family is None:
                family = _new_family(item, update_mode)
                families.append(family)
            elif item not in family["variant_items"]:
                family["variant_items"].append(item)
            if family not in event_families:
                event_families.append(family)

        allocated_delta = delta / len(event_families)
        accepted = str(event.get("outcome") or "") in ACCEPTED_OUTCOMES
        for family in event_families:
            family["support_count"] += 1
            family["accepted_count" if accepted else "rejected_count"] += 1
            family["no_effect_count"] += int(bool(event.get("no_effect")))
            family["candidate_delta_sum"] += delta
            family["allocated_delta_sum"] += allocated_delta
            family["allocated_delta_sq_sum"] += allocated_delta * allocated_delta
            family["paired_beneficial_count"] += int(
                paired.get("beneficial_count", 0) or 0
            )
            family["paired_harmful_count"] += int(
                paired.get("harmful_count", 0) or 0
            )
            if candidate_fingerprint:
                family["candidate_fingerprints"].append(candidate_fingerprint)

        family_ids = sorted(family["family_id"] for family in event_families)
        for left_index, left_id in enumerate(family_ids):
            for right_id in family_ids[left_index + 1 :]:
                key = (left_id, right_id)
                pair = pair_stats.setdefault(
                    key,
                    {
                        "family_ids": [left_id, right_id],
                        "support_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "candidate_delta_sum": 0.0,
                    },
                )
                pair["support_count"] += 1
                pair["accepted_count" if accepted else "rejected_count"] += 1
                pair["candidate_delta_sum"] += delta

    serializable_families = []
    for family in families:
        support = family["support_count"]
        mean = family["allocated_delta_sum"] / support if support else 0.0
        second_moment = (
            family["allocated_delta_sq_sum"] / support if support else 0.0
        )
        variance = max(0.0, second_moment - mean * mean)
        row = {key: value for key, value in family.items() if key != "variant_items"}
        row["variant_count"] = len(family["variant_items"])
        row["mean_allocated_delta"] = mean
        row["standard_error"] = (
            math.sqrt(variance / support) if support > 1 else None
        )
        row["accept_rate"] = family["accepted_count"] / support if support else 0.0
        row["candidate_fingerprints"] = sorted(
            set(family["candidate_fingerprints"])
        )
        serializable_families.append(row)

    serializable_families.sort(
        key=lambda row: (-row["support_count"], row["family_id"])
    )
    pairs = sorted(pair_stats.values(), key=lambda row: row["family_ids"])
    for pair in pairs:
        support = pair["support_count"]
        pair["mean_candidate_delta"] = (
            pair["candidate_delta_sum"] / support if support else 0.0
        )

    return {
        "version": AUDIT_VERSION,
        "mode": "audit_only",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "influences_training": False,
        "similarity_threshold": similarity_threshold,
        "validation_event_count": validation_event_count,
        "family_count": len(serializable_families),
        "pair_count": len(pairs),
        "delta_source_counts": delta_source_counts,
        "families": serializable_families,
        "pairs": pairs,
        "causal_warning": (
            "Candidate edits are validated jointly. Equal-share deltas and "
            "family statistics are correlational audit signals, not causal "
            "credit assignments."
        ),
    }


def save_edit_value_audit(path: str, audit: dict[str, Any]) -> None:
    """Atomically persist an audit snapshot."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)
