"""Correlational edit-value memory for candidate exploration.

The trainer observes validation reward for a *set* of edits, not a controlled
counterfactual for each edit.  This module therefore maintains conservative
family- and pair-level estimates for proposal ranking only.  It never replaces
the real validation gate and never directly quarantines an edit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from typing import Any

from skillopt.optimizer.quarantine import (
    behavior_similarity,
    edit_behavior_payload,
    partition_runtime_observable_edits,
)


EDIT_VALUE_MEMORY_VERSION = "edit_value_memory_v1"
_ACCEPTED_OUTCOMES = {"accept", "accept_new_best", "force_accept"}
_REJECTED_OUTCOMES = {"reject", "exploration_reject"}


def _family_id(item: dict, update_mode: str) -> str:
    payload = edit_behavior_payload(item, update_mode)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _event_items(event: dict) -> list[dict]:
    return [
        row.get("raw", {})
        for row in event.get("edits", [])
        if isinstance(row, dict) and isinstance(row.get("raw"), dict)
    ]


def _match_family(
    item: dict,
    families: list[dict],
    *,
    update_mode: str,
    similarity_threshold: float,
) -> tuple[dict | None, float]:
    best = None
    best_score = -1.0
    for family in families:
        if family.get("update_mode") != update_mode:
            continue
        variants = family.get("variant_items", []) or [
            family.get("representative_item", {})
        ]
        score = max(
            (
                behavior_similarity(
                    item,
                    variant,
                    update_mode=update_mode,
                )
                for variant in variants
                if isinstance(variant, dict)
            ),
            default=0.0,
        )
        if score >= similarity_threshold and score > best_score:
            best = family
            best_score = score
    return best, max(0.0, best_score)


def _new_family(item: dict, update_mode: str) -> dict:
    return {
        "family_id": _family_id(item, update_mode),
        "update_mode": update_mode,
        "behavior": edit_behavior_payload(item, update_mode),
        "representative_item": item,
        "variant_items": [item],
        "support_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "no_effect_count": 0,
        "allocated_delta_sum": 0.0,
        "candidate_delta_sum": 0.0,
        "beneficial_count": 0,
        "harmful_count": 0,
        "candidate_fingerprints": [],
    }


def build_edit_value_memory(
    edit_history: list[dict],
    *,
    update_mode: str = "patch",
    similarity_threshold: float = 0.82,
    prior_strength: float = 2.0,
) -> dict[str, Any]:
    """Estimate edit-family values from set-level validation outcomes.

    Candidate reward is divided uniformly among the distinct edit families in
    that candidate.  The estimate is intentionally labelled correlational: edit
    interactions and unobserved confounders remain possible.
    """
    families: list[dict] = []
    pair_accumulators: dict[tuple[str, str], dict[str, Any]] = {}
    validation_event_count = 0

    for event in edit_history:
        if not event.get("validation_ran"):
            continue
        event_mode = str(event.get("update_mode", "patch"))
        if event_mode != update_mode:
            continue
        outcome = str(event.get("outcome", ""))
        if outcome not in _ACCEPTED_OUTCOMES | _REJECTED_OUTCOMES:
            continue
        items = _event_items(event)
        if not items:
            continue

        event_families: list[dict] = []
        used_ids: set[str] = set()
        for item in items:
            family, _ = _match_family(
                item,
                families,
                update_mode=update_mode,
                similarity_threshold=similarity_threshold,
            )
            if family is None:
                family = _new_family(item, update_mode)
                families.append(family)
            elif len(family["variant_items"]) < 8 and all(
                edit_behavior_payload(item, update_mode)
                != edit_behavior_payload(variant, update_mode)
                for variant in family["variant_items"]
            ):
                family["variant_items"].append(item)
            if family["family_id"] not in used_ids:
                event_families.append(family)
                used_ids.add(family["family_id"])

        if not event_families:
            continue
        validation_event_count += 1
        metrics = event.get("metrics") or {}
        delta = metrics.get("hard_delta")
        if delta is None:
            delta = metrics.get("soft_delta")
        delta = float(delta or 0.0)
        allocated_delta = delta / len(event_families)
        paired = event.get("paired_summary") or {}
        beneficial = int(paired.get("beneficial_count", 0) or 0)
        harmful = int(paired.get("harmful_count", 0) or 0)
        fingerprint = str(event.get("candidate_fingerprint", ""))

        for family in event_families:
            family["support_count"] += 1
            family["accepted_count"] += int(outcome in _ACCEPTED_OUTCOMES)
            family["rejected_count"] += int(outcome in _REJECTED_OUTCOMES)
            family["no_effect_count"] += int(bool(event.get("no_effect")))
            family["allocated_delta_sum"] += allocated_delta
            family["candidate_delta_sum"] += delta
            family["beneficial_count"] += beneficial
            family["harmful_count"] += harmful
            if fingerprint and fingerprint not in family["candidate_fingerprints"]:
                family["candidate_fingerprints"].append(fingerprint)

        for left, right in itertools.combinations(
            sorted(family["family_id"] for family in event_families),
            2,
        ):
            pair = pair_accumulators.setdefault(
                (left, right),
                {
                    "family_ids": [left, right],
                    "support_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "candidate_delta_sum": 0.0,
                },
            )
            pair["support_count"] += 1
            pair["accepted_count"] += int(outcome in _ACCEPTED_OUTCOMES)
            pair["rejected_count"] += int(outcome in _REJECTED_OUTCOMES)
            pair["candidate_delta_sum"] += delta

    prior_strength = max(0.0, float(prior_strength))
    for family in families:
        support = int(family["support_count"])
        family["posterior_mean_delta"] = round(
            family["allocated_delta_sum"] / (support + prior_strength),
            8,
        )
        family["mean_candidate_delta"] = round(
            family["candidate_delta_sum"] / support if support else 0.0,
            8,
        )
        family["rejection_rate_smoothed"] = round(
            (family["rejected_count"] + 1.0) / (support + 2.0),
            8,
        )
        family["uncertainty"] = round(
            1.0 / math.sqrt(support + prior_strength + 1.0),
            8,
        )

    pairs = []
    for pair in pair_accumulators.values():
        support = int(pair["support_count"])
        pair["posterior_mean_delta"] = round(
            pair["candidate_delta_sum"] / (support + prior_strength),
            8,
        )
        pair["rejection_rate_smoothed"] = round(
            (pair["rejected_count"] + 1.0) / (support + 2.0),
            8,
        )
        pairs.append(pair)

    return {
        "version": EDIT_VALUE_MEMORY_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "credit_method": "uniform_candidate_delta_correlational",
        "causal_warning": (
            "Values are set-level correlations, not single-edit counterfactual effects."
        ),
        "update_mode": update_mode,
        "similarity_threshold": similarity_threshold,
        "prior_strength": prior_strength,
        "validation_event_count": validation_event_count,
        "family_count": len(families),
        "pair_count": len(pairs),
        "families": families,
        "pairs": pairs,
    }


def score_edit_candidate(
    items: list[dict],
    memory: dict[str, Any],
    *,
    update_mode: str = "patch",
    exploration_weight: float = 0.01,
    novelty_weight: float = 0.005,
    risk_weight: float = 0.005,
    pair_weight: float = 0.25,
    support_weight: float = 0.001,
) -> dict[str, Any]:
    """Return an auditable UCB-style proposal score without claiming causality."""
    families = list(memory.get("families", []))
    threshold = float(memory.get("similarity_threshold", 0.82))
    matches = []
    family_ids: list[str] = []
    scored_family_ids: set[str] = set()
    new_count = 0
    support_total = 0

    for item in items:
        family, similarity = _match_family(
            item,
            families,
            update_mode=update_mode,
            similarity_threshold=threshold,
        )
        support_total += int(item.get("support_count", 0) or 0)
        if family is None:
            new_count += 1
            matches.append(
                {
                    "family_id": None,
                    "similarity": 0.0,
                    "posterior_mean_delta": 0.0,
                    "rejection_rate_smoothed": 0.5,
                    "uncertainty": 1.0,
                    "behavior": edit_behavior_payload(item, update_mode),
                }
            )
            continue
        family_ids.append(str(family["family_id"]))
        duplicate_family = str(family["family_id"]) in scored_family_ids
        scored_family_ids.add(str(family["family_id"]))
        matches.append(
            {
                "family_id": family["family_id"],
                "similarity": round(similarity, 6),
                "posterior_mean_delta": family["posterior_mean_delta"],
                "rejection_rate_smoothed": family["rejection_rate_smoothed"],
                "uncertainty": family["uncertainty"],
                "duplicate_family_in_candidate": duplicate_family,
                "behavior": family.get("behavior", {}),
            }
        )

    count = max(len(items), 1)
    unique_matches = [
        row for row in matches
        if not row.get("duplicate_family_in_candidate", False)
    ]
    value = sum(float(row["posterior_mean_delta"]) for row in unique_matches)
    uncertainty = (
        sum(float(row["uncertainty"]) for row in unique_matches) / count
    )
    novelty = new_count / count
    risk = sum(
        max(0.0, float(row["rejection_rate_smoothed"]) - 0.5)
        for row in unique_matches
    ) / count

    pair_lookup = {
        tuple(sorted(pair.get("family_ids", []))): pair
        for pair in memory.get("pairs", [])
        if len(pair.get("family_ids", [])) == 2
    }
    pair_values = [
        float(pair_lookup[pair]["posterior_mean_delta"])
        for pair in itertools.combinations(sorted(set(family_ids)), 2)
        if pair in pair_lookup
    ]
    pair_value = sum(pair_values) / len(pair_values) if pair_values else 0.0
    support_bonus = math.log1p(max(0, support_total))
    final_score = (
        value
        + exploration_weight * uncertainty
        + novelty_weight * novelty
        - risk_weight * risk
        + pair_weight * pair_value
        + support_weight * support_bonus
    )
    return {
        "score": round(final_score, 8),
        "estimated_value": round(value, 8),
        "uncertainty_bonus": round(exploration_weight * uncertainty, 8),
        "novelty_bonus": round(novelty_weight * novelty, 8),
        "risk_penalty": round(risk_weight * risk, 8),
        "pair_value_bonus": round(pair_weight * pair_value, 8),
        "support_bonus": round(support_weight * support_bonus, 8),
        "new_edit_fraction": round(novelty, 6),
        "matches": matches,
    }


def format_edit_value_context(memory: dict[str, Any], limit: int = 6) -> str:
    """Render compact value evidence for the optimizer proposal context."""
    supported = []
    for family in memory.get("families", []):
        if int(family.get("support_count", 0)) <= 0:
            continue
        representative = family.get("representative_item", {})
        observable, hits = partition_runtime_observable_edits(
            [representative],
            update_mode=str(family.get("update_mode", "patch")),
        )
        if hits or not observable:
            continue
        supported.append(family)
    if not supported:
        return ""
    positive = sorted(
        [
            row for row in supported
            if float(row.get("posterior_mean_delta", 0.0)) > 0.0
        ],
        key=lambda row: float(row.get("posterior_mean_delta", 0.0)),
        reverse=True,
    )[:limit]
    risky = sorted(
        [
            row for row in supported
            if float(row.get("posterior_mean_delta", 0.0)) < 0.0
            or float(row.get("rejection_rate_smoothed", 0.5)) > 0.5
        ],
        key=lambda row: (
            float(row.get("posterior_mean_delta", 0.0)),
            -float(row.get("rejection_rate_smoothed", 0.5)),
        ),
    )[:limit]

    def _compact_text(value: Any, max_chars: int = 280) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text
        prefix = text[: max_chars + 1].rsplit(" ", 1)[0]
        return (prefix or text[:max_chars]).rstrip() + "..."

    def _render(rows: list[dict]) -> str:
        rendered = "\n".join(
            "- "
            + _compact_text(row.get("behavior", {}).get("text", ""))
            + " (estimated_delta="
            + f"{float(row.get('posterior_mean_delta', 0.0)):+.4f}, "
            + f"support={row.get('support_count', 0)}, "
            + f"reject_rate={float(row.get('rejection_rate_smoothed', 0.5)):.2f})"
            for row in rows
            if row.get("behavior", {}).get("text")
        )
        return rendered or "- (none yet)"

    return (
        "## Correlational edit-value memory\n"
        "These estimates come from multi-edit candidates. Use them for proposal "
        "prioritization only; they are not single-edit causal effects. Prefer "
        "well-supported positive families while still exploring genuinely new "
        "causal repairs.\n\n"
        "Higher-value families:\n"
        f"{_render(positive)}\n\n"
        "Riskier families:\n"
        f"{_render(risky)}"
    )


def save_edit_value_memory(path: str, memory: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(memory, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
