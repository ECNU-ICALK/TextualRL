"""Optional cross-outcome evidence for V2.3 proposal refinement.

The optimizer judges relevance and scope. Citation checks provide provenance,
not causal credit, a new acceptance rule, or extra target rollouts.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path


_enabled = False
_ROUTES = {"mixed", "stable_success", "stable_failure"}
_CARD_FIELDS = ("condition", "behavior", "observed_effect")

ANALYST_SUFFIX = """

## Cross-Group Behavioral Evidence (Training-Side Metadata Only)

Keep the current route's proposal rules and edit budget. In the same JSON
response, alongside patch, also return cross_group_cards, a list of objects:
task_id, rollout_id, quote, condition, behavior, observed_effect.
For each displayed unique trajectory, report at most ONE salient behavioral
observation, even when no edit is warranted (return patch with an empty edits
list in that case). quote must be a short verbatim runtime action/observation
span, up to 800 characters. Describe observable preconditions, the action, and
its visible effect in at most 300 characters each. Use "unknown" when the trace
does not establish a condition or effect. Do not invent a successful repair.
Choose behavior relevant to a proposed repair or worth preserving. For mixed
groups expose both sides of the divergence; for homogeneous groups describe
the actual observed behavior, not a presumed cause of the final score.
Only cite Task Block IDs and displayed trajectory IDs in this call. Do not cite
gold answers, evaluator feedback, final grading labels, hidden siblings, or
previous-step summaries as runtime evidence. Cards are evidence for later
semantic comparison, not extra edits or independent votes. Do not put any
card metadata or task IDs into executable skill content.
"""

MERGE_SUFFIX = """

## Cross-Group Scope Refinement

Alongside the candidate edits, you receive behavioral evidence from the current
training step's mixed, all-success, and all-failure task groups. Associate each
candidate ONLY with semantically relevant actions and observable preconditions.
Benchmark names, task-ID similarity, and matching outcome labels are not evidence
of relevance. Do not force all three group types to support every candidate.

Mixed evidence can suggest an improvement direction. Independent failure tasks
can corroborate an observed failure mechanism, not prove a repair. Successful
behavior can support preservation or LIMIT a repair's scope: do not prohibit a
behavior merely because it also appears in a failed task. Distinguish a genuine
contradiction under comparable conditions from different valid preconditions.
Use an observable condition in executable content when narrowing a rule; keep
supported exceptions and unrelated successful behavior intact. Only revise an
existing proposed behavior; do not invent unrelated edits to use every card.
If no relevant cross-group evidence exists, keep the original evidence scope
and uncertainty rather than inventing support or discarding the edit solely for
missing evidence. All-success and all-failure proposals remain eligible.

Maintain the normal output patch format and edit budget. For EACH output edit,
add training-only cross_group_review with decision (retain, narrow, revise, or
insufficient_evidence), reason, and evidence (a list of task_id, rollout_id,
relation [supports, limits, contradicts], reason). Cite only cards actually
used. Correlated siblings are not independent task votes. These records do not
automatically change support_count, ranking, acceptance, or causal confidence.
Keep metadata out of skill content. Existing protected sections and runtime
observability restrictions still apply.
"""


def configure_cross_group_evidence(enabled: bool = False) -> None:
    global _enabled
    if not isinstance(enabled, bool):
        raise TypeError("use_cross_group_evidence must be a boolean")
    _enabled = enabled


def cross_group_evidence_enabled() -> bool:
    return _enabled


def validate_cross_group_config(cfg: dict) -> None:
    enabled = cfg.get("use_cross_group_evidence", False)
    if not isinstance(enabled, bool):
        raise TypeError("use_cross_group_evidence must be a boolean")
    if not enabled:
        return
    if not cfg.get("group_relative_outcome_stratified_reflection", False):
        raise ValueError("use_cross_group_evidence requires V2.3 outcome-stratified reflection")
    if cfg.get("use_group_relative_edit_credit", False) or cfg.get(
        "group_relative_candidate_competition", False
    ):
        raise ValueError("use_cross_group_evidence supports the standalone V2.3 path")
    if cfg.get("skill_update_mode", "patch") not in {"patch", "edits"}:
        raise ValueError("use_cross_group_evidence currently requires patch update mode")


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalized(value: str) -> str:
    return " ".join(value.split())


def attach_cross_group_cards(result: dict, items: list[dict], prediction_dir: str,
                             route: str, formatter) -> dict:
    """Check model-selected quotes against displayed runtime trajectories."""
    result = deepcopy(result)
    sources = {}
    source_errors = []
    for item in items:
        task = str(item.get("rollout_group_id") or item.get("id") or "")
        rollout = str(item.get("id") or "")
        path = Path(prediction_dir) / rollout / "conversation.json"
        try:
            conversation = json.loads(path.read_text())
            if not isinstance(conversation, list):
                raise ValueError("conversation is not a list")
            # System records in these exports contain post-execution grading.
            runtime = [row for row in conversation
                       if isinstance(row, dict) and row.get("role") != "system"
                       and row.get("type") not in {"verification", "evaluation"}]
            sources[task, rollout] = (_normalized(formatter(runtime)), item)
        except (OSError, ValueError) as exc:
            source_errors.append({"task_id": task, "rollout_id": rollout,
                                  "error": str(exc)})

    raw = result.get("cross_group_cards", [])
    rejected, cards, seen = [], [], set()
    if not isinstance(raw, list):
        rejected.append({"reason": "malformed_card_list", "claim": raw})
        raw = []
    for card in raw:
        reason = ""
        if not isinstance(card, dict):
            rejected.append({"reason": "malformed_card", "claim": card})
            continue
        task, rollout = _text(card.get("task_id")), _text(card.get("rollout_id"))
        quote = _text(card.get("quote"))
        key = task, rollout
        if key not in sources:
            reason = "unknown_or_undisplayed_trajectory"
        elif not quote or len(quote) > 800:
            reason = "missing_or_overlong_quote"
        elif _normalized(quote) not in sources[key][0]:
            reason = "quote_not_in_runtime_trajectory"
        elif any(not _text(card.get(field)) or len(_text(card.get(field))) > 300
                 for field in _CARD_FIELDS):
            reason = "missing_or_overlong_description"
        elif key in seen:
            reason = "duplicate_trajectory_card"
        if reason:
            rejected.append({"reason": reason, "claim": card})
            continue
        seen.add(key)
        cards.append({"task_id": task, "rollout_id": rollout, "quote": quote,
                      **{field: _text(card[field]) for field in _CARD_FIELDS},
                      "route": route, "hard": int(bool(sources[key][1].get("hard")))})
    result["cross_group_cards"] = cards
    result["cross_group_card_audit"] = {
        "proposed_cards": raw, "excluded_cards": rejected,
        "source_errors": source_errors, "valid_card_count": len(cards),
        "unrepresented_rollout_ids": sorted(key[1] for key in sources if key not in seen),
        "semantic_status": "optimizer_interpretation_not_causal_verification",
    }
    return result


def collect_cross_group_evidence(raw_patches: list) -> dict:
    """Retain evidence from no-edit analyst results as well as useful patches."""
    cards = {}
    for result in raw_patches:
        if not isinstance(result, dict) or "cross_group_card_audit" not in result:
            continue
        for card in result.get("cross_group_cards", []):
            if isinstance(card, dict) and card.get("route") in _ROUTES:
                cards.setdefault((card["task_id"], card["rollout_id"]), card)
    ordered = [deepcopy(cards[key]) for key in sorted(cards)]
    return {
        "cards": ordered,
        "card_count": len(ordered),
        "distinct_task_count": len({row["task_id"] for row in ordered}),
        "route_counts": dict(Counter(row["route"] for row in ordered)),
        "source": "current_training_step_only",
    }


def annotate_cross_group_review(patch: dict, evidence: dict, *, status: str) -> dict:
    """Audit returned references without filtering edits or changing support."""
    result = deepcopy(patch)
    cards = {(row["task_id"], row["rollout_id"]): row for row in evidence.get("cards", [])}
    review_count = 0
    for edit in result.get("edits", []):
        raw = edit.get("cross_group_review")
        valid, excluded = [], []
        if isinstance(raw, dict):
            refs = raw.get("evidence", [])
            if not isinstance(refs, list):
                refs = []
            seen = set()
            for ref in refs:
                if not isinstance(ref, dict):
                    excluded.append({"reason": "malformed_reference", "claim": ref})
                    continue
                key = _text(ref.get("task_id")), _text(ref.get("rollout_id"))
                relation = _text(ref.get("relation"))
                reason = _text(ref.get("reason"))
                if key not in cards or relation not in {"supports", "limits", "contradicts"} or not reason:
                    excluded.append({"reason": "invalid_evidence_reference", "claim": ref})
                    continue
                if (key, relation) in seen:
                    continue
                seen.add((key, relation))
                valid.append({**cards[key], "relation": relation, "reason": reason})
        if valid:
            review_count += 1
        edit["cross_group_review_audit"] = {
            "proposed_review": raw, "valid_evidence": valid,
            "excluded_references": excluded,
            "distinct_task_count": len({row["task_id"] for row in valid}),
            "distinct_route_count": len({row["route"] for row in valid}),
            "semantic_status": "optimizer_judgment_not_causal_credit",
        }
    result["cross_group_audit"] = {
        "status": status, "available_card_count": len(cards),
        "available_routes": sorted({row["route"] for row in cards.values()}),
        "edits_with_traceable_review": review_count,
    }
    return result
