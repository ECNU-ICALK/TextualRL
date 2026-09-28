"""Diverse edit-candidate exploration and auditable pre-validation ranking."""
from __future__ import annotations

from typing import Any

from textualrl.optimizer.edit_value import score_edit_candidate
from textualrl.optimizer.quarantine import candidates_equivalent
from textualrl.optimizer.quarantine import edit_similarity
from textualrl.optimizer.quarantine import partition_runtime_observable_edits
from textualrl.optimizer.update_modes import describe_item, set_payload_items


EXPLORATION_INTENTS = (
    {
        "name": "primary",
        "instruction": (
            "Produce the strongest evidence-supported update under the normal "
            "aggregation policy."
        ),
    },
    {
        "name": "causal_repair",
        "instruction": (
            "Focus on the earliest observable decision that separates high- and "
            "low-return trajectories. Repair its missing precondition instead of "
            "adding a generic final-answer reminder."
        ),
    },
    {
        "name": "constraint_preservation",
        "instruction": (
            "Focus on preventing over-generalization: preserve task-defining "
            "qualifiers, exclusions, and distinctions when evidence shows they "
            "matter. Do not merely paraphrase another candidate."
        ),
    },
    {
        "name": "simplify_or_narrow",
        "instruction": (
            "Prefer deleting, narrowing, or replacing a harmful existing rule over "
            "appending more text. Preserve unrelated capabilities."
        ),
    },
    {
        "name": "evidence_disambiguation",
        "instruction": (
            "Focus on a reusable evidence-selection or ambiguity-resolution "
            "procedure with an observable applicability condition and success signal."
        ),
    },
)


def exploration_intent(index: int) -> dict[str, str]:
    if index < len(EXPLORATION_INTENTS):
        return dict(EXPLORATION_INTENTS[index])
    base = EXPLORATION_INTENTS[1 + (index - 1) % (len(EXPLORATION_INTENTS) - 1)]
    return {
        "name": f"{base['name']}_{index}",
        "instruction": base["instruction"],
    }


def format_candidate_exploration_context(
    *,
    intent: dict[str, str],
    prior_proposals: list[dict],
    update_mode: str,
) -> str:
    """Ask for a behaviorally distinct proposal, not a surface paraphrase."""
    prior_blocks = []
    for proposal in prior_proposals:
        items = proposal.get("items", [])
        rendered = "\n".join(
            f"- {describe_item(item, update_mode)}" for item in items
        )
        prior_blocks.append(
            f"Prior direction `{proposal.get('intent', 'unknown')}`:\n{rendered}"
        )
    prior_text = "\n\n".join(prior_blocks) or "(none)"
    return (
        "## Candidate exploration role\n"
        f"Direction: {intent['name']}\n"
        f"Objective: {intent['instruction']}\n\n"
        "Generate a candidate whose behavioral mechanism is distinct from every "
        "prior complete candidate below. Changing only wording, patch syntax, a "
        "heading, or the order of equivalent instructions does not count as a new "
        "direction. Every edit still requires trajectory evidence.\n\n"
        f"Prior candidates:\n{prior_text}"
    )


def proposal_is_duplicate(
    items: list[dict],
    proposals: list[dict],
    *,
    update_mode: str,
    similarity_threshold: float,
) -> bool:
    def _overlap_similarity(left: list[dict], right: list[dict]) -> float:
        if not left or not right:
            return float(not left and not right)
        edges = sorted(
            (
                edit_similarity(a, b, update_mode=update_mode),
                left_index,
                right_index,
            )
            for left_index, a in enumerate(left)
            for right_index, b in enumerate(right)
        )
        matched_left: set[int] = set()
        matched_right: set[int] = set()
        matched = 0
        for score, left_index, right_index in reversed(edges):
            if score < similarity_threshold:
                break
            if left_index in matched_left or right_index in matched_right:
                continue
            matched_left.add(left_index)
            matched_right.add(right_index)
            matched += 1
        return matched / max(len(left), len(right))

    for proposal in proposals:
        prior_items = proposal.get("items", [])
        if candidates_equivalent(
            items,
            prior_items,
            update_mode=update_mode,
            similarity_threshold=similarity_threshold,
        ):
            return True
        if _overlap_similarity(items, prior_items) >= similarity_threshold:
            return True
    return False


def prune_runtime_unobservable_proposal(
    proposal: dict,
    *,
    update_mode: str,
) -> dict:
    """Remove invalid edits while preserving a proposal's deployable subset."""
    result = dict(proposal)
    items = list(result.get("items", []))
    observable, hits = partition_runtime_observable_edits(
        items,
        update_mode=update_mode,
    )
    if not hits:
        return result

    ranked_patch = dict(result.get("ranked_patch") or {})
    set_payload_items(ranked_patch, observable, update_mode)
    result["ranked_patch"] = ranked_patch
    result["items"] = observable
    result["runtime_observability_pruned"] = [
        {
            "semantic": hit.get("semantic", {}),
            "behavior": hit.get("behavior", {}),
            "matched_reasons": hit.get("matched_reasons", []),
        }
        for hit in hits
    ]
    return result


def rank_candidate_proposals(
    proposals: list[dict],
    memory: dict[str, Any],
    *,
    update_mode: str,
    exploration_weight: float = 0.01,
    novelty_weight: float = 0.005,
    risk_weight: float = 0.005,
    pair_weight: float = 0.25,
) -> list[dict]:
    """Rank feasible proposals by edit-value UCB; preserve a full audit trail."""
    ranked = []
    for index, proposal in enumerate(proposals):
        value = score_edit_candidate(
            proposal.get("items", []),
            memory,
            update_mode=update_mode,
            exploration_weight=exploration_weight,
            novelty_weight=novelty_weight,
            risk_weight=risk_weight,
            pair_weight=pair_weight,
        )
        blocked_reasons = list(proposal.get("blocked_reasons", []))
        selection_score = float(value["score"])
        if blocked_reasons:
            selection_score -= 1_000_000.0
        ranked.append(
            {
                **proposal,
                "proposal_index": index,
                "edit_value": value,
                "blocked_reasons": blocked_reasons,
                "selection_score": round(selection_score, 8),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            len(row.get("items", [])),
            int(row["proposal_index"]),
        )
    )
    for rank, proposal in enumerate(ranked, start=1):
        proposal["selection_rank"] = rank
        proposal["selected"] = rank == 1
    return ranked
