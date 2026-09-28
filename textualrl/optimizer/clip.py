"""TextualRL gradient clipping — LLM-driven edit ranking and selection.

Analogous to gradient clipping in neural network training: ranks candidate
edits by importance and selects the top-L to apply, controlling the
effective step size. Previously core/select.py.
"""
from __future__ import annotations

from textualrl.model import chat_optimizer
from textualrl.optimizer.group_relative import select_group_relative_candidate
from textualrl.optimizer.meta_skill import format_meta_skill_context
from textualrl.optimizer.update_modes import (
    describe_item,
    get_payload_items,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from textualrl.prompts import load_prompt
from textualrl.utils import extract_json


# ── Public API ────────────────────────────────────────────────────────────────

def rank_and_select(
    skill_content: str,
    patch: dict,
    max_edits: int,
    meta_skill_context: str = "",
    update_mode: str = "patch",
    group_relative_edit_credit: bool = False,
    group_relative_max_specific_singletons: int = 1,
    group_relative_min_cross_task_support: int = 2,
    group_relative_max_exploratory: int = 1,
    group_relative_min_atomic_support_fraction: float = 1.0,
    group_relative_min_candidate_cross_task_edits: int = 0,
    group_relative_max_contrastive_edits: int = -1,
    group_relative_max_singleton_edits: int = -1,
    group_relative_block_singleton_stable_hypotheses: bool = False,
    group_relative_actionable_priority_bonus: float = 0.0,
    group_relative_causal_trace_priority_bonus: float = 0.0,
) -> dict:
    """Use a optimizer LLM to rank edits by importance, then keep top-L.

    If the edit pool is within budget, returns the patch unchanged.
    Otherwise, calls the optimizer to rank and select the most impactful edits.

    Parameters
    ----------
    skill_content : str
        Current skill document.
    patch : dict
        Merged :class:`~textualrl.types.Patch` dict with ``edits`` list.
    max_edits : int
        Maximum number of edits to keep (the "edit budget").

    Returns
    -------
    dict
        :class:`~textualrl.types.Patch` dict with selected edits and
        optional ``ranking_details``.
    """
    update_mode = normalize_update_mode(update_mode)
    if group_relative_edit_credit:
        patch, _ = select_group_relative_candidate(
            patch,
            max_edits=max_edits,
            update_mode=update_mode,
            max_specific_singletons=group_relative_max_specific_singletons,
            min_cross_task_support=group_relative_min_cross_task_support,
            max_exploratory=group_relative_max_exploratory,
            min_atomic_support_fraction=(
                group_relative_min_atomic_support_fraction
            ),
            min_candidate_cross_task_edits=(
                group_relative_min_candidate_cross_task_edits
            ),
            max_contrastive_edits=group_relative_max_contrastive_edits,
            max_singleton_edits=group_relative_max_singleton_edits,
            block_singleton_stable_hypotheses=(
                group_relative_block_singleton_stable_hypotheses
            ),
            actionable_priority_bonus=(
                group_relative_actionable_priority_bonus
            ),
            causal_trace_priority_bonus=(
                group_relative_causal_trace_priority_bonus
            ),
        )
    edits = get_payload_items(patch, update_mode)
    if len(edits) <= max_edits:
        return patch

    # Build the edit pool description for the optimizer
    edits_desc = []
    for i, edit in enumerate(edits):
        description = describe_item(edit, update_mode)
        if group_relative_edit_credit:
            credit = edit.get("_group_relative_credit", {})
            description += (
                "  group_credit="
                f"{float(credit.get('credit_score', 0.0) or 0.0):.4f}"
                "  independent_task_support="
                f"{int(credit.get('task_support_count', 0) or 0)}"
                "  mixed_task_support="
                f"{int(credit.get('mixed_task_support_count', 0) or 0)}"
                "  specificity_penalty="
                f"{float(credit.get('specificity_penalty', 0.0) or 0.0):.4f}"
            )
        edits_desc.append(f"[{i}] {description}")

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## {payload_label(update_mode, title=True)} Pool ({len(edits)} {payload_label(update_mode)}, budget={max_edits})\n"
        + "\n".join(edits_desc)
        + f"\n\nSelect the {max_edits} most important {payload_label(update_mode)}. "
        f"Return their 0-based indices in priority order."
    )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    prompt_name = "ranking_rewrite" if is_rewrite_mode(update_mode) else "ranking"

    try:
        response, _ = chat_optimizer(
            system=load_prompt(prompt_name), user=user,
            max_completion_tokens=16384, retries=3, stage="ranking",
        )
        result = extract_json(response)
        if result and "selected_indices" in result:
            indices = result["selected_indices"]
            selected = []
            seen: set[int] = set()
            for idx in indices:
                if (
                    isinstance(idx, int)
                    and 0 <= idx < len(edits)
                    and idx not in seen
                ):
                    selected.append(edits[idx])
                    seen.add(idx)
                if len(selected) >= max_edits:
                    break
            if selected:
                return {
                    "reasoning": patch.get("reasoning", "")
                    + f" [optimizer-ranked: selected {len(selected)}/{len(edits)} {payload_label(update_mode)}]",
                    payload_key(update_mode): selected,
                    "ranking_details": result,
                }
    except Exception:  # noqa: BLE001
        pass

    # Fallback: simple truncation
    return {
        "reasoning": patch.get("reasoning", "")
        + f" [fallback truncated {len(edits)}->{max_edits} {payload_label(update_mode)}]",
        payload_key(update_mode): edits[:max_edits],
    }
