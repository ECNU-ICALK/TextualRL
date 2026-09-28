"""TextualRL Aggregate stage — hierarchical patch merging.

The Aggregate stage takes independently-generated patches from the Reflect
stage and merges them into a single coherent patch via hierarchical LLM calls.
Failure-driven patches take priority over success-driven ones.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from textualrl.model import chat_optimizer
from textualrl.optimizer.cross_group import (
    MERGE_SUFFIX as CROSS_GROUP_MERGE_SUFFIX,
    annotate_cross_group_review,
    cross_group_evidence_enabled,
)
from textualrl.optimizer.group_relative import (
    get_group_relative_edit_credit_config,
    is_group_relative_edit_credit_enabled,
)
from textualrl.optimizer.meta_skill import format_meta_skill_context
from textualrl.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from textualrl.prompts import load_prompt
from textualrl.utils import extract_json


_ATOMIC_MERGE_SUFFIX = """

## Atomic Evidence-Preserving Merge Contract

Keep each output item to one independently actionable behavior. Do not bundle
unrelated rules merely to reduce item count. When equivalent source clauses
appear in multiple patches, merge their wording but preserve the union of
`evidence_task_ids`, `provenance_task_ids`, and `provenance_rollout_ids`.
Never invent an evidence ID. If two clauses have different supporting tasks or
different preconditions, emit separate items. These metadata fields are audit
evidence and must not be copied into the executable skill text.
"""

_EVIDENCE_ATOMIC_MERGE_SUFFIX = """

Keep reasoning rules separate from answer-rendering rules. Preserve each
top-level Markdown clause as an independent evidence unit. Never merge a
supported condition-action rule with an unsupported formatting heuristic. An
output-contract clause must retain the mixed-task causal trace that directly
supports it; otherwise omit that clause. Cross-task support is a priority, not
a candidate-level minimum quota.
"""


def _atomic_merge_prompt(system_prompt: str, update_mode: str) -> str:
    if (
        is_group_relative_edit_credit_enabled()
        and not is_full_rewrite_minibatch_mode(update_mode)
    ):
        prompt = system_prompt.rstrip() + _ATOMIC_MERGE_SUFFIX
        runtime = get_group_relative_edit_credit_config()
        if any(
            (
                runtime.get("soft_cross_task_target", False),
                runtime.get("atomize_all_edit_clauses", False),
                runtime.get(
                    "output_contract_require_direct_causal_evidence", False
                ),
            )
        ):
            prompt += _EVIDENCE_ATOMIC_MERGE_SUFFIX
        return prompt
    return system_prompt


# ── Internal helpers ──────────────────────────────────────────────────────────

def _merge_batch(
    skill_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    meta_skill_context: str = "",
    level: int = 1,
) -> dict:
    """Call optimizer LLM to merge a batch of patches into one."""
    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n{patches_text}"
    )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=system_prompt,
            user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(update_mode) else 16384,
            retries=3,
            stage="merge",
        )
        merged = extract_json(response)
        key = payload_key(update_mode)
        if merged and key in merged:
            for e in merged.get(key, []):
                e["merge_level"] = level
            return merged
    except Exception:  # noqa: BLE001
        pass
    # Fallback: concatenate all edits
    all_edits = []
    for p in patches:
        for e in get_payload_items(p, update_mode):
            e.setdefault("merge_level", level)
            all_edits.append(e)
    return {"reasoning": "fallback concatenation", payload_key(update_mode): all_edits}


def _hierarchical_merge(
    skill_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    batch_size: int,
    verbose: bool,
    label: str = "",
    workers: int = 16,
    meta_skill_context: str = "",
) -> dict:
    """Hierarchically merge N patches using the given system prompt.

    Same-level batches are executed in PARALLEL via ThreadPoolExecutor.
    """
    if not patches:
        return {"reasoning": "no patches", payload_key(update_mode): []}
    if len(patches) == 1:
        return patches[0]

    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        batches: list[tuple[int, list[dict]]] = []
        for i in range(0, len(current), batch_size):
            batch = current[i : i + batch_size]
            batches.append((i, batch))

        if verbose:
            print(
                f"    [aggregate {label}] level={level}  "
                f"{len(current)} patches → {len(batches)} batches "
                f"(parallel, batch_size={batch_size})"
            )

        next_level: list[dict | None] = [None] * len(batches)

        to_merge: list[tuple[int, list[dict]]] = []
        for idx, (i, batch) in enumerate(batches):
            if len(batch) == 1:
                next_level[idx] = batch[0]
            else:
                to_merge.append((idx, batch))

        if to_merge:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _merge_batch, skill_content, batch, system_prompt, update_mode,
                        meta_skill_context, level,
                    ): idx
                    for idx, batch in to_merge
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    next_level[idx] = fut.result()
                    if verbose:
                        batch_i, batch_data = batches[idx]
                        n_edits = len(get_payload_items(next_level[idx], update_mode))
                        print(
                            f"      [aggregate {label}] level={level} "
                            f"batch [{batch_i}:{batch_i+len(batch_data)}] "
                            f"→ 1 patch ({n_edits} {payload_label(update_mode)})"
                        )

        current = [x for x in next_level if x is not None]

    return current[0]


# ── Public API ────────────────────────────────────────────────────────────────


def _cross_group_final_merge(skill_content: str, patches: list[dict], evidence: dict,
                            system_prompt: str, meta_skill_context: str) -> dict:
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Candidate patches\n{json.dumps(patches, ensure_ascii=False, indent=2)}\n\n"
        "## Current-training-step behavioral evidence\n"
        + json.dumps(evidence["cards"], ensure_ascii=False, indent=2)
    )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=system_prompt.rstrip() + CROSS_GROUP_MERGE_SUFFIX,
            user=user, max_completion_tokens=16384, retries=3, stage="merge",
        )
        merged = extract_json(response)
        if (isinstance(merged, dict) and isinstance(merged.get("edits"), list)
                and all(isinstance(edit, dict) for edit in merged["edits"])):
            return annotate_cross_group_review(merged, evidence, status="model_reviewed")
    except Exception as exc:  # Match the existing merge fallback, with a visible audit.
        print(f"    [cross-group evidence] merge failed: {type(exc).__name__}")
    fallback = (patches[0] if len(patches) == 1 else {
        "reasoning": "fallback: failure first, then success",
        "edits": [edit for patch in patches for edit in patch.get("edits", [])],
    })
    return annotate_cross_group_review(fallback, evidence, status="merge_fallback_not_reviewed")

def merge_patches(
    skill_content: str,
    failure_patches: list[dict],
    success_patches: list[dict],
    batch_size: int = 8,
    verbose: bool = True,
    workers: int = 16,
    update_mode: str = "patch",
    meta_skill_context: str = "",
    cross_group_evidence: dict | None = None,
) -> dict:
    """Failure-first hierarchical merge with support count tracking.

    1. Merge failure patches independently (parallel)
    2. Merge success patches independently (parallel)
    3. Final merge: combine both groups with failure priority

    Returns a merged :class:`~textualrl.types.Patch` dict (``edits`` + ``reasoning``).
    """
    if verbose:
        print(
            f"    [3/6 AGGREGATE] "
            f"failure={len(failure_patches)} success={len(success_patches)} "
            f"(parallel, workers={workers})"
        )

    update_mode = normalize_update_mode(update_mode)
    if is_full_rewrite_minibatch_mode(update_mode):
        merge_failure_prompt = load_prompt("merge_failure_full_rewrite")
        merge_success_prompt = load_prompt("merge_success_full_rewrite")
        merge_final_prompt = load_prompt("merge_final_full_rewrite")
    elif is_rewrite_mode(update_mode):
        merge_failure_prompt = load_prompt("merge_failure_rewrite")
        merge_success_prompt = load_prompt("merge_success_rewrite")
        merge_final_prompt = load_prompt("merge_final_rewrite")
    else:
        merge_failure_prompt = load_prompt("merge_failure")
        merge_success_prompt = load_prompt("merge_success")
        merge_final_prompt = load_prompt("merge_final")

    merge_failure_prompt = _atomic_merge_prompt(
        merge_failure_prompt, update_mode
    )
    merge_success_prompt = _atomic_merge_prompt(
        merge_success_prompt, update_mode
    )
    merge_final_prompt = _atomic_merge_prompt(merge_final_prompt, update_mode)

    failure_merged = _hierarchical_merge(
        skill_content, failure_patches, merge_failure_prompt, update_mode,
        batch_size, verbose, label="failure", workers=workers,
        meta_skill_context=meta_skill_context,
    )

    success_merged = _hierarchical_merge(
        skill_content, success_patches, merge_success_prompt, update_mode,
        batch_size, verbose, label="success", workers=workers,
        meta_skill_context=meta_skill_context,
    )

    f_edits = get_payload_items(failure_merged, update_mode)
    s_edits = get_payload_items(success_merged, update_mode)

    if not f_edits and not s_edits:
        return {"reasoning": "no updates from either group", payload_key(update_mode): []}
    if (cross_group_evidence_enabled() and update_mode == "patch"
            and cross_group_evidence and cross_group_evidence.get("cards")):
        candidates = [p for p in (failure_merged, success_merged) if p.get("edits")]
        result = _cross_group_final_merge(
            skill_content, candidates, cross_group_evidence,
            merge_final_prompt, meta_skill_context,
        )
        if verbose:
            audit = result["cross_group_audit"]
            print(f"    [cross-group evidence] cards={audit['available_card_count']} "
                  f"reviewed_edits={audit['edits_with_traceable_review']} "
                  f"status={audit['status']}")
        return result
    if cross_group_evidence_enabled() and verbose:
        print("    [cross-group evidence] no traceable cards; using original merge")
    if not s_edits:
        return failure_merged
    if not f_edits:
        return success_merged

    combined_patches = [failure_merged, success_merged]
    combined_text = json.dumps(combined_patches, ensure_ascii=False, indent=2)
    if is_full_rewrite_minibatch_mode(update_mode):
        item_label = payload_label(update_mode)
        user = (
            f"## Current Skill\n{skill_content}\n\n"
            f"## Two pre-merged candidate groups to combine\n"
            f"Group 1 (from failed trajectories): "
            f"{len(f_edits)} {item_label}\n"
            f"Group 2 (from successful trajectories): "
            f"{len(s_edits)} {item_label}\n\n"
            f"{combined_text}"
        )
    else:
        user = (
            f"## Current Skill\n{skill_content}\n\n"
            f"## Two pre-merged patch groups to combine\n"
            f"Group 1 (failure-driven, HIGH priority): "
            f"{len(f_edits)} edits\n"
            f"Group 2 (success-driven, lower priority): "
            f"{len(s_edits)} edits\n\n"
            f"{combined_text}"
        )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=merge_final_prompt,
            user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(update_mode) else 16384,
            retries=3,
            stage="merge",
        )
        final = extract_json(response)
        key = payload_key(update_mode)
        if final and key in final:
            if verbose:
                print(
                    f"    [aggregate final] "
                    f"{len(f_edits)}+{len(s_edits)} → {len(final[key])} {payload_label(update_mode)}"
                )
            return final
    except Exception:  # noqa: BLE001
        pass

    return {
        "reasoning": "fallback: failure first, then success",
        payload_key(update_mode): f_edits + s_edits,
    }
