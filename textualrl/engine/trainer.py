"""ReflACT Trainer — the main training loop.

Orchestrates the 6-stage ReflACT pipeline:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject

The trainer is environment-agnostic; all environment-specific logic is
delegated to an :class:`~textualrl.envs.base.EnvAdapter` instance.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import time
from collections import defaultdict

from textualrl.datasets.base import BatchSpec
from textualrl.envs.base import EnvAdapter
from textualrl.evaluation.gate import GateResult, evaluate_gate, select_gate_score
from textualrl.evaluation.paired import build_paired_promotion_audit
from textualrl.gradient.aggregate import merge_patches
from textualrl.optimizer.cross_group import (
    collect_cross_group_evidence,
    configure_cross_group_evidence,
    validate_cross_group_config,
)
from textualrl.optimizer.meta_skill import run_meta_skill
from textualrl.optimizer.clip import rank_and_select
from textualrl.optimizer.group_relative import (
    annotate_merged_patch,
    configure_group_relative_edit_credit,
)
from textualrl.optimizer.lr_autonomous import decide_autonomous_learning_rate
from textualrl.optimizer.rewrite import rewrite_skill_from_suggestions
from textualrl.optimizer.scheduler import build_scheduler
from textualrl.optimizer.skill import apply_patch_with_report
from textualrl.optimizer.quarantine import (
    CANDIDATE_SCOPE,
    PERMANENT_EDIT_SCOPE,
    add_quarantined_candidate,
    append_edit_history_event,
    build_edit_history_event,
    find_quarantined_candidate,
    find_permanently_quarantined_edits,
    format_candidate_resample_context,
    format_permanent_edit_context,
    format_runtime_observability_context,
    load_edit_history,
    load_quarantine,
    partition_runtime_observable_edits,
    promote_repeated_edits,
    save_quarantine,
)
from textualrl.optimizer.appendix import (
    append_to_appendix_field,
    extract_appendix_notes as extract_appendix_notes_from_skill,
    inject_empty_appendix_field,
    _strip_all_appendix_fields,
)
from textualrl.optimizer.skill_aware import (
    configure_skill_aware_reflection,
    consolidate_appendix_notes,
    extract_appendix_notes as extract_appendix_notes_from_result,
)
from textualrl.optimizer.slow_update import (
    build_comparison_pairs,
    extract_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from textualrl.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_label,
    set_payload_items,
    short_item_summary,
)
from textualrl.model import (
    chat_optimizer,
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    configure_minimax_chat,
    configure_qwen_chat,
    get_token_summary,
    reset_token_tracker,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
    set_optimizer_backend,
    set_optimizer_deployment,
)
from textualrl.utils import compute_score, skill_hash


# ── Skill-aware reflection: appendix flush ───────────────────────────────────

def _flush_skill_aware_appendix(
    current_skill: str,
    all_raw_patches: list,
    step_rec: dict,
    step_dir: str,
    cfg: dict,
) -> str:
    """Append this step's EXECUTION_LAPSE notes into the protected appendix.

    Returns the (possibly) updated skill. Must be called on BOTH the normal
    update path and the skip branches: a lapse-only step yields no body
    patches by design (analysts return ``edits: []`` carriers), so the skip
    paths would otherwise silently drop every note of the step.
    """
    step_appendix_notes: list[str] = []
    for rp in all_raw_patches:
        if isinstance(rp, dict):
            step_appendix_notes.extend(extract_appendix_notes_from_result(rp))
    if not step_appendix_notes:
        return current_skill

    before_notes = extract_appendix_notes_from_skill(current_skill)
    current_skill = append_to_appendix_field(
        current_skill, step_appendix_notes,
    )
    after_notes = extract_appendix_notes_from_skill(current_skill)
    n_added = len(after_notes) - len(before_notes)
    step_rec["n_execution_lapse_notes"] = len(step_appendix_notes)
    step_rec["n_appendix_notes_added"] = n_added
    step_rec["n_appendix_notes_total"] = len(after_notes)
    with open(os.path.join(step_dir, "appendix_notes.json"), "w") as f:
        json.dump(
            {
                "step_notes": step_appendix_notes,
                "appendix_after": after_notes,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(
        f"    [skill-aware] +{n_added} appendix note(s) "
        f"(total {len(after_notes)}) from {len(step_appendix_notes)} lapse signal(s)"
    )
    # Threshold-gated LLM consolidation (paper Eq.11): when the
    # appendix grows past N notes, compact it with one optimizer
    # call (dedupe / merge / shorten). 0 disables it. Any failure
    # leaves the appendix unchanged.
    consolidate_threshold = int(
        cfg.get("skill_aware_consolidate_threshold", 0) or 0
    )
    if consolidate_threshold > 0 and len(after_notes) > consolidate_threshold:
        compacted = consolidate_appendix_notes(
            after_notes, chat_fn=chat_optimizer,
        )
        if compacted and len(compacted) < len(after_notes):
            current_skill = append_to_appendix_field(
                _strip_all_appendix_fields(current_skill), compacted,
            )
            step_rec["n_appendix_notes_consolidated"] = len(compacted)
            step_rec["n_appendix_notes_total"] = len(compacted)
            print(
                f"    [skill-aware] consolidated appendix "
                f"{len(after_notes)} -> {len(compacted)} notes"
            )
    return current_skill


# ── Patch normalization ───────────────────────────────────────────────────────

def _normalise_patches(
    raw_patches: list[dict | None],
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict]]:
    """Extract inner 'patch' sub-dict, split into failure/success lists.

    Each element is expected to conform to :class:`~textualrl.types.RawPatch`.
    """
    mode = normalize_update_mode(update_mode)
    failure: list[dict] = []
    success: list[dict] = []
    for p in raw_patches:
        if not isinstance(p, dict):
            continue
        inner = p.get("patch", p)
        if not isinstance(inner, dict):
            continue
        items = get_payload_items(inner, mode)
        if not items:
            continue
        support = max(int(p.get("batch_size", 0) or 0), 1)
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_type", p.get("source_type", "failure"))
                item.setdefault("support_count", support)
        if p.get("source_type", "failure") == "success":
            success.append(inner)
        else:
            failure.append(inner)
    return failure, success


def _normalise_longitudinal_pair_policy(policy: str | None) -> str:
    raw = str(policy or "mixed").strip().lower()
    aliases = {
        "mixed": "mixed",
        "default": "mixed",
        "random": "mixed",
        "all": "mixed",
        "changed": "changed",
        "change": "changed",
        "delta": "changed",
        "10_01": "changed",
        "01_10": "changed",
        "unchanged": "unchanged",
        "stable": "unchanged",
        "same": "unchanged",
        "00_11": "unchanged",
    }
    if raw not in aliases:
        raise ValueError(
            "optimizer.longitudinal_pair_policy must be one of "
            "mixed, changed, unchanged"
        )
    return aliases[raw]


def _probe_result_key(row: dict) -> str:
    group_id = str(row.get("rollout_group_id") or "")
    if group_id:
        return group_id
    item_id = str(row.get("id") or "")
    return re.sub(r"__sample_\d+_of_\d+$", "", item_id)


def _choose_candidate_branch(branches: list[dict]) -> dict:
    """Choose the best validation-eligible branch, preserving standard ties."""
    if not branches:
        raise ValueError("candidate branch list must not be empty")
    eligible = [branch for branch in branches if branch.get("eligible", False)]
    if not eligible:
        return branches[0]
    return max(
        eligible,
        key=lambda branch: (
            float(branch.get("gate_score", 0.0) or 0.0),
            int(branch.get("tie_priority", 0) or 0),
        ),
    )


def _competition_loss_has_attributable_regression(
    branch: dict,
    winner: dict,
) -> bool:
    """Quarantine only a losing branch with paired evidence of harm.

    A safe candidate can lose because another candidate gains more. Treating
    that loss as negative edit evidence poisons future exploration, so branch
    rank alone is never a quarantine signal.
    """
    if branch is winner or str(branch.get("name") or "") != "group_relative":
        return False
    paired = branch.get("paired")
    if not isinstance(paired, dict):
        return False
    return int(paired.get("harmful_count", 0) or 0) > 0


def _build_paired_risk_decision(
    paired_audit: dict,
    *,
    harm_weight: float = 2.0,
    min_margin: float = 0.0,
) -> dict:
    """Accept aggregate gains only when they outweigh paired regressions."""
    beneficial = int(paired_audit.get("beneficial_count", 0) or 0)
    harmful = int(paired_audit.get("harmful_count", 0) or 0)
    matched = int(paired_audit.get("matched_count", 0) or 0)
    missing = list(paired_audit.get("missing_candidate_ids", []) or [])
    extra = list(paired_audit.get("extra_candidate_ids", []) or [])
    risk_score = beneficial - float(harm_weight) * harmful
    accepted = (
        matched > 0
        and beneficial > 0
        and not missing
        and not extra
        and risk_score > float(min_margin)
    )
    return {
        "accepted": accepted,
        "beneficial_count": beneficial,
        "harmful_count": harmful,
        "matched_count": matched,
        "harm_weight": float(harm_weight),
        "min_margin": float(min_margin),
        "risk_score": round(risk_score, 6),
        "missing_candidate_ids": missing,
        "extra_candidate_ids": extra,
        "rejection_reasons": [
            reason
            for reason, present in (
                ("no_matched_items", matched <= 0),
                ("no_beneficial_transition", beneficial <= 0),
                ("missing_candidate_items", bool(missing)),
                ("extra_candidate_items", bool(extra)),
                ("insufficient_risk_margin", risk_score <= float(min_margin)),
            )
            if present
        ],
    }


def _group_probe_scores(results: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate probe rollouts by independent source task."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        if not isinstance(row, dict):
            continue
        task_id = _probe_result_key(row)
        if task_id:
            grouped[task_id].append(row)
    scores: dict[str, dict[str, float]] = {}
    for task_id, rows in grouped.items():
        scores[task_id] = {
            "hard": sum(float(row.get("hard", 0.0) or 0.0) for row in rows)
            / len(rows),
            "soft": sum(float(row.get("soft", 0.0) or 0.0) for row in rows)
            / len(rows),
            "rollout_count": len(rows),
        }
    return scores


def _build_probe_comparison(
    baseline_results: list[dict],
    candidate_results: list[dict],
    task_ids: list[str],
    *,
    require_hard_gain: bool = False,
) -> dict:
    """Build a task-paired, hard-non-degrading train-side decision."""
    baseline = _group_probe_scores(baseline_results)
    candidate = _group_probe_scores(candidate_results)
    paired = []
    missing = []
    for task_id in task_ids:
        if task_id not in baseline or task_id not in candidate:
            missing.append(task_id)
            continue
        hard_delta = candidate[task_id]["hard"] - baseline[task_id]["hard"]
        soft_delta = candidate[task_id]["soft"] - baseline[task_id]["soft"]
        paired.append(
            {
                "task_id": task_id,
                "baseline": baseline[task_id],
                "candidate": candidate[task_id],
                "hard_delta": round(hard_delta, 6),
                "soft_delta": round(soft_delta, 6),
            }
        )
    mean_hard_delta = (
        sum(row["hard_delta"] for row in paired) / len(paired)
        if paired else 0.0
    )
    mean_soft_delta = (
        sum(row["soft_delta"] for row in paired) / len(paired)
        if paired else 0.0
    )
    harmful = [row for row in paired if row["hard_delta"] < -1e-12]
    no_required_gain = require_hard_gain and mean_hard_delta <= 1e-12
    accepted = (
        bool(paired)
        and not missing
        and not harmful
        and mean_hard_delta >= -1e-12
        and not no_required_gain
    )
    return {
        "accepted": accepted,
        "matched_task_count": len(paired),
        "missing_task_ids": missing,
        "harmful_task_count": len(harmful),
        "mean_hard_delta": round(mean_hard_delta, 6),
        "mean_soft_delta": round(mean_soft_delta, 6),
        "require_hard_gain": bool(require_hard_gain),
        "paired": paired,
        "rejection_reasons": [
            reason
            for reason, present in (
                ("missing_paired_result", bool(missing)),
                ("task_hard_regression", bool(harmful)),
                ("negative_mean_hard_delta", mean_hard_delta < -1e-12),
                ("no_task_hard_gain", no_required_gain),
                ("no_matched_tasks", not paired),
            )
            if present
        ],
    }


def _edit_probe_task_ids(item: dict, max_tasks: int) -> list[str]:
    """Choose direct evidence tasks with deterministic status diversity.

    Provenance-only tasks are deliberately excluded: they show that an edit was
    generated in the same analyst call, not that the task supports this edit.
    """
    ids = item.get("evidence_task_ids", [])
    if isinstance(ids, str):
        ids = [ids]
    unique: list[str] = []
    for value in ids if isinstance(ids, list) else []:
        task_id = str(value)
        if task_id and task_id not in unique:
            unique.append(task_id)
    credit = item.get("_group_relative_credit", {})
    statuses = (
        credit.get("evidence_statuses", {})
        if isinstance(credit, dict)
        and isinstance(credit.get("evidence_statuses", {}), dict)
        else {}
    )
    contrasts = (
        credit.get("evidence_contrast_strengths", {})
        if isinstance(credit, dict)
        and isinstance(credit.get("evidence_contrast_strengths", {}), dict)
        else {}
    )
    status_order = ("mixed", "stable_failure", "stable_success", "unknown")
    buckets: dict[str, list[str]] = {status: [] for status in status_order}
    for task_id in unique:
        status = str(statuses.get(task_id) or "unknown")
        if status not in buckets:
            status = "unknown"
        buckets[status].append(task_id)
    for status, task_ids in buckets.items():
        task_ids.sort(
            key=lambda task_id: (
                -float(contrasts.get(task_id, 0.0) or 0.0),
                unique.index(task_id),
            )
        )

    limit = max(1, int(max_tasks))
    selected: list[str] = []
    for status in status_order:
        if buckets[status] and len(selected) < limit:
            selected.append(buckets[status].pop(0))
    remaining = sorted(
        [task_id for task_ids in buckets.values() for task_id in task_ids],
        key=lambda task_id: (
            -float(contrasts.get(task_id, 0.0) or 0.0),
            unique.index(task_id),
        ),
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected


def _run_group_relative_per_edit_probe(
    *,
    adapter: EnvAdapter,
    ranked_patch: dict,
    current_skill: str,
    train_items: list[dict],
    step_dir: str,
    out_root: str,
    update_mode: str,
    max_tasks_per_edit: int,
    rollout_count: int,
    seed: int,
    fail_open: bool,
) -> tuple[dict, dict]:
    """Probe each selected edit only on its training-side source tasks.

    This is intentionally separate from the official validation gate. It never
    reads ``valid_seen`` or ``valid_unseen`` and does not change their size,
    sampling, metric, or acceptance policy.
    """
    audit: dict = {
        "enabled": True,
        "scope": "train_source_tasks_only",
        "official_validation_changed": False,
        "rollouts_per_task": max(1, int(rollout_count)),
        "max_tasks_per_edit": max(1, int(max_tasks_per_edit)),
        "fail_open": bool(fail_open),
        "items": [],
    }
    ranked_items = list(get_payload_items(ranked_patch, update_mode))
    if normalize_update_mode(update_mode) != "patch":
        audit.update({"status": "unsupported_update_mode", "kept_count": len(ranked_items)})
        return ranked_patch, audit

    item_by_id: dict[str, dict] = {}
    for item in train_items:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("id") or "")
        if task_id:
            item_by_id.setdefault(task_id, item)
    if not item_by_id:
        audit.update({"status": "unavailable_train_payload", "kept_count": len(ranked_items)})
        return ranked_patch, audit

    requests = []
    union_ids: list[str] = []
    for index, item in enumerate(ranked_items):
        requested_ids = _edit_probe_task_ids(item, max_tasks_per_edit)
        matched_ids = [task_id for task_id in requested_ids if task_id in item_by_id]
        requests.append((index, item, requested_ids, matched_ids))
        for task_id in matched_ids:
            if task_id not in union_ids:
                union_ids.append(task_id)

    if not union_ids:
        audit.update({
            "status": "no_source_tasks_matched",
            "kept_count": len(ranked_items) if fail_open else 0,
        })
        result = dict(ranked_patch)
        set_payload_items(result, ranked_items if fail_open else [], update_mode)
        return result, audit

    probe_root = os.path.join(step_dir, "group_relative_per_edit_probe")
    os.makedirs(probe_root, exist_ok=True)

    def _rollout(items: list[dict], skill: str, path: str, probe_seed: int) -> list[dict]:
        batch = BatchSpec(
            phase="train",
            split="train",
            seed=probe_seed,
            batch_size=len(items),
            payload=items,
        )
        env = adapter.build_env_from_batch(batch, out_root=out_root)
        env = adapter.expand_same_task_rollouts(env, max(1, int(rollout_count)))
        return adapter.rollout(env, skill, path, use_eval_feedback=True)

    try:
        baseline_results = _rollout(
            [item_by_id[task_id] for task_id in union_ids],
            current_skill,
            os.path.join(probe_root, "baseline"),
            seed,
        )
    except Exception as exc:  # noqa: BLE001
        audit.update(
            {
                "status": "baseline_probe_error",
                "error": f"{type(exc).__name__}: {exc}",
                "kept_count": len(ranked_items) if fail_open else 0,
            }
        )
        result = dict(ranked_patch)
        set_payload_items(result, ranked_items if fail_open else [], update_mode)
        return result, audit

    kept: list[dict] = []
    for index, item, requested_ids, matched_ids in requests:
        credit = item.get("_group_relative_credit", {})
        selection_category = str(
            item.get("_group_relative_selection_category") or ""
        )
        statuses = {
            str(value)
            for value in (
                credit.get("evidence_statuses", {}).values()
                if isinstance(credit, dict)
                and isinstance(credit.get("evidence_statuses", {}), dict)
                else []
            )
        }
        if selection_category in {"cross_task", "contrastive"}:
            require_hard_gain = False
        elif selection_category in {
            "stable_hypothesis",
            "exploratory",
            "preservation",
        }:
            require_hard_gain = True
        else:
            # Backward-compatible fallback for pre-V1.7 artifacts. Mixed
            # success/failure evidence already provides a causal contrast, so
            # a source-task hard tie is sufficient. Stable-failure-only rules
            # remain hypotheses and must demonstrate a hard gain.
            require_hard_gain = (
                "stable_failure" in statuses and "mixed" not in statuses
            )
        record = {
            "edit_index": index,
            "summary": short_item_summary(item, update_mode),
            "requested_task_ids": requested_ids,
            "matched_task_ids": matched_ids,
            "evidence_statuses": sorted(statuses),
            "selection_category": selection_category or None,
            "require_hard_gain": require_hard_gain,
        }
        if not matched_ids:
            record.update(
                {
                    "accepted": bool(fail_open),
                    "status": "no_source_tasks_matched",
                }
            )
            if fail_open:
                kept.append(item)
            audit["items"].append(record)
            continue

        single_patch = dict(ranked_patch)
        set_payload_items(single_patch, [item], update_mode)
        single_skill, apply_report = apply_patch_with_report(
            current_skill, single_patch
        )
        record["apply_report"] = apply_report
        if single_skill == current_skill:
            record.update({"accepted": False, "status": "edit_not_applied"})
            audit["items"].append(record)
            continue
        try:
            candidate_results = _rollout(
                [item_by_id[task_id] for task_id in matched_ids],
                single_skill,
                os.path.join(probe_root, f"edit_{index:02d}"),
                seed,
            )
            baseline_subset = [
                row
                for row in baseline_results
                if _probe_result_key(row) in set(matched_ids)
            ]
            comparison = _build_probe_comparison(
                baseline_subset,
                candidate_results,
                matched_ids,
                require_hard_gain=require_hard_gain,
            )
            record.update(comparison)
            record["status"] = "accepted" if comparison["accepted"] else "rejected"
            if comparison["accepted"]:
                kept.append(item)
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "accepted": bool(fail_open),
                    "status": "candidate_probe_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if fail_open:
                kept.append(item)
        audit["items"].append(record)

    audit.update(
        {
            "status": "completed",
            "input_count": len(ranked_items),
            "kept_count": len(kept),
            "rejected_count": len(ranked_items) - len(kept),
            "baseline_task_count": len(_group_probe_scores(baseline_results)),
        }
    )
    result = dict(ranked_patch)
    set_payload_items(result, kept, update_mode)
    result["group_relative_per_edit_probe"] = audit
    return result, audit


def _normalise_lr_control_mode(mode: str | None) -> str:
    raw = str(mode or "fixed").strip().lower()
    aliases = {
        "fixed": "fixed",
        "manual": "fixed",
        "scheduler": "fixed",
        "scheduled": "fixed",
        "autonomous": "autonomous",
        "auto": "autonomous",
        "optimizer": "autonomous",
        "none": "none",
        "off": "none",
        "no_lr": "none",
    }
    if raw not in aliases:
        raise ValueError("optimizer.lr_control_mode must be one of fixed, autonomous, none")
    return aliases[raw]


def _filter_longitudinal_pairs(pairs: list[dict], policy: str) -> list[dict]:
    if policy == "mixed":
        return pairs
    if policy == "changed":
        keep = {"improved", "regressed"}
    elif policy == "unchanged":
        keep = {"persistent_fail", "stable_success"}
    else:
        raise ValueError(f"Unknown longitudinal pair policy: {policy}")
    return [p for p in pairs if p.get("category") in keep]


def _pair_category_counts(pairs: list[dict]) -> dict[str, int]:
    counts = {
        "improved": 0,
        "regressed": 0,
        "persistent_fail": 0,
        "stable_success": 0,
    }
    for pair in pairs:
        cat = str(pair.get("category", ""))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _safe_pair_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe[:80] or "item"


def _build_longitudinal_pairs(
    *,
    adapter: EnvAdapter,
    dataloader,
    prev_skill: str,
    curr_skill: str,
    initial_items: list[dict],
    initial_prev_results: list[dict],
    initial_curr_results: list[dict],
    prev_rollout_dir: str,
    curr_rollout_dir: str,
    policy: str,
    target_n: int,
    seed: int,
    out_root: str,
) -> tuple[list[dict], list[dict]]:
    """Build longitudinal pairs, optionally filtering by change category.

    ``mixed`` preserves the legacy behavior exactly. ``changed`` keeps only
    10/01 pairs and attempts to top up to ``target_n`` by scanning the train
    split once. ``unchanged`` keeps only 00/11 pairs and does not top up.
    """
    all_pairs = build_comparison_pairs(
        initial_prev_results,
        initial_curr_results,
        initial_items,
        prev_rollout_dir=prev_rollout_dir,
        curr_rollout_dir=curr_rollout_dir,
    )
    selected_pairs = _filter_longitudinal_pairs(all_pairs, policy)
    if policy != "changed" or len(selected_pairs) >= target_n or dataloader is None:
        return selected_pairs, all_pairs

    train_items = list(getattr(dataloader, "train_items", []) or [])
    if not train_items:
        return selected_pairs, all_pairs

    seen_ids = {str(p.get("id", "")) for p in all_pairs}
    rng = random.Random(seed)
    candidates = list(train_items)
    rng.shuffle(candidates)
    candidates = [item for item in candidates if str(item.get("id", "")) not in seen_ids]

    for idx, item in enumerate(candidates):
        if len(selected_pairs) >= target_n:
            break
        item_id = _safe_pair_id(str(item.get("id", f"item_{idx}")))
        batch = BatchSpec(
            phase="train",
            split="train",
            seed=seed + idx + 1,
            batch_size=1,
            payload=[item],
        )
        env = adapter.build_env_from_batch(batch, out_root=out_root)
        prev_dir = os.path.join(prev_rollout_dir, "topup", item_id)
        curr_dir = os.path.join(curr_rollout_dir, "topup", item_id)
        prev_results = adapter.rollout(env, prev_skill, prev_dir)
        curr_results = adapter.rollout(env, curr_skill, curr_dir)
        pair = build_comparison_pairs(
            prev_results,
            curr_results,
            [item],
            prev_rollout_dir=prev_dir,
            curr_rollout_dir=curr_dir,
        )
        all_pairs.extend(pair)
        selected_pairs.extend(_filter_longitudinal_pairs(pair, policy))

    return selected_pairs[:target_n], all_pairs


# ── History / persistence helpers ─────────────────────────────────────────────

_SECRET_KEYS = {
    "azure_api_key",
    "api_key",
    "openai_api_key",
}


def _redact_value(val: str) -> str:
    if len(val) <= 8:
        return "*" * len(val)
    return f"{val[:4]}...{val[-4:]}"


def _redact_cfg(cfg: dict) -> dict:
    redacted = dict(cfg)
    for key in list(redacted):
        if key.lower() in _SECRET_KEYS and redacted.get(key):
            redacted[key] = _redact_value(str(redacted[key]))
    return redacted

def _load_history(out_root: str) -> list[dict]:
    path = os.path.join(out_root, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_history(out_root: str, history: list[dict]) -> None:
    path = os.path.join(out_root, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _save_skill(out_root: str, step: int, content: str) -> None:
    skills_dir = os.path.join(out_root, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    with open(os.path.join(skills_dir, f"skill_v{step:04d}.md"), "w") as f:
        f.write(content)


def _load_skill(out_root: str, step: int) -> str:
    path = os.path.join(out_root, "skills", f"skill_v{step:04d}.md")
    with open(path) as f:
        return f.read()


def _load_meta_skill_content(out_root: str, epoch: int) -> str:
    if epoch <= 0:
        return ""
    path = os.path.join(
        out_root, "meta_skill", f"epoch_{epoch:02d}", "meta_skill_result.json",
    )
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            result = json.load(f)
        return str(result.get("meta_skill_content", "")).strip()
    except Exception:
        return ""


def _load_runtime_state(out_root: str) -> dict | None:
    path = os.path.join(out_root, "runtime_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def _save_runtime_state(out_root: str, state: dict) -> None:
    path = os.path.join(out_root, "runtime_state.json")
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_epoch_step_buffer(
    out_root: str,
    *,
    epoch: int,
    steps_per_epoch: int,
    last_completed_step: int,
) -> tuple[list[dict], list[int]]:
    """Rebuild the current epoch's optimizer context after a step resume."""
    first_step = (epoch - 1) * steps_per_epoch + 1
    final_step = min(last_completed_step, epoch * steps_per_epoch)
    if final_step < first_step:
        return [], []

    buffer: list[dict] = []
    missing: list[int] = []
    for step in range(first_step, final_step + 1):
        path = os.path.join(
            out_root,
            "steps",
            f"step_{step:04d}",
            "trajectory_digest.json",
        )
        try:
            with open(path) as f:
                entry = json.load(f)
        except (OSError, json.JSONDecodeError):
            missing.append(step)
            continue
        if isinstance(entry, dict):
            buffer.append(entry)
        else:
            missing.append(step)
    return buffer, missing


def _resolve_train_size(cfg: dict, dataloader) -> int:
    configured = int(cfg.get("train_size", 0) or 0)
    inferred: int | None = None

    if dataloader is not None:
        getter = getattr(dataloader, "get_train_size", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if value is not None:
                inferred = int(value)
        elif hasattr(dataloader, "train_items"):
            try:
                inferred = len(getattr(dataloader, "train_items"))
            except Exception:
                inferred = None

    if inferred is not None and inferred <= 0:
        inferred = None

    if configured > 0 and inferred is not None and configured != inferred:
        raise ValueError(
            f"Configured train_size={configured} does not match loaded train split "
            f"size={inferred}. Fix the config or the dataset split."
        )

    train_size = configured if configured > 0 else inferred
    if train_size is None or train_size <= 0:
        raise ValueError(
            "Unable to determine train_size automatically. "
            "Provide train.train_size in the config for this environment."
        )
    return int(train_size)


def _compute_task_type_buckets(results: list[dict], task_types: list[str]) -> dict[str, dict]:
    """Compute per-task-type success rates."""
    buckets: dict[str, dict] = {}
    for task in task_types + ["overall"]:
        buckets[task] = {"total": 0, "hard": 0, "soft": 0.0}
    for r in results:
        tt = r.get("task_type", "other")
        for key in [tt, "overall"]:
            if key not in buckets:
                buckets[key] = {"total": 0, "hard": 0, "soft": 0.0}
            buckets[key]["total"] += 1
            buckets[key]["hard"] += float(r.get("hard", 0))
            buckets[key]["soft"] += float(r.get("soft", 0.0))
    return buckets


def _same_task_group_stats(results: list[dict]) -> dict[str, int]:
    """Summarize rollout outcomes at source-task rather than sample level."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        group_id = str(row.get("rollout_group_id") or row.get("id", ""))
        groups[group_id].append(row)

    stats = {
        "task_group_count": len(groups),
        "stable_success_count": 0,
        "mixed_count": 0,
        "stable_failure_count": 0,
    }
    for members in groups.values():
        success_count = sum(bool(row.get("hard", 0)) for row in members)
        if success_count == len(members):
            stats["stable_success_count"] += 1
        elif success_count == 0:
            stats["stable_failure_count"] += 1
        else:
            stats["mixed_count"] += 1
    return stats


def _format_rejection_buffer(buffer: list[dict]) -> str:
    """**DEPRECATED** — kept for backward compat; use _format_step_buffer."""
    return _format_step_buffer(buffer)


def _extract_failure_patterns(
    rollout_results: list[dict],
    step_dir: str,
) -> list[dict]:
    """Extract compact failure patterns from rollout results.

    Uses analyst ``failure_summary`` from minibatch patches when available,
    otherwise falls back to ``fail_reason`` prefix grouping.
    """
    failures = [r for r in rollout_results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9]
    if not failures:
        return []

    # Group by fail_reason prefix. Repeated samples from one source task count
    # as one task-level failure pattern, not independent support.
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in failures:
        reason = r.get("fail_reason", "unknown")
        prefix = reason.split(":")[0].strip() if ":" in reason else reason
        groups[prefix].append(r)

    # Try richer descriptions from analyst patches
    analyst_descs: list[str] = []
    patch_globs = [
        os.path.join(step_dir, "patches", "minibatch_fail_*.json"),
        os.path.join(step_dir, "batch_*", "patches", "minibatch_fail_*.json"),
    ]
    seen_patch_files: set[str] = set()
    for pattern in patch_globs:
        for fname in sorted(glob.glob(pattern)):
            if fname in seen_patch_files:
                continue
            seen_patch_files.add(fname)
            try:
                with open(fname) as f:
                    patch = json.load(f)
                for fs in patch.get("failure_summary", []):
                    ft = fs.get("failure_type", "")
                    sd = fs.get("description", "")
                    analyst_descs.append(f"{ft}: {sd}" if sd else ft)
            except Exception:
                pass

    patterns = []
    desc_iter = iter(analyst_descs)
    for prefix, items in groups.items():
        desc = next(desc_iter, None) or prefix
        task_ids = sorted({
            str(r.get("rollout_group_id") or r.get("id", "?"))
            for r in items
        })
        patterns.append({
            "pattern": desc,
            "count": len(task_ids),
            "task_ids": task_ids,
        })
    return patterns


def _format_step_buffer(buffer: list[dict]) -> str:
    """Format the unified step buffer into a single context block.

    Each entry captures what happened at a previous step: failure patterns
    observed during rollout, and — when the step was rejected — the specific
    edits that were tried and the resulting score drop.

    Returns empty string when *buffer* is empty.
    """
    if not buffer:
        return ""

    parts = [
        "Below is a summary of previous steps in this epoch. "
        "Use it to avoid repeating ineffective edits and to prioritise "
        "failure patterns that remain unsolved.\n"
    ]

    for entry in buffer:
        step = entry["step"]
        action = entry["action"]
        n_fail = entry.get("n_fail", 0)
        n_total = entry.get("n_total", "?")

        parts.append(f"### Step {step} — {action.upper()} ({n_fail}/{n_total} failed)")

        # Failure patterns
        for p in entry.get("failure_patterns", []):
            ids = ", ".join(p["task_ids"])
            parts.append(f'  - "{p["pattern"]}" (×{p["count"]}, tasks: {ids})')

        # Rejected edits (only present on reject)
        rejected = entry.get("rejected_edits", [])
        if rejected:
            score_before = entry.get("score_before", "?")
            score_after = entry.get("score_after", "?")
            parts.append(
                f"  Rejected edits (score {score_before} → {score_after}):"
            )
            for i, e in enumerate(rejected, 1):
                if e.get("op") is not None:
                    op = e.get("op", "?")
                    content = e.get("content", "")
                    target = e.get("target", "")
                    if target:
                        parts.append(f'    {i}. [{op}] target="{target}" → "{content}"')
                    else:
                        parts.append(f'    {i}. [{op}] "{content}"')
                else:
                    kind = e.get("type", "?")
                    title = e.get("title", "")
                    instruction = e.get("instruction", "")
                    parts.append(f'    {i}. [{kind}] "{title}" → "{instruction}"')

    return "\n".join(parts)


# ── Trainer ──────────────────────────────────────────────────────────────────

class ReflACTTrainer:
    """Main ReflACT training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. See ``configs/alfworld_default.yaml``
        for the full list of keys.
    adapter : EnvAdapter
        Environment adapter instance.
    """

    def __init__(self, cfg: dict, adapter: EnvAdapter) -> None:
        self.cfg = cfg
        self.adapter = adapter

    def train(self) -> dict:
        """Execute the full ReflACT training loop. Returns summary dict."""
        cfg = self.cfg
        cfg.setdefault("use_cross_group_evidence", False)
        validate_cross_group_config(cfg)
        use_cross_group_evidence = cfg["use_cross_group_evidence"]
        configure_cross_group_evidence(use_cross_group_evidence)
        if use_cross_group_evidence:
            print("  [cross-group evidence] enabled: training-only scope refinement; "
                  "K, edit budget and validation unchanged")
        adapter = self.adapter
        out_root = cfg["out_root"]
        os.makedirs(out_root, exist_ok=True)

        # ── Adapter setup (one-time init) ────────────────────────────
        adapter.setup(cfg)
        dataloader = adapter.get_dataloader()

        def _build_train_env(batch: BatchSpec):
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size, batch.seed

        def _build_eval_env(split: str, env_num: int, seed: int):
            if dataloader is None:
                env_manager = adapter.build_eval_env(
                    env_num=env_num,
                    split=split,
                    seed=seed,
                    out_root=out_root,
                )
                actual_n = len(env_manager) if hasattr(env_manager, "__len__") else env_num
                return env_manager, actual_n

            batch = dataloader.build_eval_batch(
                env_num=env_num,
                split=split,
                seed=seed,
                out_root=out_root,
            )
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size

        # ── Configure models ─────────────────────────────────────────────
        backend = cfg.get("model_backend", "azure_openai")
        configure_azure_openai(
            endpoint=(
                cfg.get("azure_openai_endpoint")
                or cfg.get("azure_endpoint")
                or None
            ),
            api_version=(
                cfg.get("azure_openai_api_version")
                or cfg.get("azure_api_version")
                or None
            ),
            api_key=(
                cfg.get("azure_openai_api_key")
                or cfg.get("azure_api_key")
                or None
            ),
            auth_mode=cfg.get("azure_openai_auth_mode") or None,
            ad_scope=cfg.get("azure_openai_ad_scope") or None,
            managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
            optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
            optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
            optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
            optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
            optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
            optimizer_managed_identity_client_id=(
                cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
            ),
            target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
            target_api_version=cfg.get("target_azure_openai_api_version") or None,
            target_api_key=cfg.get("target_azure_openai_api_key") or None,
            target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
            target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
            target_managed_identity_client_id=(
                cfg.get("target_azure_openai_managed_identity_client_id") or None
            ),
        )
        optimizer_backend = cfg.get("optimizer_backend")
        target_backend = cfg.get("target_backend")
        if not optimizer_backend or not target_backend:
            if backend in {"claude", "claude_chat"}:
                optimizer_backend = optimizer_backend or "claude_chat"
                target_backend = target_backend or "claude_chat"
            elif backend in {"codex", "codex_exec"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "codex_exec"
            elif backend == "claude_code_exec":
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "claude_code_exec"
            elif backend in {"qwen", "qwen_chat"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "qwen_chat"
            else:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "openai_chat"
            cfg["optimizer_backend"] = optimizer_backend
            cfg["target_backend"] = target_backend
        set_optimizer_backend(optimizer_backend)
        set_target_backend(target_backend)
        set_optimizer_deployment(cfg["optimizer_model"])
        set_target_deployment(cfg["target_model"])
        configure_codex_exec(
            path=cfg.get("codex_exec_path", "codex"),
            sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
            profile=cfg.get("codex_exec_profile", ""),
            full_auto=cfg.get("codex_exec_full_auto", False),
            reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
            use_sdk=cfg.get("codex_exec_use_sdk", None),
            network_access=cfg.get("codex_exec_network_access", False),
            web_search=cfg.get("codex_exec_web_search", False),
            approval_policy=cfg.get("codex_exec_approval_policy", "never"),
        )
        configure_claude_code_exec(
            path=cfg.get("claude_code_exec_path", "claude"),
            profile=cfg.get("claude_code_exec_profile", ""),
            use_sdk=cfg.get("claude_code_exec_use_sdk", None),
            effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
            max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
        )
        configure_qwen_chat(
            base_url=cfg.get("qwen_chat_base_url") or None,
            api_key=cfg.get("qwen_chat_api_key") or None,
            temperature=cfg.get("qwen_chat_temperature"),
            timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
            max_tokens=cfg.get("qwen_chat_max_tokens"),
            enable_thinking=cfg.get("qwen_chat_enable_thinking"),
            optimizer_base_url=cfg.get("optimizer_qwen_chat_base_url") or None,
            optimizer_api_key=cfg.get("optimizer_qwen_chat_api_key") or None,
            optimizer_temperature=cfg.get("optimizer_qwen_chat_temperature"),
            optimizer_timeout_seconds=cfg.get("optimizer_qwen_chat_timeout_seconds"),
            optimizer_max_tokens=cfg.get("optimizer_qwen_chat_max_tokens"),
            optimizer_enable_thinking=cfg.get("optimizer_qwen_chat_enable_thinking"),
            target_base_url=cfg.get("target_qwen_chat_base_url") or None,
            target_api_key=cfg.get("target_qwen_chat_api_key") or None,
            target_temperature=cfg.get("target_qwen_chat_temperature"),
            target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
            target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
            target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
        )
        configure_minimax_chat(
            base_url=cfg.get("minimax_base_url") or None,
            api_key=cfg.get("minimax_api_key") or None,
            temperature=cfg.get("minimax_temperature"),
            max_tokens=cfg.get("minimax_max_tokens"),
            enable_thinking=cfg.get("minimax_enable_thinking"),
        )
        minimax_model_cfg = cfg.get("minimax_model")
        if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
            set_target_deployment(str(minimax_model_cfg))
        os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] = (
            "1"
            if target_backend == "codex_exec" and cfg.get("codex_trace_to_optimizer", False)
            else "0"
        )
        reasoning = cfg.get("reasoning_effort", "") or None
        set_reasoning_effort(reasoning)
        print(
            f"  [model config] backend={backend}  "
            f"optimizer={cfg['optimizer_model']} ({optimizer_backend})  "
            f"target={cfg['target_model']} ({target_backend})  "
            f"reasoning={reasoning or 'off'}"
        )

        # ── Initialize Ray ───────────────────────────────────────────────
        if adapter.requires_ray():
            try:
                import ray
            except ImportError as e:
                raise ImportError(
                    "This environment requires ray, but ray is not installed."
                ) from e

            if not ray.is_initialized():
                ray.init(num_gpus=0)

        # ── Load initial skill ───────────────────────────────────────────
        skill_init_path = os.path.abspath(cfg["skill_init"])
        if os.path.exists(skill_init_path):
            with open(skill_init_path) as f:
                skill_init = f.read()
            print(f"  [initial skill] {skill_init_path} ({len(skill_init)} chars)")
        else:
            skill_init = ""
            print("  [initial skill] no initial skill file — starting from blank")

        # ── Training parameters ──────────────────────────────────────────
        batch_size = cfg["batch_size"]
        num_epochs = cfg["num_epochs"]
        same_task_rollouts = int(cfg.get("same_task_rollouts", 1) or 1)
        use_group_relative_edit_credit = bool(
            cfg.get("use_group_relative_edit_credit", False)
        )
        group_relative_candidate_competition = bool(
            cfg.get("group_relative_candidate_competition", False)
        )
        group_relative_similarity_threshold = float(
            cfg.get("group_relative_semantic_similarity_threshold", 0.45)
        )
        group_relative_direct_evidence_similarity_threshold = float(
            cfg.get(
                "group_relative_direct_evidence_similarity_threshold",
                0.65,
            )
        )
        group_relative_min_atomic_support_fraction = float(
            cfg.get("group_relative_min_atomic_support_fraction", 1.0)
        )
        group_relative_credit_aggregation = str(
            cfg.get("group_relative_credit_aggregation", "weakest")
        ).strip().lower()
        group_relative_model_semantic_matching = bool(
            cfg.get("group_relative_model_semantic_matching", False)
        )
        group_relative_atomize_append_edits = bool(
            cfg.get("group_relative_atomize_append_edits", False)
        )
        group_relative_atomize_all_edit_clauses = bool(
            cfg.get("group_relative_atomize_all_edit_clauses", False)
        )
        group_relative_specificity_penalty_weight = float(
            cfg.get("group_relative_specificity_penalty_weight", 0.75)
        )
        group_relative_max_specific_singletons = int(
            cfg.get("group_relative_max_specific_singletons", 1)
        )
        group_relative_min_cross_task_support = int(
            cfg.get("group_relative_min_cross_task_support", 2)
        )
        group_relative_max_exploratory = int(
            cfg.get("group_relative_max_exploratory", 1)
        )
        group_relative_min_candidate_cross_task_edits = int(
            cfg.get("group_relative_min_candidate_cross_task_edits", 0)
        )
        group_relative_soft_cross_task_target = bool(
            cfg.get("group_relative_soft_cross_task_target", False)
        )
        group_relative_verified_only_candidate = bool(
            cfg.get("group_relative_verified_only_candidate", False)
        )
        group_relative_max_contrastive_edits = int(
            cfg.get("group_relative_max_contrastive_edits", -1)
        )
        group_relative_max_singleton_edits = int(
            cfg.get("group_relative_max_singleton_edits", -1)
        )
        group_relative_block_singleton_stable_hypotheses = bool(
            cfg.get(
                "group_relative_block_singleton_stable_hypotheses",
                False,
            )
        )
        group_relative_actionable_priority_bonus = float(
            cfg.get("group_relative_actionable_priority_bonus", 0.0)
        )
        group_relative_causal_trace_priority_bonus = float(
            cfg.get("group_relative_causal_trace_priority_bonus", 0.0)
        )
        group_relative_contrastive_max_groups = int(
            cfg.get("group_relative_contrastive_max_groups", 8)
        )
        group_relative_taskwise_reflection = bool(
            cfg.get("group_relative_taskwise_reflection", False)
        )
        group_relative_taskwise_edit_budget = int(
            cfg.get("group_relative_taskwise_edit_budget", 1)
        )
        group_relative_outcome_stratified_reflection = bool(
            cfg.get("group_relative_outcome_stratified_reflection", False)
        )
        group_relative_min_credit_score = float(
            cfg.get("group_relative_min_credit_score", 0.0)
        )
        group_relative_require_actionable_units = bool(
            cfg.get("group_relative_require_actionable_units", True)
        )
        group_relative_require_causal_trace = bool(
            cfg.get("group_relative_require_causal_trace", True)
        )
        group_relative_output_contract_direct_causal = bool(
            cfg.get(
                "group_relative_output_contract_require_direct_causal_evidence",
                False,
            )
        )
        group_relative_probe_enabled = bool(
            cfg.get("group_relative_probe_enabled", False)
        )
        group_relative_probe_max_tasks_per_edit = int(
            cfg.get("group_relative_probe_max_tasks_per_edit", 4)
        )
        group_relative_probe_rollouts = int(
            cfg.get("group_relative_probe_rollouts", 1)
        )
        group_relative_probe_fail_open = bool(
            cfg.get("group_relative_probe_fail_open", False)
        )
        accumulation = cfg["accumulation"]
        seed = cfg["seed"]
        merge_bs = cfg["merge_batch_size"]
        max_analyst_rounds = int(cfg.get("max_analyst_rounds", 3) or 3)
        update_mode = normalize_update_mode(cfg.get("skill_update_mode", "patch"))
        lr_control_mode = _normalise_lr_control_mode(cfg.get("lr_control_mode", "fixed"))
        if is_full_rewrite_minibatch_mode(update_mode):
            lr_control_mode = "none"
        longitudinal_pair_policy = _normalise_longitudinal_pair_policy(
            cfg.get("longitudinal_pair_policy", "mixed")
        )
        rewrite_reasoning_effort = cfg.get("rewrite_reasoning_effort", "high")
        if rewrite_reasoning_effort == "":
            rewrite_reasoning_effort = None
        rewrite_max_completion_tokens = int(cfg.get("rewrite_max_completion_tokens", 64000))
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if same_task_rollouts <= 0:
            raise ValueError(
                "same_task_rollouts must be positive, got "
                f"{same_task_rollouts}"
            )
        if group_relative_probe_enabled and not use_group_relative_edit_credit:
            raise ValueError(
                "group_relative_probe_enabled requires "
                "use_group_relative_edit_credit=true"
            )
        if (
            group_relative_candidate_competition
            and not use_group_relative_edit_credit
        ):
            raise ValueError(
                "group_relative_candidate_competition requires "
                "use_group_relative_edit_credit=true"
            )
        if group_relative_candidate_competition and update_mode != "patch":
            raise ValueError(
                "group_relative_candidate_competition currently requires "
                "skill_update_mode=patch"
            )
        if (
            group_relative_verified_only_candidate
            and not use_group_relative_edit_credit
        ):
            raise ValueError(
                "group_relative_verified_only_candidate requires "
                "use_group_relative_edit_credit=true"
            )
        if group_relative_credit_aggregation not in {"weakest", "mean"}:
            raise ValueError(
                "group_relative_credit_aggregation must be weakest or mean"
            )
        if group_relative_min_candidate_cross_task_edits < 0:
            raise ValueError(
                "group_relative_min_candidate_cross_task_edits must be "
                "non-negative"
            )
        if group_relative_max_contrastive_edits < -1:
            raise ValueError(
                "group_relative_max_contrastive_edits must be -1 or "
                "non-negative"
            )
        if group_relative_max_singleton_edits < -1:
            raise ValueError(
                "group_relative_max_singleton_edits must be -1 or "
                "non-negative"
            )
        if (
            group_relative_actionable_priority_bonus < 0.0
            or group_relative_causal_trace_priority_bonus < 0.0
        ):
            raise ValueError(
                "group-relative priority bonuses must be non-negative"
            )
        if group_relative_probe_rollouts <= 0:
            raise ValueError("group_relative_probe_rollouts must be positive")
        if accumulation <= 0:
            raise ValueError(f"accumulation must be positive, got {accumulation}")

        train_size = _resolve_train_size(cfg, dataloader)
        steps_per_epoch = math.ceil(train_size / (batch_size * accumulation))
        batches_per_epoch = steps_per_epoch * accumulation
        total_steps = num_epochs * steps_per_epoch

        # Persist resolved derived fields so config.json / summary.json match
        # the actual runtime recipe.
        cfg["train_size"] = train_size
        cfg["steps_per_epoch"] = steps_per_epoch
        cfg["batches_per_epoch"] = batches_per_epoch
        cfg["samples_per_epoch"] = train_size
        cfg["same_task_rollouts"] = same_task_rollouts
        cfg["train_rollout_executions_per_epoch"] = (
            train_size * same_task_rollouts
        )
        cfg["skill_update_mode"] = update_mode
        cfg["lr_control_mode"] = lr_control_mode

        # Save config after deriving runtime values.
        with open(os.path.join(out_root, "config.json"), "w") as f:
            json.dump(_redact_cfg(cfg), f, indent=2, ensure_ascii=False)

        train_pool_size = train_size

        scheduler = build_scheduler(
            mode=cfg.get("lr_scheduler", "constant"),
            max_lr=cfg["edit_budget"],
            min_lr=cfg.get("min_edit_budget", 2),
            total_steps=total_steps,
        )

        # Fixed training pool: base seeds (each seed = one deterministic batch)
        if dataloader is not None:
            base_seeds = dataloader.make_base_seeds(
                steps_per_epoch=steps_per_epoch,
                accumulation=accumulation,
                seed=seed,
            )
        else:
            base_seeds = [seed + i + 1 for i in range(batches_per_epoch)]

        print(f"\n  [config] epochs={num_epochs} steps/epoch={steps_per_epoch} "
              f"(auto) accum={accumulation} batch_size={batch_size}")
        print(f"  [config] train_size={train_size}")
        print(f"  [config] batches/epoch={batches_per_epoch} "
              f"total_steps={total_steps} "
              f"unique_tasks/epoch={train_pool_size} "
              f"same_task_rollouts={same_task_rollouts} "
              f"train_executions/epoch={train_pool_size * same_task_rollouts}")
        print(f"  [config] lr_scheduler={cfg.get('lr_scheduler', 'constant')} "
              f"edit_budget={cfg['edit_budget']} "
              f"min_edit_budget={cfg.get('min_edit_budget', 2)}")
        print(f"  [config] skill_update_mode={update_mode} "
              f"lr_control_mode={lr_control_mode} "
              f"rewrite_reasoning_effort={rewrite_reasoning_effort or 'off'} "
              f"rewrite_max_completion_tokens={rewrite_max_completion_tokens} "
              f"max_analyst_rounds={max_analyst_rounds}")
        print(f"  [config] longitudinal_pair_policy={longitudinal_pair_policy}")
        print(f"  [config] base_seeds={base_seeds}")

        # ── Resume check ─────────────────────────────────────────────────
        history = _load_history(out_root)
        runtime_state = _load_runtime_state(out_root)
        if runtime_state:
            last_step = int(runtime_state.get("last_completed_step", 0) or 0)
            current_skill_path = runtime_state.get("current_skill_path") or os.path.join(
                out_root, "skills", f"skill_v{last_step:04d}.md",
            )
            with open(current_skill_path) as f:
                current_skill = f.read()
            best_skill_path = runtime_state.get("best_skill_path") or os.path.join(
                out_root, "best_skill.md",
            )
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = current_skill
            current_score = float(runtime_state.get("current_score", -1.0) or -1.0)
            best_score = float(runtime_state.get("best_score", current_score) or current_score)
            best_step = runtime_state.get("best_step", last_step)
            current_origin = str(
                runtime_state.get("current_origin")
                or (f"step_{last_step:04d}" if last_step > 0 else "initial_skill")
            )
            best_origin = str(runtime_state.get("best_origin") or current_origin)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f} "
                f"(origin={current_origin})"
            )
        elif history:
            last_step = history[-1]["step"]
            current_skill = _load_skill(out_root, last_step)
            best_rec = max(history, key=lambda h: h.get("best_score", 0.0))
            best_score = best_rec["best_score"]
            best_step = best_rec["best_step"]
            best_skill_path = os.path.join(out_root, "best_skill.md")
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = _load_skill(out_root, best_step)
            current_score = history[-1].get("current_score", best_score)
            current_origin = f"step_{last_step:04d}"
            best_origin = f"step_{int(best_step):04d}" if isinstance(best_step, int) else str(best_step)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f}"
            )
        else:
            current_skill = skill_init
            best_skill = skill_init
            best_score = -1.0
            current_score = -1.0
            best_step = 0
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            resume_from = 1

        _save_skill(out_root, 0, skill_init)

        use_skill_aware = cfg.get("use_skill_aware_reflection", False)
        # Publish the toggle process-wide so run_minibatch_reflect resolves it
        # from config for EVERY env adapter — no per-benchmark wiring needed.
        configure_skill_aware_reflection(
            use_skill_aware,
            cfg.get("skill_aware_appendix_source", "both"),
        )
        configure_group_relative_edit_credit(
            use_group_relative_edit_credit,
            semantic_similarity_threshold=(
                group_relative_similarity_threshold
            ),
            direct_evidence_similarity_threshold=(
                group_relative_direct_evidence_similarity_threshold
            ),
            min_atomic_support_fraction=(
                group_relative_min_atomic_support_fraction
            ),
            atomize_append_edits=group_relative_atomize_append_edits,
            atomize_all_edit_clauses=(
                group_relative_atomize_all_edit_clauses
            ),
            specificity_penalty_weight=(
                group_relative_specificity_penalty_weight
            ),
            max_specific_singletons=(
                group_relative_max_specific_singletons
            ),
            min_cross_task_support=(
                group_relative_min_cross_task_support
            ),
            max_exploratory=group_relative_max_exploratory,
            min_candidate_cross_task_edits=(
                group_relative_min_candidate_cross_task_edits
            ),
            soft_cross_task_target=group_relative_soft_cross_task_target,
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
            contrastive_max_groups=(
                group_relative_contrastive_max_groups
            ),
            taskwise_reflection=group_relative_taskwise_reflection,
            taskwise_edit_budget=group_relative_taskwise_edit_budget,
            outcome_stratified_reflection=(
                group_relative_outcome_stratified_reflection
            ),
            min_credit_score=group_relative_min_credit_score,
            require_actionable_units=(
                group_relative_require_actionable_units
            ),
            require_causal_trace=group_relative_require_causal_trace,
            output_contract_require_direct_causal_evidence=(
                group_relative_output_contract_direct_causal
            ),
            credit_aggregation=group_relative_credit_aggregation,
            model_semantic_matching=(
                group_relative_model_semantic_matching
            ),
        )

        def _set_group_relative_runtime(enabled: bool) -> None:
            configure_group_relative_edit_credit(
                enabled,
                semantic_similarity_threshold=(
                    group_relative_similarity_threshold
                ),
                direct_evidence_similarity_threshold=(
                    group_relative_direct_evidence_similarity_threshold
                ),
                min_atomic_support_fraction=(
                    group_relative_min_atomic_support_fraction
                ),
                atomize_append_edits=group_relative_atomize_append_edits,
                atomize_all_edit_clauses=(
                    group_relative_atomize_all_edit_clauses
                ),
                specificity_penalty_weight=(
                    group_relative_specificity_penalty_weight
                ),
                max_specific_singletons=(
                    group_relative_max_specific_singletons
                ),
                min_cross_task_support=(
                    group_relative_min_cross_task_support
                ),
                max_exploratory=group_relative_max_exploratory,
                min_candidate_cross_task_edits=(
                    group_relative_min_candidate_cross_task_edits
                ),
                soft_cross_task_target=(
                    group_relative_soft_cross_task_target
                ),
                max_contrastive_edits=(
                    group_relative_max_contrastive_edits
                ),
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
                contrastive_max_groups=(
                    group_relative_contrastive_max_groups
                ),
                taskwise_reflection=group_relative_taskwise_reflection,
                taskwise_edit_budget=group_relative_taskwise_edit_budget,
                outcome_stratified_reflection=(
                    group_relative_outcome_stratified_reflection
                ),
                min_credit_score=group_relative_min_credit_score,
                require_actionable_units=(
                    group_relative_require_actionable_units
                ),
                require_causal_trace=group_relative_require_causal_trace,
                output_contract_require_direct_causal_evidence=(
                    group_relative_output_contract_direct_causal
                ),
                credit_aggregation=group_relative_credit_aggregation,
                model_semantic_matching=(
                    group_relative_model_semantic_matching
                ),
            )
        if (
            group_relative_outcome_stratified_reflection
            and not use_group_relative_edit_credit
        ):
            print(
                "  [outcome-stratified reflection] standalone proposal mode: "
                "mixed tasks are analyzed locally; homogeneous tasks are "
                "deduplicated and pooled across tasks; downstream clean "
                "hard-gate aggregation and validation are unchanged"
            )
        if use_group_relative_edit_credit:
            print(
                "  [group-relative edit credit] enabled: "
                f"K={same_task_rollouts}, "
                "atomic task provenance, within-task divergence, "
                "adaptive evidence selection; validation unchanged"
            )
            print(
                "  [group-relative edit credit] "
                f"cross_task_min={group_relative_min_cross_task_support}, "
                f"max_exploratory={group_relative_max_exploratory}, "
                f"contrastive_groups={group_relative_contrastive_max_groups}, "
                f"taskwise={group_relative_taskwise_reflection}, "
                f"taskwise_L={group_relative_taskwise_edit_budget}, "
                "outcome_stratified="
                f"{group_relative_outcome_stratified_reflection}, "
                f"credit_floor={group_relative_min_credit_score:.2f}, "
                f"train_probe={group_relative_probe_enabled}, "
                f"candidate_competition={group_relative_candidate_competition}, "
                "direct_evidence_similarity="
                f"{group_relative_direct_evidence_similarity_threshold:.2f}, "
                "min_atomic_support="
                f"{group_relative_min_atomic_support_fraction:.2f}, "
                f"actionable={group_relative_require_actionable_units}, "
                f"causal_trace={group_relative_require_causal_trace}"
            )
            print(
                "  [group-relative evidence-atomic V2.1] "
                f"soft_cross_target={group_relative_soft_cross_task_target}, "
                f"verified_only={group_relative_verified_only_candidate}, "
                f"atomize_all={group_relative_atomize_all_edit_clauses}, "
                f"credit_aggregation={group_relative_credit_aggregation}, "
                "semantic_matcher="
                f"{'optimizer_model' if group_relative_model_semantic_matching else 'deterministic'}, "
                "output_contract_direct_causal="
                f"{group_relative_output_contract_direct_causal}"
            )
            print(
                "  [group-relative candidate composition] "
                "cross_task_edits_min="
                f"{group_relative_min_candidate_cross_task_edits}, "
                "contrastive_edits_max="
                f"{group_relative_max_contrastive_edits}, "
                "singleton_edits_max="
                f"{group_relative_max_singleton_edits}, "
                "block_singleton_stable="
                f"{group_relative_block_singleton_stable_hypotheses}, "
                "priority_bonus(actionable,causal)="
                f"({group_relative_actionable_priority_bonus:.2f},"
                f"{group_relative_causal_trace_priority_bonus:.2f})"
            )
        if use_skill_aware:
            current_skill = inject_empty_appendix_field(current_skill)

        def _persist_runtime_state(last_completed_step: int) -> None:
            _save_runtime_state(
                out_root,
                {
                    "last_completed_step": last_completed_step,
                    "current_skill_path": os.path.join(
                        out_root, "skills", f"skill_v{last_completed_step:04d}.md",
                    ),
                    "current_score": current_score,
                    "current_origin": current_origin,
                    "best_skill_path": os.path.join(out_root, "best_skill.md"),
                    "best_score": best_score,
                    "best_step": best_step,
                    "best_origin": best_origin,
                },
            )

        # ── Selection cache ──────────────────────────────────────────────
        sel_cache: dict[str, tuple[float, float]] = {}
        sel_result_cache: dict[str, list[dict]] = {}
        current_selection_results: list[dict] | None = None
        best_selection_results: list[dict] | None = None
        for rec in history:
            sh = rec.get("candidate_hash", "")
            if sh and rec.get("selection_hard") is not None:
                sel_cache[sh] = (rec["selection_hard"], rec["selection_soft"])

        # ── Baseline evaluation on selection set ─────────────────────────
        # `use_gate=False` keeps validation running (selection rollout +
        # scoring are unconditional below) but force-accepts every candidate
        # instead of gating it; final skill is chosen manually afterwards.
        use_gate = cfg.get("use_gate", True) is not False
        paired_non_degrading = bool(cfg.get("paired_non_degrading", False))
        paired_risk_sensitive = bool(
            cfg.get("paired_risk_sensitive", False)
        )
        paired_harm_weight = float(cfg.get("paired_harm_weight", 2.0))
        paired_risk_min_margin = float(
            cfg.get("paired_risk_min_margin", 0.0)
        )
        paired_audit_enabled = bool(
            cfg.get("paired_audit", paired_non_degrading)
        ) or paired_risk_sensitive
        paired_require_hard_gain = bool(cfg.get("paired_require_hard_gain", True))
        quarantine_rejected_edits = bool(cfg.get("quarantine_rejected_edits", False))
        reject_unobservable_runtime_edits = bool(
            cfg.get("reject_unobservable_runtime_edits", False)
        )
        quarantine_resample_attempts = int(
            cfg.get("quarantine_resample_attempts", 3) or 0
        )
        quarantine_bad_edit_threshold = int(
            cfg.get("quarantine_bad_edit_threshold", 3) or 0
        )
        quarantine_edit_similarity_threshold = float(
            cfg.get("quarantine_edit_similarity_threshold", 0.88)
        )
        quarantine_behavior_similarity_threshold = float(
            cfg.get("quarantine_behavior_similarity_threshold", 0.72)
        )
        if quarantine_resample_attempts < 0:
            raise ValueError(
                "optimizer.quarantine_resample_attempts must be non-negative, "
                f"got {quarantine_resample_attempts}"
            )
        if quarantine_bad_edit_threshold < 0:
            raise ValueError(
                "optimizer.quarantine_bad_edit_threshold must be non-negative, "
                f"got {quarantine_bad_edit_threshold}"
            )
        if not 0.0 <= quarantine_edit_similarity_threshold <= 1.0:
            raise ValueError(
                "optimizer.quarantine_edit_similarity_threshold must be in "
                f"[0, 1], got {quarantine_edit_similarity_threshold}"
            )
        if not 0.0 <= quarantine_behavior_similarity_threshold <= 1.0:
            raise ValueError(
                "optimizer.quarantine_behavior_similarity_threshold must be in "
                f"[0, 1], got {quarantine_behavior_similarity_threshold}"
            )
        if paired_harm_weight <= 0.0:
            raise ValueError(
                "evaluation.paired_harm_weight must be positive, "
                f"got {paired_harm_weight}"
            )
        quarantine_path = os.path.join(out_root, "quarantine.json")
        edit_history_path = os.path.join(out_root, "edit_history.jsonl")
        quarantine_records = (
            load_quarantine(quarantine_path) if quarantine_rejected_edits else []
        )
        edit_history = load_edit_history(edit_history_path)

        def _record_edit_event(items: list[dict], **fields) -> dict:
            event = build_edit_history_event(
                items,
                update_mode=update_mode,
                **fields,
            )
            saved = append_edit_history_event(edit_history_path, event)
            edit_history.append(saved)
            return saved

        if quarantine_rejected_edits:
            migrated_permanent_edits = promote_repeated_edits(
                quarantine_records,
                edit_history,
                threshold=quarantine_bad_edit_threshold,
                behavior_similarity_threshold=(
                    quarantine_behavior_similarity_threshold
                ),
            )
            if migrated_permanent_edits:
                save_quarantine(quarantine_path, quarantine_records)
        gate_metric = str(cfg.get("gate_metric", "hard")).strip().lower()
        if gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError(
                f"evaluation.gate_metric must be 'hard' | 'soft' | 'mixed', "
                f"got {gate_metric!r}"
            )
        gate_mixed_weight = float(cfg.get("gate_mixed_weight", 0.5))
        use_semantic_density = bool(cfg.get("use_semantic_density", False))
        semantic_density_weight = float(cfg.get("semantic_density_weight", 0.05))
        leading_words_raw = cfg.get("leading_words", None)
        leading_words = None
        if leading_words_raw is not None:
            if isinstance(leading_words_raw, str):
                leading_words = [w.strip() for w in leading_words_raw.split(",") if w.strip()]
            else:
                leading_words = list(leading_words_raw)
        if not 0.0 <= gate_mixed_weight <= 1.0:
            raise ValueError(
                f"evaluation.gate_mixed_weight must be in [0, 1], "
                f"got {gate_mixed_weight}"
            )
        print(
            f"  [gate] metric={gate_metric}"
            + (
                f" mixed_weight={gate_mixed_weight}"
                if gate_metric == "mixed"
                else ""
            )
            + ("" if use_gate
               else "  (DISABLED → validation runs, candidates force-accepted)")
        )
        if paired_audit_enabled:
            print(
                "  [paired audit] enabled: record per-item beneficial/harmful transitions"
            )
        if paired_non_degrading:
            print(
                "  [paired gate] strict non-degrading override enabled"
                + (" and require >=1 hard gain" if paired_require_hard_gain else "")
            )
        if paired_risk_sensitive:
            print(
                "  [paired risk gate] require beneficial - "
                f"{paired_harm_weight:g}*harmful > "
                f"{paired_risk_min_margin:g}"
            )
        if quarantine_rejected_edits:
            candidate_record_count = sum(
                record.get("scope") == CANDIDATE_SCOPE
                for record in quarantine_records
            )
            permanent_edit_count = sum(
                record.get("scope") == PERMANENT_EDIT_SCOPE
                for record in quarantine_records
            )
            print(
                f"  [quarantine] enabled: {candidate_record_count} prior candidate(s), "
                f"{permanent_edit_count} permanent bad edit(s), "
                f"resample_attempts={quarantine_resample_attempts}, "
                f"bad_edit_threshold={quarantine_bad_edit_threshold}, "
                f"candidate_similarity={quarantine_edit_similarity_threshold:.2f}, "
                "behavior_similarity="
                f"{quarantine_behavior_similarity_threshold:.2f}"
            )
        if reject_unobservable_runtime_edits:
            print(
                "  [runtime observability] enabled: optimizer edits may only use "
                "inference-time evidence"
            )
        print(
            f"  [edit history] {edit_history_path} "
            f"({len(edit_history)} prior event(s))"
        )
        slow_gate_with_selection = bool(
            cfg.get("slow_update_gate_with_selection", False)
        )
        print(
            "  [slow update] acceptance="
            + ("gated (selection-set validation)"
               if slow_gate_with_selection
               else "force-accept (unconditional)")
        )
        if current_score < 0:
            print(f"\n{'='*60}")
            print("  BASELINE — evaluate initial skill on Selection set (valid_seen)")
            print(f"{'='*60}")
            sel_env, sel_n = _build_eval_env(
                split="valid_seen",
                env_num=cfg["sel_env_num"],
                seed=seed,
            )
            print(f"  Selection items: {sel_n}")
            baseline_dir = os.path.join(out_root, "selection_eval_baseline")
            baseline_results = adapter.rollout(sel_env, skill_init, baseline_dir)
            baseline_hard, baseline_soft = compute_score(baseline_results)
            current_score = select_gate_score(
                baseline_hard, baseline_soft, gate_metric, gate_mixed_weight,
                skill_content=skill_init,
                use_semantic_density=use_semantic_density,
                semantic_density_weight=semantic_density_weight,
                leading_words=leading_words,
            )
            best_score = current_score
            sh = skill_hash(skill_init)
            sel_cache[sh] = (baseline_hard, baseline_soft)
            sel_result_cache[sh] = baseline_results
            current_selection_results = baseline_results
            best_selection_results = baseline_results
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            _persist_runtime_state(0)
            print(
                f"  [baseline result] selection hard={baseline_hard:.4f} "
                f"soft={baseline_soft:.4f} "
                f"gate[{gate_metric}]={current_score:.4f}"
            )

        # A resumed paired-gate run needs per-item outcomes for the active
        # skill, not only the aggregate score stored in runtime_state.json.
        if paired_audit_enabled and current_selection_results is None:
            print("  [paired gate] rebuilding current per-item selection baseline")
            sel_env, sel_n = _build_eval_env(
                split="valid_seen",
                env_num=cfg["sel_env_num"],
                seed=seed,
            )
            resume_eval_dir = os.path.join(out_root, "selection_eval_resume_current")
            current_selection_results = adapter.rollout(
                sel_env, current_skill, resume_eval_dir,
            )
            resume_hard, resume_soft = compute_score(current_selection_results)
            current_hash = skill_hash(current_skill)
            sel_cache[current_hash] = (resume_hard, resume_soft)
            sel_result_cache[current_hash] = current_selection_results
            current_score = select_gate_score(
                resume_hard,
                resume_soft,
                gate_metric,
                gate_mixed_weight,
                skill_content=current_skill,
                use_semantic_density=use_semantic_density,
                semantic_density_weight=semantic_density_weight,
                leading_words=leading_words,
            )
            if current_skill == best_skill:
                best_score = current_score
                best_selection_results = current_selection_results
            _persist_runtime_state(max(resume_from - 1, 0))
            print(
                f"  [paired gate] rebuilt {sel_n} items: "
                f"hard={resume_hard:.4f} soft={resume_soft:.4f}"
            )

        # ── Training loop ────────────────────────────────────────────────
        t_loop_start = time.time()

        if resume_from > total_steps:
            print(f"\n  [skip] all {total_steps} steps complete — jumping to evaluation")

        global_step = 0
        for epoch in range(1, num_epochs + 1):
            if dataloader is not None:
                epoch_batches = dataloader.plan_train_epoch(
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                    accumulation=accumulation,
                    batch_size=batch_size,
                    seed=seed,
                    out_root=out_root,
                )
                shuffled_seeds = [batch.seed for batch in epoch_batches]
            else:
                epoch_batches = []
                epoch_rng = random.Random(seed + epoch * 1000)
                shuffled_seeds = base_seeds.copy()
                epoch_rng.shuffle(shuffled_seeds)

            # Step buffer: accumulates per-step context (failure patterns +
            # rejected edits) within this epoch so optimizers see full history.
            step_buffer, missing_step_digests = _load_epoch_step_buffer(
                out_root,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                last_completed_step=resume_from - 1,
            )
            if step_buffer:
                print(
                    f"  [resume] rebuilt epoch {epoch} step buffer from "
                    f"{len(step_buffer)} completed step digest(s)"
                )
            if missing_step_digests:
                print(
                    "  [resume warning] missing trajectory digest(s) for "
                    + ", ".join(str(step) for step in missing_step_digests)
                )
            active_meta_skill = (
                _load_meta_skill_content(out_root, epoch - 1)
                if cfg.get("use_meta_skill", False)
                else ""
            )

            print(
                f"\n  [EPOCH {epoch}/{num_epochs}] "
                f"shuffled_seeds={shuffled_seeds}"
            )
            if active_meta_skill:
                print(
                    f"  [meta skill] loaded from epoch {epoch - 1} "
                    f"({len(active_meta_skill)} chars)"
                )

            for step_in_epoch in range(steps_per_epoch):
                # A diagnostic run stops only between fully persisted steps.
                # Keep the epoch plan and every in-step evaluation unchanged.
                stop_after_step = int(cfg.get("stop_after_step", 0) or 0)
                if stop_after_step > 0 and global_step >= stop_after_step:
                    stopped_summary = {
                        "status": "stopped_after_requested_step",
                        "requested_stop_after_step": stop_after_step,
                        "last_completed_step": global_step,
                        "completed_steps": len(history),
                        "planned_total_steps": total_steps,
                        "best_selection_hard": best_score,
                        "current_selection_hard": current_score,
                        "best_step": best_step,
                        "epoch_completed": step_in_epoch == 0,
                        "loop_wall_time_s": round(time.time() - t_loop_start, 1),
                        "token_summary": get_token_summary(),
                        "scope": "Step-limited diagnostic; no final unseen evaluation performed.",
                    }
                    with open(os.path.join(out_root, "step_limit_summary.json"), "w") as f:
                        json.dump(stopped_summary, f, indent=2, ensure_ascii=False)
                    print(f"\n  [step limit] stopped after completed step {global_step}", flush=True)
                    return stopped_summary
                global_step += 1
                if global_step < resume_from:
                    continue

                step_t0 = time.time()
                step_dir = os.path.join(out_root, "steps", f"step_{global_step:04d}")
                os.makedirs(step_dir, exist_ok=True)

                tokens_before = get_token_summary()

                print(
                    f"\n  [STEP {global_step}/{total_steps}] "
                    f"epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"{'='*30}"
                )

                step_rec: dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "timing": {},
                    "tokens": {},
                }
                permanent_edit_context = format_permanent_edit_context(
                    quarantine_records
                )
                runtime_observability_context = (
                    format_runtime_observability_context()
                    if reject_unobservable_runtime_edits
                    else ""
                )
                step_meta_skill_context = "\n\n".join(
                    part
                    for part in (
                        active_meta_skill,
                        permanent_edit_context,
                        runtime_observability_context,
                    )
                    if part.strip()
                )

                # ── Accumulation: Rollout + Reflect ──────────────────────
                all_failure_patches: list[dict] = []
                all_success_patches: list[dict] = []
                all_raw_patches: list[dict | None] = []
                group_relative_failure_patches: list[dict] = []
                group_relative_success_patches: list[dict] = []
                all_rollout_results: list[dict] = []
                all_train_items: list[dict] = []
                accum_rollout_stats: list[dict] = []
                total_rollout_time = 0.0
                total_reflect_time = 0.0

                for a in range(accumulation):
                    batch_idx = step_in_epoch * accumulation + a
                    if dataloader is not None:
                        batch_spec = epoch_batches[batch_idx]
                        for train_item in list(batch_spec.payload or []):
                            if isinstance(train_item, dict):
                                all_train_items.append(train_item)
                        train_env, train_n, batch_seed = _build_train_env(batch_spec)
                    else:
                        batch_seed = shuffled_seeds[batch_idx]
                        train_env = adapter.build_train_env(
                            batch_size=batch_size,
                            seed=batch_seed,
                            out_root=out_root,
                        )
                        train_n = len(train_env) if hasattr(train_env, "__len__") else batch_size

                    train_env = adapter.expand_same_task_rollouts(
                        train_env,
                        same_task_rollouts,
                    )
                    train_execution_n = (
                        len(train_env)
                        if hasattr(train_env, "__len__")
                        else train_n * same_task_rollouts
                    )

                    # Directory routing
                    if accumulation > 1:
                        batch_dir = os.path.join(step_dir, f"batch_{a}")
                    else:
                        batch_dir = step_dir

                    rollout_dir = os.path.join(batch_dir, "rollout")
                    patches_dir = os.path.join(batch_dir, "patches")

                    # ① ROLLOUT ────────────────────────────────────────────
                    t_phase = time.time()
                    print(
                        f"    [1/6 ROLLOUT] unique_train_items={train_n} "
                        f"same_task_K={same_task_rollouts} "
                        f"executions={train_execution_n} "
                        f"(from pool, batch_seed={batch_seed})"
                    )
                    rollout_results = adapter.rollout(
                        train_env, current_skill, rollout_dir,
                        use_eval_feedback=True,
                    )
                    r_hard, r_soft = compute_score(rollout_results)
                    total_rollout_time += time.time() - t_phase
                    all_rollout_results.extend(rollout_results)
                    print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")

                    # ② REFLECT ────────────────────────────────────────────
                    t_phase = time.time()
                    pred_dir = os.path.join(rollout_dir, "predictions")

                    # Build step context from buffer
                    step_buffer_context = _format_step_buffer(step_buffer)

                    if group_relative_candidate_competition:
                        if group_relative_verified_only_candidate:
                            raw_patches = []
                            print(
                                "    [candidate competition] verified-only "
                                "policy skips unverified standard reflection"
                            )
                        else:
                            _set_group_relative_runtime(False)
                            try:
                                raw_patches = adapter.reflect(
                                    rollout_results, current_skill, batch_dir,
                                    prediction_dir=pred_dir,
                                    patches_dir=os.path.join(
                                        patches_dir, "standard"
                                    ),
                                    random_seed=batch_seed,
                                    step_buffer_context=step_buffer_context,
                                    meta_skill_context=step_meta_skill_context,
                                )
                            finally:
                                _set_group_relative_runtime(True)
                        try:
                            group_raw = adapter.reflect(
                                rollout_results, current_skill, batch_dir,
                                prediction_dir=pred_dir,
                                patches_dir=os.path.join(
                                    patches_dir, "group_relative"
                                ),
                                random_seed=batch_seed,
                                step_buffer_context=step_buffer_context,
                                meta_skill_context=step_meta_skill_context,
                            )
                        except Exception as exc:  # noqa: BLE001
                            group_raw = []
                            print(
                                "    [candidate competition] auxiliary "
                                "group-relative reflection failed; standard "
                                f"branch continues: {exc}"
                            )
                        group_failure, group_success = _normalise_patches(
                            group_raw,
                            update_mode=update_mode,
                        )
                        group_relative_failure_patches.extend(group_failure)
                        group_relative_success_patches.extend(group_success)
                    else:
                        raw_patches = adapter.reflect(
                            rollout_results, current_skill, batch_dir,
                            prediction_dir=pred_dir, patches_dir=patches_dir,
                            random_seed=batch_seed,
                            step_buffer_context=step_buffer_context,
                            meta_skill_context=step_meta_skill_context,
                        )
                        group_failure = []
                        group_success = []
                    failure_patches, success_patches = _normalise_patches(
                        raw_patches,
                        update_mode=update_mode,
                    )
                    all_failure_patches.extend(failure_patches)
                    all_success_patches.extend(success_patches)
                    all_raw_patches.extend(raw_patches)
                    total_reflect_time += time.time() - t_phase

                    print(
                        f"    [2/6 done] failure_patches={len(failure_patches)} "
                        f"success_patches={len(success_patches)}"
                        + (
                            "; group_relative_failure="
                            f"{len(group_failure)} group_relative_success="
                            f"{len(group_success)}"
                            if group_relative_candidate_competition
                            else ""
                        )
                    )

                    # Track per-batch stats
                    accum_rollout_stats.append({
                        "batch_idx": a,
                        "batch_seed": batch_seed,
                        "n_envs": len(rollout_results),
                        "n_unique_tasks": train_n,
                        "same_task_rollouts": same_task_rollouts,
                        "hard": r_hard,
                        "soft": r_soft,
                        "n_failure_patches": len(failure_patches),
                        "n_success_patches": len(success_patches),
                        "n_group_relative_failure_patches": len(group_failure),
                        "n_group_relative_success_patches": len(group_success),
                    })

                # ── End of accumulation loop ─────────────────────────────

                # Aggregate rollout stats across batches
                total_n = sum(b["n_envs"] for b in accum_rollout_stats)
                agg_hard = sum(b["hard"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)
                agg_soft = sum(b["soft"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)

                step_rec["rollout_hard"] = round(agg_hard, 6)
                step_rec["rollout_soft"] = round(agg_soft, 6)
                step_rec["rollout_n"] = total_n
                step_rec["rollout_unique_task_n"] = sum(
                    b["n_unique_tasks"] for b in accum_rollout_stats
                )
                step_rec["same_task_rollouts"] = same_task_rollouts
                step_rec["same_task_group_stats"] = _same_task_group_stats(
                    all_rollout_results
                )
                step_rec["accumulation_batches"] = accum_rollout_stats
                step_rec["timing"]["rollout_s"] = round(total_rollout_time, 1)
                step_rec["timing"]["reflect_s"] = round(total_reflect_time, 1)

                n_total_patches = len(all_failure_patches) + len(all_success_patches)
                cross_group_merge_kwargs = {}
                if use_cross_group_evidence:
                    cross_group_pool = collect_cross_group_evidence(all_raw_patches)
                    cross_group_merge_kwargs["cross_group_evidence"] = cross_group_pool
                    with open(os.path.join(step_dir, "cross_group_evidence.json"), "w") as f:
                        json.dump(cross_group_pool, f, ensure_ascii=False, indent=2)
                    step_rec["cross_group_evidence"] = {
                        key: value for key, value in cross_group_pool.items() if key != "cards"
                    }
                step_rec["n_patches"] = n_total_patches
                step_rec["n_failure_patches"] = len(all_failure_patches)
                step_rec["n_success_patches"] = len(all_success_patches)
                step_rec["n_group_relative_failure_patches"] = len(
                    group_relative_failure_patches
                )
                step_rec["n_group_relative_success_patches"] = len(
                    group_relative_success_patches
                )

                if accumulation > 1:
                    print(
                        f"    [accum done] total: failure={len(all_failure_patches)} "
                        f"success={len(all_success_patches)} "
                        f"from {accumulation} batches"
                    )

                # ── No patches? Skip ─────────────────────────────────────
                if (
                    not all_failure_patches
                    and not all_success_patches
                    and not group_relative_failure_patches
                    and not group_relative_success_patches
                ):
                    # Skill-aware: a lapse-only step has no body patches but
                    # may still carry appendix notes — flush them BEFORE
                    # skipping, or they would be silently dropped.
                    if use_skill_aware:
                        current_skill = _flush_skill_aware_appendix(
                            current_skill, all_raw_patches, step_rec, step_dir, cfg,
                        )
                    step_rec["action"] = "skip_no_patches"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable patches — skill unchanged")
                    continue

                # ③ AGGREGATE ──────────────────────────────────────────────
                t_phase = time.time()
                if (
                    group_relative_verified_only_candidate
                    and group_relative_candidate_competition
                ):
                    merged_patch = {
                        "reasoning": (
                            "standard branch disabled by verified-only policy"
                        ),
                        "edits": [],
                    }
                else:
                    if group_relative_candidate_competition:
                        _set_group_relative_runtime(False)
                    try:
                        merged_patch = merge_patches(
                            current_skill,
                            all_failure_patches,
                            all_success_patches,
                            batch_size=merge_bs,
                            verbose=True,
                            workers=cfg["analyst_workers"],
                            update_mode=update_mode,
                            meta_skill_context=step_meta_skill_context,
                            **cross_group_merge_kwargs,
                        )
                    finally:
                        if group_relative_candidate_competition:
                            _set_group_relative_runtime(True)
                group_relative_credit_audit = None
                main_group_relative_credit = (
                    use_group_relative_edit_credit
                    and not group_relative_candidate_competition
                )
                if main_group_relative_credit:
                    merged_patch, group_relative_credit_audit = annotate_merged_patch(
                        merged_patch,
                        all_failure_patches + all_success_patches,
                        update_mode=update_mode,
                        semantic_similarity_threshold=(
                            group_relative_similarity_threshold
                        ),
                        direct_evidence_similarity_threshold=(
                            group_relative_direct_evidence_similarity_threshold
                        ),
                        specificity_penalty_weight=(
                            group_relative_specificity_penalty_weight
                        ),
                    )
                    with open(
                        os.path.join(step_dir, "group_relative_edit_credit.json"),
                        "w",
                    ) as f:
                        json.dump(
                            group_relative_credit_audit,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    step_rec["group_relative_edit_credit"] = {
                        "source_edit_count": group_relative_credit_audit[
                            "source_edit_count"
                        ],
                        "merged_edit_count": group_relative_credit_audit[
                            "merged_edit_count"
                        ],
                        "validation_changed": False,
                    }

                group_relative_merged_patch = None
                group_relative_branch_credit_audit = None
                if group_relative_candidate_competition and (
                    group_relative_failure_patches
                    or group_relative_success_patches
                ):
                    group_relative_merged_patch = merge_patches(
                        current_skill,
                        group_relative_failure_patches,
                        group_relative_success_patches,
                        batch_size=merge_bs,
                        verbose=True,
                        workers=cfg["analyst_workers"],
                        update_mode=update_mode,
                        meta_skill_context=step_meta_skill_context,
                    )
                    (
                        group_relative_merged_patch,
                        group_relative_branch_credit_audit,
                    ) = annotate_merged_patch(
                        group_relative_merged_patch,
                        group_relative_failure_patches
                        + group_relative_success_patches,
                        update_mode=update_mode,
                        semantic_similarity_threshold=(
                            group_relative_similarity_threshold
                        ),
                        direct_evidence_similarity_threshold=(
                            group_relative_direct_evidence_similarity_threshold
                        ),
                        specificity_penalty_weight=(
                            group_relative_specificity_penalty_weight
                        ),
                    )
                    with open(
                        os.path.join(
                            step_dir,
                            "group_relative_branch_merged_patch.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(
                            group_relative_merged_patch,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    with open(
                        os.path.join(
                            step_dir,
                            "group_relative_branch_credit.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(
                            group_relative_branch_credit_audit,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                with open(os.path.join(step_dir, "merged_patch.json"), "w") as f:
                    json.dump(merged_patch, f, ensure_ascii=False, indent=2)

                merged_items = get_payload_items(merged_patch, update_mode)
                n_edits_merged = len(merged_items)
                step_rec["n_edits_merged"] = n_edits_merged
                step_rec["timing"]["aggregate_s"] = round(time.time() - t_phase, 1)
                print(f"    [3/6 done] merged {n_edits_merged} {payload_label(update_mode)}")

                # ④ SELECT ─────────────────────────────────────────────────
                t_phase = time.time()
                lr_decision = None
                group_relative_ranked_patch = None
                if is_full_rewrite_minibatch_mode(update_mode):
                    edit_budget = None
                    ranked_patch = merged_patch
                    ranked_items = merged_items
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["edit_budget"] = None
                    step_rec["lr_control_mode"] = "none"
                    with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                else:
                    if lr_control_mode == "autonomous":
                        lr_decision = decide_autonomous_learning_rate(
                            skill_content=current_skill,
                            merged_patch=merged_patch,
                            update_mode=update_mode,
                            rollout_hard=agg_hard,
                            rollout_soft=agg_soft,
                            rollout_n=total_n,
                            step_buffer_context=step_buffer_context,
                            meta_skill_context=step_meta_skill_context,
                        )
                        edit_budget = int(lr_decision["learning_rate"])
                        with open(os.path.join(step_dir, "lr_decision.json"), "w") as f:
                            json.dump(lr_decision, f, ensure_ascii=False, indent=2)
                        with open(os.path.join(out_root, "lr_history.jsonl"), "a") as f:
                            f.write(json.dumps({
                                "step": global_step,
                                "epoch": epoch,
                                **lr_decision,
                            }, ensure_ascii=False) + "\n")
                    else:
                        edit_budget = scheduler.step()
                    ranked_patch = rank_and_select(
                        current_skill, merged_patch,
                        max_edits=edit_budget,
                        update_mode=update_mode,
                        meta_skill_context=step_meta_skill_context,
                        group_relative_edit_credit=(
                            main_group_relative_credit
                        ),
                        group_relative_max_specific_singletons=(
                            group_relative_max_specific_singletons
                        ),
                        group_relative_min_cross_task_support=(
                            group_relative_min_cross_task_support
                        ),
                        group_relative_max_exploratory=(
                            group_relative_max_exploratory
                        ),
                        group_relative_min_atomic_support_fraction=(
                            group_relative_min_atomic_support_fraction
                        ),
                        group_relative_min_candidate_cross_task_edits=(
                            group_relative_min_candidate_cross_task_edits
                        ),
                        group_relative_max_contrastive_edits=(
                            group_relative_max_contrastive_edits
                        ),
                        group_relative_max_singleton_edits=(
                            group_relative_max_singleton_edits
                        ),
                        group_relative_block_singleton_stable_hypotheses=(
                            group_relative_block_singleton_stable_hypotheses
                        ),
                        group_relative_actionable_priority_bonus=(
                            group_relative_actionable_priority_bonus
                        ),
                        group_relative_causal_trace_priority_bonus=(
                            group_relative_causal_trace_priority_bonus
                        ),
                    )
                    if group_relative_merged_patch is not None:
                        group_relative_ranked_patch = rank_and_select(
                            current_skill,
                            group_relative_merged_patch,
                            max_edits=edit_budget,
                            update_mode=update_mode,
                            meta_skill_context=step_meta_skill_context,
                            group_relative_edit_credit=True,
                            group_relative_max_specific_singletons=(
                                group_relative_max_specific_singletons
                            ),
                            group_relative_min_cross_task_support=(
                                group_relative_min_cross_task_support
                            ),
                            group_relative_max_exploratory=(
                                group_relative_max_exploratory
                            ),
                            group_relative_min_atomic_support_fraction=(
                                group_relative_min_atomic_support_fraction
                            ),
                            group_relative_min_candidate_cross_task_edits=(
                                group_relative_min_candidate_cross_task_edits
                            ),
                            group_relative_max_contrastive_edits=(
                                group_relative_max_contrastive_edits
                            ),
                            group_relative_max_singleton_edits=(
                                group_relative_max_singleton_edits
                            ),
                            group_relative_block_singleton_stable_hypotheses=(
                                group_relative_block_singleton_stable_hypotheses
                            ),
                            group_relative_actionable_priority_bonus=(
                                group_relative_actionable_priority_bonus
                            ),
                            group_relative_causal_trace_priority_bonus=(
                                group_relative_causal_trace_priority_bonus
                            ),
                        )
                        with open(
                            os.path.join(
                                step_dir,
                                "group_relative_branch_ranked_edits.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                group_relative_ranked_patch,
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)

                    ranked_items = get_payload_items(ranked_patch, update_mode)
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["edit_budget"] = edit_budget
                    step_rec["lr_control_mode"] = lr_control_mode
                    if lr_decision is not None:
                        step_rec["lr_decision"] = lr_decision
                    if main_group_relative_credit:
                        step_rec["group_relative_selection"] = ranked_patch.get(
                            "group_relative_selection", {}
                        )

                quarantine_collisions = []
                matched_quarantine = None
                blocked_permanent_edits = []
                permanent_edit_hits = {}
                unobservable_runtime_edits = []
                runtime_observability_hits = {}
                hard_observability_prune_events = []

                def _prune_hard_unobservable_items(
                    payload,
                    items,
                    *,
                    candidate_attempt,
                ):
                    if not reject_unobservable_runtime_edits or not items:
                        return items, []
                    observable, hits = partition_runtime_observable_edits(
                        items,
                        update_mode=update_mode,
                    )
                    if not hits:
                        return items, []
                    set_payload_items(payload, observable, update_mode)
                    rendered_hits = [
                        {
                            "behavior": hit.get("behavior", {}),
                            "semantic": hit.get("semantic", {}),
                            "matched_reasons": hit.get("matched_reasons", []),
                        }
                        for hit in hits
                    ]
                    hard_observability_prune_events.append(
                        {
                            "candidate_attempt": candidate_attempt,
                            "rejected": rendered_hits,
                            "remaining_edit_count": len(observable),
                        }
                    )
                    for hit in hits:
                        behavior_text = str(
                            hit.get("behavior", {}).get("text", "")
                        )
                        runtime_observability_hits[behavior_text] = hit
                    _record_edit_event(
                        [hit["item"] for hit in hits],
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        stage="pre_validation",
                        candidate_attempt=candidate_attempt,
                        validation_ran=False,
                        outcome="hard_unobservable_edit_pruned",
                        blocked_reasons=[
                            "hard_unobservable_runtime_dependency"
                        ],
                        runtime_observability_hits=rendered_hits,
                    )
                    print(
                        "    [pre-validation] pruned "
                        f"{len(hits)} hard-unobservable edit(s); "
                        f"{len(observable)} deployable edit(s) remain"
                    )
                    return observable, hits

                def _persist_hard_observability_report(items):
                    if not hard_observability_prune_events:
                        return
                    with open(
                        os.path.join(
                            step_dir,
                            "hard_runtime_observability_filter.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(
                            {
                                "scope": "hard_unobservable_edit",
                                "policy": "prune_per_edit_resample_only_if_empty",
                                "events": hard_observability_prune_events,
                                "final_candidate_items": items,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

                ranked_items, initial_hard_hits = _prune_hard_unobservable_items(
                    ranked_patch,
                    ranked_items,
                    candidate_attempt=0,
                )
                n_edits_ranked = len(ranked_items)
                step_rec["n_edits_ranked"] = n_edits_ranked
                if initial_hard_hits:
                    with open(
                        os.path.join(step_dir, "ranked_edits.json"), "w"
                    ) as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                if initial_hard_hits and not ranked_items:
                    unobservable_runtime_edits = initial_hard_hits
                step_rec["runtime_observability_hits"] = len(
                    runtime_observability_hits
                )
                step_rec["hard_unobservable_edits_pruned"] = sum(
                    len(event["rejected"])
                    for event in hard_observability_prune_events
                )
                _persist_hard_observability_report(ranked_items)

                if quarantine_rejected_edits and ranked_items:
                    matched_quarantine = find_quarantined_candidate(
                        ranked_items,
                        quarantine_records,
                        update_mode=update_mode,
                        similarity_threshold=quarantine_edit_similarity_threshold,
                    )
                    blocked_permanent_edits = (
                        find_permanently_quarantined_edits(
                            ranked_items,
                            quarantine_records,
                            update_mode=update_mode,
                            behavior_similarity_threshold=(
                                quarantine_behavior_similarity_threshold
                            ),
                        )
                    )

                candidate_primary_branch = (
                    "group_relative_single"
                    if main_group_relative_credit
                    else "standard"
                )
                group_relative_branch_items: list[dict] = []
                group_relative_branch_audit = {
                    "enabled": bool(group_relative_candidate_competition),
                    "eligible": False,
                    "rejection_reasons": [],
                }
                if (
                    matched_quarantine is not None
                    or blocked_permanent_edits
                    or unobservable_runtime_edits
                    or group_relative_ranked_patch is not None
                ):
                    resample_attempt = 0
                    while (
                        (
                            matched_quarantine is not None
                            or bool(blocked_permanent_edits)
                            or bool(unobservable_runtime_edits)
                        )
                        and resample_attempt < quarantine_resample_attempts
                    ):
                        resample_attempt += 1
                        if matched_quarantine is not None:
                            quarantine_collisions.append(matched_quarantine)
                        for hit in blocked_permanent_edits:
                            family_id = str(
                                hit.get("permanent_record", {}).get(
                                    "family_id", ""
                                )
                            )
                            permanent_edit_hits[family_id] = hit
                        for hit in unobservable_runtime_edits:
                            behavior_text = str(
                                hit.get("behavior", {}).get("text", "")
                            )
                            runtime_observability_hits[behavior_text] = hit
                        unique_collisions = {
                            str(record.get("fingerprint", "")): record
                            for record in quarantine_collisions
                        }
                        avoidance_context = format_candidate_resample_context(
                            list(unique_collisions.values())
                        )
                        observability_context = (
                            format_runtime_observability_context(
                                unobservable_runtime_edits
                            )
                            if reject_unobservable_runtime_edits
                            else ""
                        )
                        resample_meta_context = "\n\n".join(
                            part
                            for part in (
                                step_meta_skill_context,
                                avoidance_context,
                                observability_context,
                            )
                            if part.strip()
                        )
                        block_reasons = []
                        if matched_quarantine is not None:
                            block_reasons.append("exact rejected candidate")
                        if blocked_permanent_edits:
                            block_reasons.append(
                                f"{len(blocked_permanent_edits)} permanent bad edit(s)"
                            )
                        if unobservable_runtime_edits:
                            block_reasons.append(
                                f"{len(unobservable_runtime_edits)} runtime-unobservable edit(s)"
                            )
                        _record_edit_event(
                            ranked_items,
                            step=global_step,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            stage="pre_validation",
                            candidate_attempt=resample_attempt - 1,
                            validation_ran=False,
                            outcome="blocked_pre_validation",
                            blocked_reasons=block_reasons,
                            matched_candidate_fingerprint=(
                                matched_quarantine.get("fingerprint", "")
                                if matched_quarantine is not None
                                else ""
                            ),
                            permanent_edit_hits=[
                                {
                                    "family_id": hit.get(
                                        "permanent_record", {}
                                    ).get("family_id", ""),
                                    "similarity": hit.get("similarity", 0.0),
                                }
                                for hit in blocked_permanent_edits
                            ],
                            runtime_observability_hits=[
                                {
                                    "behavior": hit.get("behavior", {}),
                                    "matched_reasons": hit.get(
                                        "matched_reasons", []
                                    ),
                                }
                                for hit in unobservable_runtime_edits
                            ],
                        )
                        print(
                            "    [pre-validation] "
                            + " and ".join(block_reasons)
                            + "; "
                            f"resampling edits ({resample_attempt}/"
                            f"{quarantine_resample_attempts})"
                        )

                        # Re-run aggregation as well as selection so a new edit can
                        # enter the pool when every prior merged edit was selected.
                        if group_relative_candidate_competition:
                            _set_group_relative_runtime(False)
                        try:
                            merged_patch = merge_patches(
                                current_skill,
                                all_failure_patches,
                                all_success_patches,
                                batch_size=merge_bs,
                                verbose=True,
                                workers=cfg["analyst_workers"],
                                update_mode=update_mode,
                                meta_skill_context=resample_meta_context,
                                **cross_group_merge_kwargs,
                            )
                        finally:
                            if group_relative_candidate_competition:
                                _set_group_relative_runtime(True)
                        if main_group_relative_credit:
                            merged_patch, resample_credit_audit = annotate_merged_patch(
                                merged_patch,
                                all_failure_patches + all_success_patches,
                                update_mode=update_mode,
                                semantic_similarity_threshold=(
                                    group_relative_similarity_threshold
                                ),
                                direct_evidence_similarity_threshold=(
                                    group_relative_direct_evidence_similarity_threshold
                                ),
                                specificity_penalty_weight=(
                                    group_relative_specificity_penalty_weight
                                ),
                            )
                            with open(
                                os.path.join(
                                    step_dir,
                                    "group_relative_edit_credit_resample_"
                                    f"{resample_attempt:02d}.json",
                                ),
                                "w",
                            ) as f:
                                json.dump(
                                    resample_credit_audit,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        merged_items = get_payload_items(merged_patch, update_mode)
                        n_edits_merged = len(merged_items)
                        with open(
                            os.path.join(
                                step_dir,
                                f"merged_patch_resample_{resample_attempt:02d}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(merged_patch, f, ensure_ascii=False, indent=2)

                        if is_full_rewrite_minibatch_mode(update_mode):
                            ranked_patch = merged_patch
                        else:
                            ranked_patch = rank_and_select(
                                current_skill,
                                merged_patch,
                                max_edits=edit_budget,
                                update_mode=update_mode,
                                meta_skill_context=resample_meta_context,
                                group_relative_edit_credit=(
                                    main_group_relative_credit
                                ),
                                group_relative_max_specific_singletons=(
                                    group_relative_max_specific_singletons
                                ),
                                group_relative_min_cross_task_support=(
                                    group_relative_min_cross_task_support
                                ),
                                group_relative_max_exploratory=(
                                    group_relative_max_exploratory
                                ),
                                group_relative_min_atomic_support_fraction=(
                                    group_relative_min_atomic_support_fraction
                                ),
                                group_relative_min_candidate_cross_task_edits=(
                                    group_relative_min_candidate_cross_task_edits
                                ),
                                group_relative_max_contrastive_edits=(
                                    group_relative_max_contrastive_edits
                                ),
                                group_relative_max_singleton_edits=(
                                    group_relative_max_singleton_edits
                                ),
                                group_relative_block_singleton_stable_hypotheses=(
                                    group_relative_block_singleton_stable_hypotheses
                                ),
                                group_relative_actionable_priority_bonus=(
                                    group_relative_actionable_priority_bonus
                                ),
                                group_relative_causal_trace_priority_bonus=(
                                    group_relative_causal_trace_priority_bonus
                                ),
                            )
                        ranked_items = get_payload_items(ranked_patch, update_mode)
                        n_edits_ranked = len(ranked_items)
                        with open(
                            os.path.join(
                                step_dir,
                                f"ranked_edits_resample_{resample_attempt:02d}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                        ranked_items, resampled_hard_hits = (
                            _prune_hard_unobservable_items(
                                ranked_patch,
                                ranked_items,
                                candidate_attempt=resample_attempt,
                            )
                        )
                        n_edits_ranked = len(ranked_items)
                        if resampled_hard_hits:
                            with open(
                                os.path.join(
                                    step_dir,
                                    f"ranked_edits_resample_{resample_attempt:02d}.json",
                                ),
                                "w",
                            ) as f:
                                json.dump(
                                    ranked_patch,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        unobservable_runtime_edits = (
                            resampled_hard_hits
                            if resampled_hard_hits and not ranked_items
                            else []
                        )
                        matched_quarantine = None
                        blocked_permanent_edits = []
                        if quarantine_rejected_edits:
                            matched_quarantine = find_quarantined_candidate(
                                ranked_items,
                                quarantine_records,
                                update_mode=update_mode,
                                similarity_threshold=quarantine_edit_similarity_threshold,
                            )
                            blocked_permanent_edits = (
                                find_permanently_quarantined_edits(
                                    ranked_items,
                                    quarantine_records,
                                    update_mode=update_mode,
                                    behavior_similarity_threshold=(
                                        quarantine_behavior_similarity_threshold
                                    ),
                                )
                            )

                    step_rec["quarantine_candidate_collisions"] = len(
                        quarantine_collisions
                    )
                    step_rec["quarantine_permanent_edit_hits"] = len(
                        permanent_edit_hits
                    )
                    step_rec["runtime_observability_hits"] = len(
                        runtime_observability_hits
                    )
                    step_rec["hard_unobservable_edits_pruned"] = sum(
                        len(event["rejected"])
                        for event in hard_observability_prune_events
                    )
                    _persist_hard_observability_report(ranked_items)
                    step_rec["quarantine_resample_attempts"] = resample_attempt
                    step_rec["pre_validation_resample_attempts"] = resample_attempt
                    if resample_attempt:
                        with open(
                            os.path.join(step_dir, "quarantine_filter_report.json"),
                            "w",
                        ) as f:
                            json.dump(
                                {
                                    "scope": "candidate_edit_set",
                                    "collisions": quarantine_collisions,
                                    "permanent_edit_hits": list(
                                        permanent_edit_hits.values()
                                    ),
                                    "runtime_observability_hits": list(
                                        runtime_observability_hits.values()
                                    ),
                                    "resample_attempts": resample_attempt,
                                    "resolved": (
                                        matched_quarantine is None
                                        and not blocked_permanent_edits
                                        and not unobservable_runtime_edits
                                    ),
                                    "final_candidate_items": ranked_items,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                    if group_relative_ranked_patch is not None:
                        group_relative_selection_audit = dict(
                            group_relative_ranked_patch.get(
                                "group_relative_selection", {}
                            )
                        )
                        group_relative_branch_items = list(
                            get_payload_items(
                                group_relative_ranked_patch,
                                update_mode,
                            )
                        )
                        if (
                            group_relative_probe_enabled
                            and group_relative_branch_items
                        ):
                            t_group_probe = time.time()
                            group_probe_input_items = list(
                                group_relative_branch_items
                            )
                            group_probe_input_count = len(
                                group_probe_input_items
                            )
                            (
                                group_relative_ranked_patch,
                                group_branch_probe_audit,
                            ) = _run_group_relative_per_edit_probe(
                                adapter=adapter,
                                ranked_patch=group_relative_ranked_patch,
                                current_skill=current_skill,
                                train_items=all_train_items,
                                step_dir=os.path.join(
                                    step_dir,
                                    "group_relative_branch",
                                ),
                                out_root=out_root,
                                update_mode=update_mode,
                                max_tasks_per_edit=(
                                    group_relative_probe_max_tasks_per_edit
                                ),
                                rollout_count=(
                                    group_relative_probe_rollouts
                                ),
                                seed=seed + global_step * 100003,
                                fail_open=group_relative_probe_fail_open,
                            )
                            group_relative_branch_items = list(
                                get_payload_items(
                                    group_relative_ranked_patch,
                                    update_mode,
                                )
                            )
                            step_rec[
                                "group_relative_branch_per_edit_probe"
                            ] = {
                                "status": group_branch_probe_audit.get(
                                    "status"
                                ),
                                "input_count": group_branch_probe_audit.get(
                                    "input_count", group_probe_input_count
                                ),
                                "kept_count": group_branch_probe_audit.get(
                                    "kept_count",
                                    len(group_relative_branch_items),
                                ),
                                "rejected_count": (
                                    group_branch_probe_audit.get(
                                        "rejected_count",
                                        group_probe_input_count
                                        - len(group_relative_branch_items),
                                    )
                                ),
                                "official_validation_changed": False,
                            }
                            step_rec["timing"][
                                "group_relative_branch_train_probe_s"
                            ] = round(time.time() - t_group_probe, 1)
                            with open(
                                os.path.join(
                                    step_dir,
                                    "group_relative_branch_per_edit_probe.json",
                                ),
                                "w",
                            ) as f:
                                json.dump(
                                    group_branch_probe_audit,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            with open(
                                os.path.join(
                                    step_dir,
                                    "group_relative_branch_ranked_edits.json",
                                ),
                                "w",
                            ) as f:
                                json.dump(
                                    group_relative_ranked_patch,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            rejected_group_probe_items = [
                                group_probe_input_items[
                                    int(record["edit_index"])
                                ]
                                for record in group_branch_probe_audit.get(
                                    "items", []
                                )
                                if not record.get("accepted", False)
                                and isinstance(
                                    record.get("edit_index"), int
                                )
                                and 0
                                <= int(record["edit_index"])
                                < len(group_probe_input_items)
                            ]
                            if rejected_group_probe_items:
                                _record_edit_event(
                                    rejected_group_probe_items,
                                    step=global_step,
                                    epoch=epoch,
                                    step_in_epoch=step_in_epoch,
                                    stage=(
                                        "group_relative_branch_train_probe"
                                    ),
                                    candidate_attempt=0,
                                    validation_ran=False,
                                    outcome=(
                                        "rejected_by_train_source_probe"
                                    ),
                                    blocked_reasons=[
                                        "evidence_tier_source_task_gate"
                                    ],
                                )
                            print(
                                "    [group-relative branch probe] "
                                f"{group_probe_input_count} -> "
                                f"{len(group_relative_branch_items)} edits; "
                                "standard branch unchanged"
                            )
                        if reject_unobservable_runtime_edits:
                            (
                                group_relative_branch_items,
                                group_hard_observability_hits,
                            ) = partition_runtime_observable_edits(
                                group_relative_branch_items,
                                update_mode=update_mode,
                            )
                        else:
                            group_hard_observability_hits = []
                        set_payload_items(
                            group_relative_ranked_patch,
                            group_relative_branch_items,
                            update_mode,
                        )
                        group_policy_rejection_reasons = list(
                            group_relative_selection_audit.get(
                                "candidate_rejection_reasons", []
                            )
                        )
                        post_filter_cross_task_count = sum(
                            item.get(
                                "_group_relative_selection_category"
                            )
                            == "cross_task"
                            for item in group_relative_branch_items
                        )
                        if (
                            not group_policy_rejection_reasons
                            and not group_relative_soft_cross_task_target
                            and post_filter_cross_task_count
                            < group_relative_min_candidate_cross_task_edits
                        ):
                            group_policy_rejection_reasons.append(
                                "post_filter_cross_task_quota_not_met"
                            )
                        group_exact_quarantine = None
                        group_permanent_hits = []
                        if (
                            quarantine_rejected_edits
                            and group_relative_branch_items
                        ):
                            group_exact_quarantine = find_quarantined_candidate(
                                group_relative_branch_items,
                                quarantine_records,
                                update_mode=update_mode,
                                similarity_threshold=(
                                    quarantine_edit_similarity_threshold
                                ),
                            )
                            group_permanent_hits = (
                                find_permanently_quarantined_edits(
                                    group_relative_branch_items,
                                    quarantine_records,
                                    update_mode=update_mode,
                                    behavior_similarity_threshold=(
                                        quarantine_behavior_similarity_threshold
                                    ),
                                )
                            )
                        branch_filter_reasons = [
                            reason
                            for reason, present in (
                                (
                                    "no_atomic_evidence_supported_edits",
                                    not group_relative_branch_items,
                                ),
                                (
                                    "hard_unobservable_runtime_dependency",
                                    bool(group_hard_observability_hits)
                                    and not group_relative_branch_items,
                                ),
                                (
                                    "exact_or_near_rejected_candidate",
                                    group_exact_quarantine is not None,
                                ),
                                (
                                    "permanent_bad_edit",
                                    bool(group_permanent_hits),
                                ),
                            )
                            if present
                        ]
                        group_rejection_reasons = list(
                            dict.fromkeys(
                                group_policy_rejection_reasons
                                + branch_filter_reasons
                            )
                        )
                        group_relative_branch_audit = {
                            "enabled": True,
                            "eligible": not group_rejection_reasons,
                            "input_count": len(
                                get_payload_items(
                                    group_relative_ranked_patch,
                                    update_mode,
                                )
                            )
                            + len(group_hard_observability_hits),
                            "retained_count": len(
                                group_relative_branch_items
                            ),
                            "selection_policy": (
                                group_relative_selection_audit.get("policy")
                            ),
                            "required_cross_task_edit_count": (
                                group_relative_min_candidate_cross_task_edits
                            ),
                            "post_filter_cross_task_edit_count": (
                                post_filter_cross_task_count
                            ),
                            "hard_unobservable_count": len(
                                group_hard_observability_hits
                            ),
                            "permanent_bad_edit_count": len(
                                group_permanent_hits
                            ),
                            "exact_quarantine_match": (
                                group_exact_quarantine is not None
                            ),
                            "rejection_reasons": group_rejection_reasons,
                        }
                        with open(
                            os.path.join(
                                step_dir,
                                "candidate_branch_prevalidation.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                group_relative_branch_audit,
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                        main_branch_blocked = (
                            matched_quarantine is not None
                            or bool(blocked_permanent_edits)
                            or bool(unobservable_runtime_edits)
                            or not ranked_items
                        )
                        if group_relative_verified_only_candidate:
                            group_relative_branch_audit[
                                "standard_fallback_allowed"
                            ] = False
                            group_relative_branch_audit[
                                "standard_branch_retained_count"
                            ] = len(ranked_items)
                            if group_relative_branch_audit["eligible"]:
                                ranked_patch = group_relative_ranked_patch
                                ranked_items = group_relative_branch_items
                                n_edits_ranked = len(ranked_items)
                                candidate_primary_branch = (
                                    "group_relative_verified"
                                )
                                group_relative_branch_audit[
                                    "verified_only_decision"
                                ] = "use_verified_group_relative"
                                matched_quarantine = None
                                blocked_permanent_edits = []
                                unobservable_runtime_edits = []
                                print(
                                    "    [candidate competition] verified-only "
                                    "policy selected the evidence-backed branch"
                                )
                            else:
                                set_payload_items(
                                    ranked_patch,
                                    [],
                                    update_mode,
                                )
                                ranked_items = []
                                n_edits_ranked = 0
                                candidate_primary_branch = (
                                    "verified_only_no_candidate"
                                )
                                group_relative_branch_audit[
                                    "verified_only_decision"
                                ] = "skip_unverified_standard_fallback"
                                print(
                                    "    [candidate competition] verified-only "
                                    "policy rejected the unverified standard "
                                    "fallback"
                                )
                            group_relative_ranked_patch = None
                            group_relative_branch_items = []
                            with open(
                                os.path.join(step_dir, "ranked_edits.json"),
                                "w",
                            ) as f:
                                json.dump(
                                    ranked_patch,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        elif (
                            main_branch_blocked
                            and group_relative_branch_audit["eligible"]
                        ):
                            ranked_patch = group_relative_ranked_patch
                            ranked_items = group_relative_branch_items
                            n_edits_ranked = len(ranked_items)
                            candidate_primary_branch = (
                                "group_relative_fallback"
                            )
                            group_relative_ranked_patch = None
                            group_relative_branch_items = []
                            matched_quarantine = None
                            blocked_permanent_edits = []
                            unobservable_runtime_edits = []
                            print(
                                "    [candidate competition] standard "
                                "branch blocked; using deployable "
                                "group-relative fallback"
                            )
                        with open(
                            os.path.join(
                                step_dir,
                                "candidate_branch_prevalidation.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                group_relative_branch_audit,
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                    if (
                        matched_quarantine is not None
                        or blocked_permanent_edits
                        or unobservable_runtime_edits
                        or (resample_attempt and not ranked_items)
                    ):
                        if ranked_items and (
                            matched_quarantine is not None
                            or blocked_permanent_edits
                            or unobservable_runtime_edits
                        ):
                            _record_edit_event(
                                ranked_items,
                                step=global_step,
                                epoch=epoch,
                                step_in_epoch=step_in_epoch,
                                stage="pre_validation",
                                candidate_attempt=resample_attempt,
                                validation_ran=False,
                                outcome="blocked_after_resample_exhaustion",
                                blocked_reasons=[
                                    reason
                                    for reason, present in (
                                        (
                                            "exact_or_near_rejected_candidate",
                                            matched_quarantine is not None,
                                        ),
                                        (
                                            "permanent_bad_edit",
                                            bool(blocked_permanent_edits),
                                        ),
                                        (
                                            "unobservable_runtime_dependency",
                                            bool(unobservable_runtime_edits),
                                        ),
                                    )
                                    if present
                                ],
                                runtime_observability_hits=[
                                    {
                                        "behavior": hit.get("behavior", {}),
                                        "matched_reasons": hit.get(
                                            "matched_reasons", []
                                        ),
                                    }
                                    for hit in unobservable_runtime_edits
                                ],
                            )
                        step_rec["action"] = (
                            "skip_quarantined_candidate"
                            if matched_quarantine is not None
                            else (
                                "skip_permanent_bad_edit_candidate"
                                if blocked_permanent_edits
                                else (
                                    "skip_unobservable_runtime_candidate"
                                    if unobservable_runtime_edits
                                    else "skip_empty_resampled_candidate"
                                )
                            )
                        )
                        step_rec["current_score"] = current_score
                        step_rec["best_score"] = best_score
                        step_rec["best_step"] = best_step
                        step_rec["skill_len"] = len(current_skill)
                        step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                        history.append(step_rec)
                        _save_history(out_root, history)
                        _save_skill(out_root, global_step, current_skill)
                        _persist_runtime_state(global_step)
                        with open(
                            os.path.join(step_dir, "step_record.json"), "w"
                        ) as f:
                            json.dump(step_rec, f, indent=2, ensure_ascii=False)
                        if matched_quarantine is not None:
                            print(
                                "    [skip] candidate remained quarantined after "
                                f"{resample_attempt} resample attempt(s); "
                                "validation was not run"
                            )
                        elif blocked_permanent_edits:
                            print(
                                "    [skip] candidate still contained "
                                f"{len(blocked_permanent_edits)} permanently "
                                "quarantined edit(s); validation was not run"
                            )
                        elif unobservable_runtime_edits:
                            print(
                                "    [skip] candidate still depended on "
                                f"{len(unobservable_runtime_edits)} inference-time "
                                "unobservable signal(s); validation was not run"
                            )
                        else:
                            print(
                                "    [skip] pre-validation resampling returned no edits; "
                                "validation was not run"
                            )
                        continue

                    # Keep the canonical artifacts and counters aligned with the
                    # resampled candidate that will actually be validated.
                    if resample_attempt:
                        with open(
                            os.path.join(step_dir, "merged_patch.json"), "w"
                        ) as f:
                            json.dump(merged_patch, f, ensure_ascii=False, indent=2)
                        with open(
                            os.path.join(step_dir, "ranked_edits.json"), "w"
                        ) as f:
                            json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                    step_rec["n_edits_merged"] = n_edits_merged
                    step_rec["n_edits_ranked"] = n_edits_ranked
                step_rec["timing"]["select_s"] = round(time.time() - t_phase, 1)

                support_counts = [
                    item.get("support_count", 0) for item in ranked_items if isinstance(item, dict)
                ]
                step_rec["support_counts"] = support_counts
                if is_full_rewrite_minibatch_mode(update_mode):
                    print(
                        f"    [4/6 SELECT] skipped LR/select; "
                        f"using {n_edits_ranked} merged {payload_label(update_mode)}"
                    )
                else:
                    print(
                        f"    [4/6 SELECT] "
                        f"{n_edits_merged} -> {n_edits_ranked} {payload_label(update_mode)} "
                        f"(budget={edit_budget}, lr_control={lr_control_mode})"
                    )

                if (
                    group_relative_probe_enabled
                    and not group_relative_candidate_competition
                    and ranked_items
                ):
                    t_probe = time.time()
                    probe_input_items = list(ranked_items)
                    ranked_patch, probe_audit = _run_group_relative_per_edit_probe(
                        adapter=adapter,
                        ranked_patch=ranked_patch,
                        current_skill=current_skill,
                        train_items=all_train_items,
                        step_dir=step_dir,
                        out_root=out_root,
                        update_mode=update_mode,
                        max_tasks_per_edit=(
                            group_relative_probe_max_tasks_per_edit
                        ),
                        rollout_count=group_relative_probe_rollouts,
                        seed=seed + global_step * 100003,
                        fail_open=group_relative_probe_fail_open,
                    )
                    ranked_items = get_payload_items(ranked_patch, update_mode)
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["support_counts"] = [
                        item.get("support_count", 0)
                        for item in ranked_items
                        if isinstance(item, dict)
                    ]
                    step_rec["n_edits_ranked_after_probe"] = n_edits_ranked
                    step_rec["group_relative_per_edit_probe"] = {
                        "status": probe_audit.get("status"),
                        "input_count": probe_audit.get(
                            "input_count", len(probe_input_items)
                        ),
                        "kept_count": probe_audit.get(
                            "kept_count", len(ranked_items)
                        ),
                        "rejected_count": probe_audit.get(
                            "rejected_count",
                            len(probe_input_items) - len(ranked_items),
                        ),
                        "official_validation_changed": False,
                    }
                    step_rec["timing"]["train_probe_s"] = round(
                        time.time() - t_probe, 1
                    )
                    with open(
                        os.path.join(
                            step_dir, "group_relative_per_edit_probe.json"
                        ),
                        "w",
                    ) as f:
                        json.dump(probe_audit, f, ensure_ascii=False, indent=2)
                    with open(
                        os.path.join(step_dir, "ranked_edits.json"), "w"
                    ) as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                    rejected_probe_items = [
                        probe_input_items[int(record["edit_index"])]
                        for record in probe_audit.get("items", [])
                        if not record.get("accepted", False)
                        and isinstance(record.get("edit_index"), int)
                        and 0 <= int(record["edit_index"]) < len(probe_input_items)
                    ]
                    if rejected_probe_items:
                        _record_edit_event(
                            rejected_probe_items,
                            step=global_step,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            stage="train_probe",
                            candidate_attempt=step_rec.get(
                                "quarantine_resample_attempts", 0
                            ),
                            validation_ran=False,
                            outcome="rejected_by_train_source_probe",
                            blocked_reasons=[
                                "task_paired_train_hard_non_degrading_gate"
                            ],
                        )
                    print(
                        "    [train per-edit probe] "
                        f"{len(probe_input_items)} -> {len(ranked_items)} edits; "
                        "official validation unchanged"
                    )

                if not ranked_items:
                    if use_skill_aware:
                        current_skill = _flush_skill_aware_appendix(
                            current_skill,
                            all_raw_patches,
                            step_rec,
                            step_dir,
                            cfg,
                        )
                    step_rec["action"] = (
                        "skip_no_train_probe_supported_edits"
                        if group_relative_probe_enabled
                        else "skip_no_evidence_supported_edits"
                    )
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(
                        os.path.join(step_dir, "step_record.json"), "w"
                    ) as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print(
                        "    [skip] no evidence-supported edits remain; "
                        "official validation was not run"
                    )
                    continue

                # ⑤ UPDATE ─────────────────────────────────────────────────
                t_phase = time.time()
                rewrite_result = None
                if update_mode == "rewrite_from_suggestions":
                    rewrite_result = rewrite_skill_from_suggestions(
                        current_skill,
                        ranked_patch,
                        step_buffer_context=step_buffer_context,
                        env=cfg.get("env"),
                        reasoning_effort=rewrite_reasoning_effort,
                        max_completion_tokens=rewrite_max_completion_tokens,
                    )
                    if rewrite_result and rewrite_result.get("new_skill"):
                        candidate_skill = rewrite_result["new_skill"]
                        apply_report = []
                        with open(os.path.join(step_dir, "rewrite_result.json"), "w") as f:
                            json.dump(rewrite_result, f, ensure_ascii=False, indent=2)
                    else:
                        candidate_skill = current_skill
                        apply_report = []
                elif is_full_rewrite_minibatch_mode(update_mode):
                    skill_candidates = get_payload_items(ranked_patch, update_mode)
                    selected_candidate = next(
                        (
                            item for item in skill_candidates
                            if isinstance(item, dict) and str(item.get("new_skill", "")).strip()
                        ),
                        None,
                    )
                    if selected_candidate:
                        candidate_skill = str(selected_candidate["new_skill"]).rstrip() + "\n"
                        apply_report = []
                        rewrite_result = {
                            "reasoning": ranked_patch.get("reasoning", ""),
                            "change_summary": selected_candidate.get("change_summary", []),
                            "title": selected_candidate.get("title", ""),
                            "source_type": selected_candidate.get("source_type", ""),
                        }
                        with open(os.path.join(step_dir, "full_rewrite_result.json"), "w") as f:
                            json.dump(
                                {
                                    "selected_candidate": selected_candidate,
                                    "merged_patch": ranked_patch,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    else:
                        candidate_skill = current_skill
                        apply_report = []
                else:
                    candidate_skill, apply_report = apply_patch_with_report(current_skill, ranked_patch)

                alternate_candidate_skill = None
                alternate_apply_report: list[dict] = []
                if (
                    group_relative_ranked_patch is not None
                    and group_relative_branch_audit.get("eligible", False)
                ):
                    (
                        proposed_group_relative_skill,
                        alternate_apply_report,
                    ) = apply_patch_with_report(
                        current_skill,
                        group_relative_ranked_patch,
                    )
                    if (
                        proposed_group_relative_skill != current_skill
                        and proposed_group_relative_skill != candidate_skill
                    ):
                        alternate_candidate_skill = (
                            proposed_group_relative_skill
                        )
                        branch_dir = os.path.join(
                            step_dir, "candidate_branches"
                        )
                        os.makedirs(branch_dir, exist_ok=True)
                        with open(
                            os.path.join(
                                branch_dir,
                                "group_relative_candidate_skill.md",
                            ),
                            "w",
                        ) as f:
                            f.write(alternate_candidate_skill)
                        with open(
                            os.path.join(
                                branch_dir,
                                "group_relative_apply_report.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                alternate_apply_report,
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    else:
                        group_relative_branch_audit["eligible"] = False
                        group_relative_branch_audit[
                            "rejection_reasons"
                        ].append(
                            "no_distinct_applied_skill"
                        )
                step_rec["candidate_primary_branch"] = (
                    candidate_primary_branch
                )
                step_rec["candidate_branch_competition"] = (
                    group_relative_branch_audit
                )
                with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                    f.write(candidate_skill)
                if apply_report:
                    with open(os.path.join(step_dir, "edit_apply_report.json"), "w") as f:
                        json.dump(apply_report, f, indent=2, ensure_ascii=False)

                cand_hash = skill_hash(candidate_skill)
                step_rec["candidate_hash"] = cand_hash
                step_rec["candidate_skill_len"] = len(candidate_skill)
                if rewrite_result:
                    step_rec["rewrite_change_summary"] = rewrite_result.get("change_summary", [])
                if apply_report:
                    step_rec["edit_apply_summary"] = {
                        "total": len(apply_report),
                        "applied": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("applied")
                        ),
                        "skipped": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("skipped")
                        ),
                        "errors": sum(
                            1 for row in apply_report if row.get("status") == "error"
                        ),
                    }
                step_rec["timing"]["update_s"] = round(time.time() - t_phase, 1)
                if (
                    update_mode == "rewrite_from_suggestions"
                    and rewrite_result is None
                ) or (
                    is_full_rewrite_minibatch_mode(update_mode)
                    and rewrite_result is None
                ):
                    _record_edit_event(
                        ranked_items,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        stage="update",
                        candidate_attempt=step_rec.get(
                            "quarantine_resample_attempts", 0
                        ),
                        validation_ran=False,
                        outcome="skip_no_rewrite",
                        apply_report=apply_report,
                    )
                    # Skill-aware: flush appendix notes before skipping (see
                    # the skip_no_patches branch above).
                    if use_skill_aware:
                        current_skill = _flush_skill_aware_appendix(
                            current_skill, all_raw_patches, step_rec, step_dir, cfg,
                        )
                    step_rec["action"] = "skip_no_rewrite"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable rewrite generated — skill unchanged")
                    continue
                print(
                    f"    [5/6 UPDATE] "
                    f"skill_len {len(current_skill)} -> {len(candidate_skill)}"
                )

                # ⑥ EVALUATE ───────────────────────────────────────────────
                t_phase = time.time()
                if alternate_candidate_skill is not None:
                    branch_candidates = [
                        {
                            "name": candidate_primary_branch,
                            "skill": candidate_skill,
                            "patch": ranked_patch,
                            "items": ranked_items,
                            "apply_report": apply_report,
                            "tie_priority": 1,
                        },
                        {
                            "name": "group_relative",
                            "skill": alternate_candidate_skill,
                            "patch": group_relative_ranked_patch,
                            "items": group_relative_branch_items,
                            "apply_report": alternate_apply_report,
                            "tie_priority": 0,
                        },
                    ]
                    branch_results = []
                    for branch in branch_candidates:
                        branch_hash = skill_hash(branch["skill"])
                        branch_selection_results = sel_result_cache.get(
                            branch_hash
                        )
                        if branch_hash in sel_cache and (
                            not paired_audit_enabled
                            or branch_selection_results is not None
                        ):
                            branch_hard, branch_soft = sel_cache[branch_hash]
                        else:
                            branch_env, branch_n = _build_eval_env(
                                split="valid_seen",
                                env_num=cfg["sel_env_num"],
                                seed=seed,
                            )
                            print(
                                "    [candidate competition] "
                                f"branch={branch['name']} "
                                f"selection items={branch_n}"
                            )
                            branch_eval_dir = os.path.join(
                                step_dir,
                                "candidate_branches",
                                str(branch["name"]),
                                "selection_eval",
                            )
                            try:
                                branch_selection_results = adapter.rollout(
                                    branch_env,
                                    branch["skill"],
                                    branch_eval_dir,
                                )
                            except Exception as exc:  # noqa: BLE001
                                if branch["name"] != "group_relative":
                                    raise
                                branch.update(
                                    {
                                        "hash": branch_hash,
                                        "hard": -1.0,
                                        "soft": -1.0,
                                        "gate_score": -1.0,
                                        "selection_results": None,
                                        "eligible": False,
                                        "gate_action": "evaluation_error",
                                        "paired": None,
                                        "error": str(exc),
                                    }
                                )
                                branch_results.append(branch)
                                print(
                                    "    [candidate competition] auxiliary "
                                    "group-relative validation failed; "
                                    f"standard branch continues: {exc}"
                                )
                                continue
                            branch_hard, branch_soft = compute_score(
                                branch_selection_results
                            )
                            sel_cache[branch_hash] = (
                                branch_hard,
                                branch_soft,
                            )
                            sel_result_cache[branch_hash] = (
                                branch_selection_results
                            )

                        branch_gate_score = select_gate_score(
                            branch_hard,
                            branch_soft,
                            gate_metric,
                            gate_mixed_weight,
                            skill_content=branch["skill"],
                            use_semantic_density=use_semantic_density,
                            semantic_density_weight=(
                                semantic_density_weight
                            ),
                            leading_words=leading_words,
                        )
                        branch_gate = (
                            evaluate_gate(
                                candidate_skill=branch["skill"],
                                cand_hard=branch_hard,
                                current_skill=current_skill,
                                current_score=current_score,
                                best_skill=best_skill,
                                best_score=best_score,
                                best_step=best_step,
                                global_step=global_step,
                                cand_soft=branch_soft,
                                metric=gate_metric,
                                mixed_weight=gate_mixed_weight,
                                use_semantic_density=(
                                    use_semantic_density
                                ),
                                semantic_density_weight=(
                                    semantic_density_weight
                                ),
                                leading_words=leading_words,
                            )
                            if use_gate
                            else None
                        )
                        branch_paired = None
                        branch_paired_risk = None
                        paired_accepted = True
                        if paired_audit_enabled:
                            if (
                                current_selection_results is None
                                or branch_selection_results is None
                            ):
                                raise RuntimeError(
                                    "candidate competition requires paired "
                                    "selection results"
                                )
                            branch_paired = build_paired_promotion_audit(
                                current_selection_results,
                                branch_selection_results,
                                require_hard_gain=(
                                    paired_require_hard_gain
                                ),
                            )
                            if paired_non_degrading:
                                paired_accepted = bool(
                                    branch_paired["accepted"]
                                )
                            if paired_risk_sensitive:
                                branch_paired_risk = (
                                    _build_paired_risk_decision(
                                        branch_paired,
                                        harm_weight=paired_harm_weight,
                                        min_margin=paired_risk_min_margin,
                                    )
                                )
                                paired_accepted = (
                                    paired_accepted
                                    and bool(branch_paired_risk["accepted"])
                                )
                        gate_accepted = (
                            True
                            if not use_gate
                            else branch_gate.action
                            in {"accept", "accept_new_best"}
                        )
                        branch["hash"] = branch_hash
                        branch["hard"] = branch_hard
                        branch["soft"] = branch_soft
                        branch["gate_score"] = branch_gate_score
                        branch["selection_results"] = (
                            branch_selection_results
                        )
                        branch["eligible"] = (
                            gate_accepted and paired_accepted
                        )
                        branch["gate_action"] = (
                            branch_gate.action
                            if branch_gate is not None
                            else "force_accept"
                        )
                        branch["paired"] = branch_paired
                        branch["paired_risk"] = branch_paired_risk
                        branch_results.append(branch)

                    winning_branch = _choose_candidate_branch(
                        branch_results
                    )
                    group_loser_quarantine_added = False
                    for branch in branch_results:
                        if (
                            branch["name"] != "group_relative"
                            or branch is winning_branch
                        ):
                            continue
                        group_validation_ran = (
                            branch.get("gate_action")
                            != "evaluation_error"
                        )
                        _record_edit_event(
                            branch["items"],
                            step=global_step,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            stage="candidate_competition",
                            candidate_attempt=0,
                            validation_ran=group_validation_ran,
                            outcome="rejected_by_candidate_competition",
                            blocked_reasons=[
                                "standard_branch_won_unchanged_validation"
                            ],
                            metrics={
                                "candidate_hard": branch["hard"],
                                "candidate_soft": branch["soft"],
                                "candidate_gate_score": branch[
                                    "gate_score"
                                ],
                                "winner_gate_score": winning_branch[
                                    "gate_score"
                                ],
                            },
                            apply_report=branch["apply_report"],
                        )
                        if (
                            quarantine_rejected_edits
                            and group_validation_ran
                            and branch["items"]
                            and _competition_loss_has_attributable_regression(
                                branch,
                                winning_branch,
                            )
                        ):
                            added_group_quarantine = (
                                add_quarantined_candidate(
                                    quarantine_records,
                                    branch["items"],
                                    update_mode=update_mode,
                                    step=global_step,
                                    reason=(
                                        "candidate_competition_attributable_regression"
                                    ),
                                    similarity_threshold=(
                                        quarantine_edit_similarity_threshold
                                    ),
                                )
                            )
                            group_loser_quarantine_added = (
                                added_group_quarantine is not None
                            )
                            if added_group_quarantine is not None:
                                promote_repeated_edits(
                                    quarantine_records,
                                    edit_history,
                                    threshold=(
                                        quarantine_bad_edit_threshold
                                    ),
                                    behavior_similarity_threshold=(
                                        quarantine_behavior_similarity_threshold
                                    ),
                                )
                                save_quarantine(
                                    quarantine_path,
                                    quarantine_records,
                                )

                    branch_audit = {
                        "policy": (
                            "unchanged_validation_best_non_degrading"
                        ),
                        "validation_split": "valid_seen",
                        "validation_changed": False,
                        "winner": winning_branch["name"],
                        "group_loser_quarantine_added": (
                            group_loser_quarantine_added
                        ),
                        "branches": [
                            {
                                "name": branch["name"],
                                "hash": branch["hash"],
                                "hard": branch["hard"],
                                "soft": branch["soft"],
                                "gate_score": branch["gate_score"],
                                "gate_action": branch["gate_action"],
                                "paired_accepted": (
                                    branch["paired"]["accepted"]
                                    if branch["paired"] is not None
                                    else None
                                ),
                                "paired_risk_accepted": (
                                    branch["paired_risk"]["accepted"]
                                    if branch.get("paired_risk") is not None
                                    else None
                                ),
                                "eligible": branch["eligible"],
                                "error": branch.get("error"),
                            }
                            for branch in branch_results
                        ],
                    }
                    branch_dir = os.path.join(
                        step_dir, "candidate_branches"
                    )
                    os.makedirs(branch_dir, exist_ok=True)
                    with open(
                        os.path.join(branch_dir, "competition.json"),
                        "w",
                    ) as f:
                        json.dump(
                            branch_audit,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    step_rec["candidate_branch_competition"] = (
                        branch_audit
                    )
                    if winning_branch["skill"] != candidate_skill:
                        candidate_skill = winning_branch["skill"]
                        ranked_patch = winning_branch["patch"]
                        ranked_items = winning_branch["items"]
                        apply_report = winning_branch["apply_report"]
                        cand_hash = winning_branch["hash"]
                        candidate_primary_branch = winning_branch["name"]
                        step_rec["candidate_primary_branch"] = (
                            candidate_primary_branch
                        )
                        step_rec["candidate_hash"] = cand_hash
                        step_rec["candidate_skill_len"] = len(
                            candidate_skill
                        )
                        step_rec["n_edits_ranked"] = len(ranked_items)
                        step_rec["support_counts"] = [
                            item.get("support_count", 0)
                            for item in ranked_items
                            if isinstance(item, dict)
                        ]
                        step_rec["edit_apply_summary"] = {
                            "total": len(apply_report),
                            "applied": sum(
                                1
                                for row in apply_report
                                if str(row.get("status", "")).startswith(
                                    "applied"
                                )
                            ),
                            "skipped": sum(
                                1
                                for row in apply_report
                                if str(row.get("status", "")).startswith(
                                    "skipped"
                                )
                            ),
                            "errors": sum(
                                1
                                for row in apply_report
                                if row.get("status") == "error"
                            ),
                        }
                        with open(
                            os.path.join(step_dir, "candidate_skill.md"),
                            "w",
                        ) as f:
                            f.write(candidate_skill)
                        with open(
                            os.path.join(step_dir, "ranked_edits.json"),
                            "w",
                        ) as f:
                            json.dump(
                                ranked_patch,
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        if apply_report:
                            with open(
                                os.path.join(
                                    step_dir,
                                    "edit_apply_report.json",
                                ),
                                "w",
                            ) as f:
                                json.dump(
                                    apply_report,
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                    print(
                        "    [candidate competition] winner="
                        f"{winning_branch['name']} "
                        f"hard={winning_branch['hard']:.4f} "
                        f"soft={winning_branch['soft']:.4f}"
                    )
                if current_selection_results is not None:
                    before_eval_hard, before_eval_soft = compute_score(
                        current_selection_results
                    )
                else:
                    cached_current_metrics = sel_cache.get(skill_hash(current_skill))
                    if cached_current_metrics is not None:
                        before_eval_hard, before_eval_soft = cached_current_metrics
                    else:
                        before_eval_hard, before_eval_soft = None, None
                candidate_selection_results: list[dict] | None = None
                if cand_hash in sel_cache and (
                    not paired_audit_enabled or cand_hash in sel_result_cache
                ):
                    cand_hard, cand_soft = sel_cache[cand_hash]
                    candidate_selection_results = sel_result_cache.get(cand_hash)
                    print(
                        f"    [6/6 EVALUATE] "
                        f"cache hit {cand_hash}: hard={cand_hard:.4f}"
                    )
                else:
                    sel_env, sel_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    print(f"    [6/6 EVALUATE] selection items={sel_n}")
                    sel_eval_dir = os.path.join(step_dir, "selection_eval")
                    sel_results = adapter.rollout(sel_env, candidate_skill, sel_eval_dir)
                    cand_hard, cand_soft = compute_score(sel_results)
                    sel_cache[cand_hash] = (cand_hard, cand_soft)
                    sel_result_cache[cand_hash] = sel_results
                    candidate_selection_results = sel_results

                step_rec["selection_hard"] = cand_hard
                step_rec["selection_soft"] = cand_soft

                paired_audit = None
                paired_risk = None
                if paired_audit_enabled:
                    if current_selection_results is None or candidate_selection_results is None:
                        raise RuntimeError(
                            "paired promotion requires current and candidate per-item results"
                        )
                    paired_audit = build_paired_promotion_audit(
                        current_selection_results,
                        candidate_selection_results,
                        require_hard_gain=paired_require_hard_gain,
                    )
                    step_rec["paired_promotion"] = paired_audit
                    if paired_risk_sensitive:
                        paired_risk = _build_paired_risk_decision(
                            paired_audit,
                            harm_weight=paired_harm_weight,
                            min_margin=paired_risk_min_margin,
                        )
                        step_rec["paired_risk"] = paired_risk
                    with open(
                        os.path.join(step_dir, "paired_promotion_audit.json"), "w"
                    ) as f:
                        json.dump(paired_audit, f, ensure_ascii=False, indent=2)
                    print(
                        "    [paired gate] "
                        f"beneficial={paired_audit['beneficial_count']} "
                        f"harmful={paired_audit['harmful_count']} "
                        f"matched={paired_audit['matched_count']}"
                    )
                    if paired_risk is not None:
                        print(
                            "    [paired risk gate] "
                            f"score={paired_risk['risk_score']:.4f} "
                            f"accepted={paired_risk['accepted']}"
                        )

                gate = evaluate_gate(
                    candidate_skill=candidate_skill,
                    cand_hard=cand_hard,
                    current_skill=current_skill,
                    current_score=current_score,
                    best_skill=best_skill,
                    best_score=best_score,
                    best_step=best_step,
                    global_step=global_step,
                    cand_soft=cand_soft,
                    metric=gate_metric,
                    mixed_weight=gate_mixed_weight,
                    use_semantic_density=use_semantic_density,
                    semantic_density_weight=semantic_density_weight,
                    leading_words=leading_words,
                ) if use_gate else None
                cand_gate_score = select_gate_score(
                    cand_hard, cand_soft, gate_metric, gate_mixed_weight,
                    skill_content=candidate_skill,
                    use_semantic_density=use_semantic_density,
                    semantic_density_weight=semantic_density_weight,
                    leading_words=leading_words,
                )
                if not use_gate:
                    # Validation ran (scores recorded above) but the gate is
                    # disabled: force-accept the candidate as the new current
                    # skill. Best-so-far is still tracked for convenience; the
                    # final skill is selected manually from the trajectory.
                    if cand_gate_score > best_score:
                        fa_best_skill = candidate_skill
                        fa_best_score = cand_gate_score
                        fa_best_step = global_step
                    else:
                        fa_best_skill = best_skill
                        fa_best_score = best_score
                        fa_best_step = best_step
                    gate = GateResult(
                        action="force_accept",
                        current_skill=candidate_skill,
                        current_score=cand_gate_score,
                        best_skill=fa_best_skill,
                        best_score=fa_best_score,
                        best_step=fa_best_step,
                    )
                elif (
                    gate.action in {"accept", "accept_new_best"}
                    and (
                        (
                            paired_non_degrading
                            and paired_audit is not None
                            and not paired_audit["accepted"]
                        )
                        or (
                            paired_risk_sensitive
                            and paired_risk is not None
                            and not paired_risk["accepted"]
                        )
                    )
                ):
                    gate = GateResult(
                        action="reject",
                        current_skill=current_skill,
                        current_score=current_score,
                        best_skill=best_skill,
                        best_score=best_score,
                        best_step=best_step,
                    )
                    step_rec["paired_gate_override"] = True
                    step_rec["paired_gate_override_policy"] = (
                        "risk_sensitive"
                        if paired_risk_sensitive
                        and paired_risk is not None
                        and not paired_risk["accepted"]
                        else "strict_non_degrading"
                    )
                step_rec["gate_metric"] = gate_metric
                step_rec["candidate_gate_score"] = cand_gate_score
                step_rec["action"] = gate.action
                prev_current = current_score
                prev_best = best_score
                current_skill = gate.current_skill
                current_score = gate.current_score
                best_skill = gate.best_skill
                best_score = gate.best_score
                best_step = gate.best_step
                if gate.action in {"accept", "accept_new_best", "force_accept"}:
                    current_origin = f"step_{global_step:04d}"
                    if candidate_selection_results is not None:
                        current_selection_results = candidate_selection_results
                if gate.action == "accept_new_best" or (
                    gate.action == "force_accept" and best_step == global_step
                ):
                    best_origin = current_origin
                    if candidate_selection_results is not None:
                        best_selection_results = candidate_selection_results

                if use_skill_aware:
                    current_skill = _flush_skill_aware_appendix(
                        current_skill, all_raw_patches, step_rec, step_dir, cfg,
                    )

                if gate_metric == "hard":
                    score_label = f"hard={cand_hard:.4f}"
                elif gate_metric == "soft":
                    score_label = f"soft={cand_soft:.4f}"
                else:
                    score_label = (
                        f"mixed[w={gate_mixed_weight}]={cand_gate_score:.4f} "
                        f"(hard={cand_hard:.4f} soft={cand_soft:.4f})"
                    )
                if gate.action == "accept_new_best":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT (new best) "
                        f"{score_label} > prev best {prev_best:.4f}"
                    )
                elif gate.action == "accept":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT "
                        f"{score_label} > current={prev_current:.4f}"
                    )
                elif gate.action == "force_accept":
                    print(
                        f"    [6/6 EVALUATE] FORCE-ACCEPT (gate disabled) "
                        f"{score_label}"
                    )
                elif step_rec.get("paired_gate_override"):
                    policy = step_rec.get(
                        "paired_gate_override_policy",
                        "strict_non_degrading",
                    )
                    reasons = ",".join(
                        (
                            paired_risk.get("rejection_reasons", [])
                            if policy == "risk_sensitive" and paired_risk
                            else paired_audit.get("rejection_reasons", [])
                            if paired_audit
                            else []
                        )
                    )
                    print(
                        "    [6/6 EVALUATE] REJECT "
                        f"(paired {policy} gate: {reasons}) "
                        f"{score_label}"
                    )
                else:
                    print(
                        f"    [6/6 EVALUATE] REJECT "
                        f"{score_label} <= current={current_score:.4f}"
                    )

                no_effect = (
                    before_eval_hard is not None
                    and before_eval_soft is not None
                    and abs(cand_hard - before_eval_hard) <= 1e-12
                    and abs(cand_soft - before_eval_soft) <= 1e-12
                )
                edit_event = _record_edit_event(
                    ranked_items,
                    step=global_step,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    stage="validation",
                    candidate_attempt=step_rec.get(
                        "quarantine_resample_attempts", 0
                    ),
                    validation_ran=True,
                    outcome=gate.action,
                    no_effect=no_effect,
                    metrics={
                        "before_hard": before_eval_hard,
                        "before_soft": before_eval_soft,
                        "candidate_hard": cand_hard,
                        "candidate_soft": cand_soft,
                        "hard_delta": (
                            cand_hard - before_eval_hard
                            if before_eval_hard is not None
                            else None
                        ),
                        "soft_delta": (
                            cand_soft - before_eval_soft
                            if before_eval_soft is not None
                            else None
                        ),
                        "gate_metric": gate_metric,
                        "candidate_gate_score": cand_gate_score,
                    },
                    paired_summary=(
                        {
                            "beneficial_count": paired_audit.get(
                                "beneficial_count", 0
                            ),
                            "harmful_count": paired_audit.get(
                                "harmful_count", 0
                            ),
                            "stable_success_count": paired_audit.get(
                                "stable_success_count", 0
                            ),
                            "stable_failure_count": paired_audit.get(
                                "stable_failure_count", 0
                            ),
                            "mean_soft_delta": paired_audit.get(
                                "mean_soft_delta", 0.0
                            ),
                        }
                        if paired_audit is not None
                        else None
                    ),
                    apply_report=apply_report,
                )
                step_rec["edit_history_candidate_fingerprint"] = edit_event[
                    "candidate_fingerprint"
                ]
                step_rec["edit_history_no_effect"] = no_effect

                if (
                    quarantine_rejected_edits
                    and gate.action == "reject"
                    and ranked_items
                ):
                    added_quarantine = add_quarantined_candidate(
                        quarantine_records,
                        ranked_items,
                        update_mode=update_mode,
                        step=global_step,
                        reason=(
                            "paired_non_degrading_reject"
                            if step_rec.get("paired_gate_override")
                            else "validation_reject"
                        ),
                        similarity_threshold=quarantine_edit_similarity_threshold,
                    )
                    newly_permanent_edits = promote_repeated_edits(
                        quarantine_records,
                        edit_history,
                        threshold=quarantine_bad_edit_threshold,
                        behavior_similarity_threshold=(
                            quarantine_behavior_similarity_threshold
                        ),
                    )
                    save_quarantine(quarantine_path, quarantine_records)
                    step_rec["quarantine_added_count"] = int(
                        added_quarantine is not None
                    )
                    step_rec["quarantine_total_count"] = len(quarantine_records)
                    step_rec["quarantine_new_permanent_edit_count"] = len(
                        newly_permanent_edits
                    )
                    if newly_permanent_edits:
                        step_rec["quarantine_new_permanent_edits"] = [
                            record.get("semantic", {})
                            for record in newly_permanent_edits
                        ]
                    print(
                        "    [quarantine] recorded rejected candidate edit set "
                        f"(items={len(ranked_items)}, total={len(quarantine_records)})"
                    )
                    if newly_permanent_edits:
                        print(
                            "    [quarantine] permanently blocked "
                            f"{len(newly_permanent_edits)} edit(s) after reaching "
                            f"the {quarantine_bad_edit_threshold}-candidate threshold"
                        )

                step_rec["timing"]["evaluate_s"] = round(time.time() - t_phase, 1)

                # ── Step buffer: unified failure patterns + rejected edits ─
                action = step_rec.get("action", "unknown")
                group_stats = _same_task_group_stats(all_rollout_results)
                n_total = group_stats["task_group_count"] or 1
                n_fail = (
                    group_stats["mixed_count"]
                    + group_stats["stable_failure_count"]
                )
                failure_patterns = _extract_failure_patterns(
                    all_rollout_results, step_dir,
                )

                buf_entry: dict = {
                    "step": global_step,
                    "action": action,
                    "n_total": n_total,
                    "n_fail": n_fail,
                    "failure_patterns": failure_patterns,
                }

                # Attach rejected edits when the step was rejected
                if "reject" in action and ranked_patch:
                    rejected_edits = [
                        short_item_summary(item, update_mode)
                        for item in ranked_items
                        if isinstance(item, dict)
                    ]
                    buf_entry["score_before"] = current_score
                    buf_entry["score_after"] = cand_gate_score
                    buf_entry["rejected_edits"] = rejected_edits

                step_buffer.append(buf_entry)

                # Persist step digest for step buffer context
                digest_path = os.path.join(step_dir, "trajectory_digest.json")
                with open(digest_path, "w") as f:
                    json.dump(buf_entry, f, indent=2, ensure_ascii=False)

                # ── Token snapshot ───────────────────────────────────────
                tokens_after = get_token_summary()
                step_tokens: dict = {}
                for stage in tokens_after:
                    if stage == "_total":
                        continue
                    after = tokens_after[stage]
                    before = tokens_before.get(stage, {})
                    step_tokens[stage] = {
                        "calls": after.get("calls", 0) - before.get("calls", 0),
                        "prompt_tokens": after.get("prompt_tokens", 0)
                        - before.get("prompt_tokens", 0),
                        "completion_tokens": after.get("completion_tokens", 0)
                        - before.get("completion_tokens", 0),
                    }
                step_rec["tokens"] = step_tokens

                # ── Save state ───────────────────────────────────────────
                step_rec["current_score"] = current_score
                step_rec["best_score"] = best_score
                step_rec["best_step"] = best_step
                step_rec["current_origin"] = current_origin
                step_rec["best_origin"] = best_origin
                step_rec["skill_len"] = len(current_skill)
                step_rec["wall_time_s"] = round(time.time() - step_t0, 1)

                _save_skill(out_root, global_step, current_skill)
                with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                    f.write(best_skill)
                history.append(step_rec)
                _save_history(out_root, history)
                _persist_runtime_state(global_step)
                with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                    json.dump(step_rec, f, indent=2, ensure_ascii=False)

                timing = step_rec["timing"]
                print(
                    f"\n  [STEP {global_step} done] "
                    f"epoch={epoch} action={step_rec['action']} "
                    f"current={current_score:.4f} best={best_score:.4f} "
                    f"dt={step_rec['wall_time_s']}s\n"
                    f"    timing: rollout={timing.get('rollout_s',0)}s "
                    f"reflect={timing.get('reflect_s',0)}s "
                    f"aggregate={timing.get('aggregate_s',0)}s "
                    f"select={timing.get('select_s',0)}s "
                    f"evaluate={timing.get('evaluate_s',0)}s"
                )

            epoch_last_step_skill = current_skill
            epoch_comparison_pairs: list[dict] | None = None

            # ── SLOW UPDATE (end of epoch) ──────────────────────────────
            use_slow = cfg.get("use_slow_update", False)
            if use_slow:
                slow_dir = os.path.join(out_root, "slow_update", f"epoch_{epoch:02d}")
                slow_done_path = os.path.join(slow_dir, "slow_result.json")

                if os.path.exists(slow_done_path):
                    # Resume support
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"resumed — already done"
                    )
                    with open(slow_done_path) as f:
                        slow_saved = json.load(f)
                    comparison_path = os.path.join(slow_dir, "comparison_pairs.json")
                    if os.path.exists(comparison_path):
                        try:
                            with open(comparison_path) as f:
                                epoch_comparison_pairs = json.load(f)
                        except Exception:
                            epoch_comparison_pairs = None
                    if (
                        slow_saved.get("slow_update_content")
                        and epoch >= 2
                    ):
                        action = slow_saved.get("action")
                        if slow_gate_with_selection:
                            # Gated mode (follow SkillReflection): re-apply the
                            # guidance to current_skill only when it was accepted.
                            if action in {"accept", "accept_new_best"}:
                                current_skill = replace_slow_update_field(
                                    current_skill,
                                    slow_saved["slow_update_content"],
                                )
                        elif action in {
                            "accept", "accept_new_best", "force_accept",
                        }:
                            # Force-accept mode: re-apply guidance to
                            # current_skill only. best_skill must remain a
                            # faithful snapshot of the val-best step and must
                            # NOT receive force-injected slow-update content.
                            current_skill = replace_slow_update_field(
                                current_skill, slow_saved["slow_update_content"],
                            )
                elif epoch == 1:
                    # Epoch 1: inject empty placeholder
                    os.makedirs(slow_dir, exist_ok=True)
                    current_skill = inject_empty_slow_update_field(current_skill)
                    current_origin = f"slow_update_placeholder_epoch_{epoch:02d}"
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill)
                    with open(slow_done_path, "w") as f:
                        json.dump({"action": "inject_placeholder", "epoch": epoch}, f, indent=2)
                    _persist_runtime_state(global_step)
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"injected empty placeholder"
                    )
                else:
                    # Epoch 2+: longitudinal comparison
                    os.makedirs(slow_dir, exist_ok=True)
                    print(
                        f"\n  {'='*60}\n"
                        f"  SLOW UPDATE — Epoch {epoch} "
                        f"(comparing epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    # 1. Get skill from last step of previous epoch
                    prev_epoch_records = [
                        h for h in history if h.get("epoch") == epoch - 1
                    ]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_skill = _load_skill(out_root, prev_epoch_last_step)

                    # 2. Sample items from train set
                    slow_n = cfg.get("slow_update_samples", 20)
                    slow_seed = seed + epoch * 2000
                    if dataloader is not None:
                        slow_batch = dataloader.build_train_batch(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                        slow_env = adapter.build_env_from_batch(
                            slow_batch, out_root=out_root,
                        )
                    else:
                        slow_env = adapter.build_train_env(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                    slow_items = list(slow_env) if hasattr(slow_env, "__iter__") else slow_env
                    print(f"    [slow update] sampled {len(slow_items)} train items (seed={slow_seed})")

                    # 3. Rollout with both skills
                    t_slow = time.time()
                    prev_rollout_dir = os.path.join(slow_dir, "rollout_prev")
                    curr_rollout_dir = os.path.join(slow_dir, "rollout_curr")
                    results_prev = adapter.rollout(slow_env, prev_skill, prev_rollout_dir)
                    results_curr = adapter.rollout(slow_env, current_skill, curr_rollout_dir)

                    prev_hard, _ = compute_score(results_prev)
                    curr_hard, _ = compute_score(results_curr)
                    print(
                        f"    [slow update] prev epoch hard={prev_hard:.4f}  "
                        f"curr epoch hard={curr_hard:.4f}"
                    )

                    # 4. Build and save structured comparison pairs
                    comparison_pairs, all_comparison_pairs = _build_longitudinal_pairs(
                        adapter=adapter,
                        dataloader=dataloader,
                        prev_skill=prev_skill,
                        curr_skill=current_skill,
                        initial_items=slow_items,
                        initial_prev_results=results_prev,
                        initial_curr_results=results_curr,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        policy=longitudinal_pair_policy,
                        target_n=slow_n,
                        seed=slow_seed,
                        out_root=out_root,
                    )
                    epoch_comparison_pairs = comparison_pairs
                    if all_comparison_pairs is not comparison_pairs:
                        save_comparison_pairs(
                            all_comparison_pairs,
                            os.path.join(slow_dir, "comparison_pairs_all.json"),
                        )
                    save_comparison_pairs(
                        comparison_pairs,
                        os.path.join(slow_dir, "comparison_pairs.json"),
                    )
                    n_regressed = sum(1 for p in comparison_pairs if p["category"] == "regressed")
                    n_improved = sum(1 for p in comparison_pairs if p["category"] == "improved")
                    n_persist = sum(1 for p in comparison_pairs if p["category"] == "persistent_fail")
                    n_stable = sum(1 for p in comparison_pairs if p["category"] == "stable_success")
                    print(
                        f"    [slow update] comparison: "
                        f"regressed={n_regressed} improved={n_improved} "
                        f"persistent_fail={n_persist} stable_success={n_stable} "
                        f"policy={longitudinal_pair_policy} "
                        f"kept={len(comparison_pairs)}/{len(all_comparison_pairs)}"
                    )

                    # 5. Extract previous slow update guidance for reflection
                    existing_guidance = extract_slow_update_field(current_skill)

                    # 6. Optimizer analysis (with reflection on previous guidance)
                    slow_result = run_slow_update(
                        current_skill,
                        results_prev,
                        results_curr,
                        slow_items,
                        prev_skill=prev_skill,
                        prev_slow_update_content=existing_guidance,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        comparison_pairs=comparison_pairs,
                    )
                    slow_time = round(time.time() - t_slow, 1)

                    if slow_result and slow_result.get("slow_update_content"):
                        slow_candidate = replace_slow_update_field(
                            current_skill, slow_result["slow_update_content"],
                        )
                        slow_candidate_hash = skill_hash(slow_candidate)
                        with open(os.path.join(slow_dir, "candidate_skill.md"), "w") as f:
                            f.write(slow_candidate)
                        slow_result["time_s"] = slow_time
                        slow_result["prev_hard"] = prev_hard
                        slow_result["curr_hard"] = curr_hard
                        slow_result["candidate_hash"] = slow_candidate_hash
                        slow_result["update_origin"] = "slow_update_momentum"
                        slow_result["update_target"] = (
                            "Address longitudinal regressions and persistent failures "
                            "observed across adjacent epochs."
                        )

                        # Slow update acceptance — two modes selected via
                        # `optimizer.slow_update_gate_with_selection`.
                        if slow_gate_with_selection:
                            # ── Gated mode (follow SkillReflection) ──────────
                            # Evaluate the slow-update candidate on the
                            # selection set and accept/reject via the same
                            # validation gate used for step-level updates.
                            slow_candidate_selection_results: list[dict] | None = None
                            if slow_candidate_hash in sel_cache and (
                                not paired_audit_enabled
                                or slow_candidate_hash in sel_result_cache
                            ):
                                slow_sel_hard, slow_sel_soft = sel_cache[
                                    slow_candidate_hash
                                ]
                                slow_candidate_selection_results = sel_result_cache.get(
                                    slow_candidate_hash
                                )
                                print(
                                    f"    [slow gate] cache hit: "
                                    f"hard={slow_sel_hard:.4f}"
                                )
                            else:
                                sel_env, sel_n = _build_eval_env(
                                    split="valid_seen",
                                    env_num=cfg["sel_env_num"],
                                    seed=seed,
                                )
                                print(f"    [slow gate] selection items={sel_n}")
                                slow_eval_dir = os.path.join(
                                    slow_dir, "selection_eval",
                                )
                                slow_eval_results = adapter.rollout(
                                    sel_env, slow_candidate, slow_eval_dir,
                                )
                                slow_sel_hard, slow_sel_soft = compute_score(
                                    slow_eval_results
                                )
                                sel_cache[slow_candidate_hash] = (
                                    slow_sel_hard, slow_sel_soft,
                                )
                                sel_result_cache[slow_candidate_hash] = slow_eval_results
                                slow_candidate_selection_results = slow_eval_results

                            slow_paired_audit = None
                            slow_paired_risk = None
                            if paired_audit_enabled:
                                if (
                                    current_selection_results is None
                                    or slow_candidate_selection_results is None
                                ):
                                    raise RuntimeError(
                                        "paired slow promotion requires per-item results"
                                    )
                                slow_paired_audit = build_paired_promotion_audit(
                                    current_selection_results,
                                    slow_candidate_selection_results,
                                    require_hard_gain=paired_require_hard_gain,
                                )
                                slow_result["paired_promotion"] = slow_paired_audit
                                if paired_risk_sensitive:
                                    slow_paired_risk = (
                                        _build_paired_risk_decision(
                                            slow_paired_audit,
                                            harm_weight=paired_harm_weight,
                                            min_margin=paired_risk_min_margin,
                                        )
                                    )
                                    slow_result["paired_risk"] = slow_paired_risk
                                with open(
                                    os.path.join(
                                        slow_dir, "paired_promotion_audit.json"
                                    ),
                                    "w",
                                ) as f:
                                    json.dump(
                                        slow_paired_audit,
                                        f,
                                        ensure_ascii=False,
                                        indent=2,
                                    )

                            slow_gate = evaluate_gate(
                                candidate_skill=slow_candidate,
                                cand_hard=slow_sel_hard,
                                current_skill=current_skill,
                                current_score=current_score,
                                best_skill=best_skill,
                                best_score=best_score,
                                best_step=best_step,
                                global_step=global_step,
                                cand_soft=slow_sel_soft,
                                metric=gate_metric,
                                mixed_weight=gate_mixed_weight,
                                use_semantic_density=use_semantic_density,
                                semantic_density_weight=semantic_density_weight,
                                leading_words=leading_words,
                            )
                            if (
                                slow_gate.action in {"accept", "accept_new_best"}
                                and (
                                    (
                                        paired_non_degrading
                                        and slow_paired_audit is not None
                                        and not slow_paired_audit["accepted"]
                                    )
                                    or (
                                        paired_risk_sensitive
                                        and slow_paired_risk is not None
                                        and not slow_paired_risk["accepted"]
                                    )
                                )
                            ):
                                slow_gate = GateResult(
                                    action="reject",
                                    current_skill=current_skill,
                                    current_score=current_score,
                                    best_skill=best_skill,
                                    best_score=best_score,
                                    best_step=best_step,
                                )
                                slow_result["paired_gate_override"] = True
                            slow_result["selection_hard"] = slow_sel_hard
                            slow_result["selection_soft"] = slow_sel_soft
                            slow_result["action"] = slow_gate.action
                            prev_current = current_score
                            prev_best = best_score
                            current_skill = slow_gate.current_skill
                            current_score = slow_gate.current_score
                            best_skill = slow_gate.best_skill
                            best_score = slow_gate.best_score
                            best_step = slow_gate.best_step
                            if slow_gate.action in {"accept", "accept_new_best"}:
                                current_origin = (
                                    f"slow_update_epoch_{epoch:02d}"
                                )
                                if slow_candidate_selection_results is not None:
                                    current_selection_results = (
                                        slow_candidate_selection_results
                                    )
                            if slow_gate.action == "accept_new_best":
                                best_origin = current_origin
                                if slow_candidate_selection_results is not None:
                                    best_selection_results = (
                                        slow_candidate_selection_results
                                    )
                                print(
                                    f"    [slow gate] ACCEPT (new best) "
                                    f"hard={slow_sel_hard:.4f} > "
                                    f"prev best {prev_best:.4f}"
                                )
                            elif slow_gate.action == "accept":
                                print(
                                    f"    [slow gate] ACCEPT "
                                    f"hard={slow_sel_hard:.4f} > "
                                    f"current={prev_current:.4f}"
                                )
                            else:
                                print(
                                    f"    [slow gate] REJECT "
                                    f"hard={slow_sel_hard:.4f} <= "
                                    f"current={current_score:.4f}"
                                )
                            print(
                                f"    [slow update] guidance written "
                                f"({len(slow_result['slow_update_content'])} "
                                f"chars), {slow_time}s"
                            )
                        else:
                            # ── Force-accept mode (default) ──────────────────
                            # The epoch-level longitudinal guidance is injected
                            # into current_skill ONLY, so training continues
                            # with the accumulated slow memory. best_skill is
                            # left untouched: it must remain a faithful snapshot
                            # of the val-best step (which may be a pre-slow step
                            # such as S_0 carrying no slow_update field at all).
                            slow_content = slow_result["slow_update_content"]
                            current_skill = replace_slow_update_field(
                                current_skill, slow_content,
                            )
                            # Update caches so downstream steps use the
                            # slow-update-injected skill for hashing.
                            slow_candidate_hash = skill_hash(current_skill)
                            sel_cache[slow_candidate_hash] = (current_score, 0.0)

                            slow_result["action"] = "force_accept"
                            current_origin = f"slow_update_epoch_{epoch:02d}"

                            print(
                                f"    [slow update] force-injected into "
                                f"current only "
                                f"({len(slow_content)} chars), "
                                f"{slow_time}s"
                            )
                    else:
                        slow_result = slow_result or {}
                        slow_result["action"] = "no_content"
                        slow_result["time_s"] = slow_time
                        print(
                            f"    [slow update] no guidance produced, "
                            f"{slow_time}s"
                        )

                    # 5. Save
                    with open(slow_done_path, "w") as f:
                        json.dump(slow_result, f, indent=2, ensure_ascii=False)
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill)
                    _persist_runtime_state(global_step)

                    print(
                        f"\n  [SLOW UPDATE epoch {epoch} done] "
                        f"current={current_score:.4f} best={best_score:.4f}"
                    )

            # ── META SKILL (end of epoch, optimizer-side memory) ─────────
            use_meta_skill = cfg.get("use_meta_skill", False)
            if use_meta_skill:
                meta_skill_dir = os.path.join(out_root, "meta_skill", f"epoch_{epoch:02d}")
                meta_skill_done_path = os.path.join(meta_skill_dir, "meta_skill_result.json")
                os.makedirs(meta_skill_dir, exist_ok=True)

                if os.path.exists(meta_skill_done_path):
                    print(f"\n  [META SKILL epoch {epoch}] resumed — already done")
                elif epoch == 1:
                    with open(meta_skill_done_path, "w") as f:
                        json.dump(
                            {"action": "skip_first_epoch", "epoch": epoch},
                            f, indent=2, ensure_ascii=False,
                        )
                    print(f"\n  [META SKILL epoch {epoch}] skipped — first epoch")
                else:
                    print(
                        f"\n  {'='*60}\n"
                        f"  META SKILL — Epoch {epoch} "
                        f"(optimizer memory from epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    prev_epoch_records = [h for h in history if h.get("epoch") == epoch - 1]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_skill = _load_skill(out_root, prev_epoch_last_step)
                    prev_meta_skill = _load_meta_skill_content(out_root, epoch - 1)

                    if epoch_comparison_pairs is None:
                        meta_n = cfg.get("slow_update_samples", 20)
                        meta_seed = seed + epoch * 2000
                        if dataloader is not None:
                            meta_batch = dataloader.build_train_batch(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                            meta_env = adapter.build_env_from_batch(
                                meta_batch, out_root=out_root,
                            )
                        else:
                            meta_env = adapter.build_train_env(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                        meta_items = list(meta_env) if hasattr(meta_env, "__iter__") else meta_env
                        prev_rollout_dir = os.path.join(meta_skill_dir, "rollout_prev")
                        curr_rollout_dir = os.path.join(meta_skill_dir, "rollout_curr")
                        results_prev = adapter.rollout(meta_env, prev_skill, prev_rollout_dir)
                        results_curr = adapter.rollout(meta_env, epoch_last_step_skill, curr_rollout_dir)
                        epoch_comparison_pairs, all_meta_comparison_pairs = _build_longitudinal_pairs(
                            adapter=adapter,
                            dataloader=dataloader,
                            prev_skill=prev_skill,
                            curr_skill=epoch_last_step_skill,
                            initial_items=meta_items,
                            initial_prev_results=results_prev,
                            initial_curr_results=results_curr,
                            prev_rollout_dir=prev_rollout_dir,
                            curr_rollout_dir=curr_rollout_dir,
                            policy=longitudinal_pair_policy,
                            target_n=meta_n,
                            seed=meta_seed,
                            out_root=out_root,
                        )
                        if all_meta_comparison_pairs is not epoch_comparison_pairs:
                            save_comparison_pairs(
                                all_meta_comparison_pairs,
                                os.path.join(meta_skill_dir, "comparison_pairs_all.json"),
                            )
                        save_comparison_pairs(
                            epoch_comparison_pairs,
                            os.path.join(meta_skill_dir, "comparison_pairs.json"),
                        )
                        meta_counts = _pair_category_counts(epoch_comparison_pairs)
                        print(
                            f"    [meta skill] comparison: "
                            f"regressed={meta_counts.get('regressed', 0)} "
                            f"improved={meta_counts.get('improved', 0)} "
                            f"persistent_fail={meta_counts.get('persistent_fail', 0)} "
                            f"stable_success={meta_counts.get('stable_success', 0)} "
                            f"policy={longitudinal_pair_policy} "
                            f"kept={len(epoch_comparison_pairs)}/{len(all_meta_comparison_pairs)}"
                        )

                    t_meta_skill = time.time()
                    meta_skill_result = run_meta_skill(
                        prev_skill=prev_skill,
                        curr_skill=epoch_last_step_skill,
                        comparison_pairs=epoch_comparison_pairs or [],
                        prev_meta_skill_content=prev_meta_skill,
                    )
                    meta_skill_time = round(time.time() - t_meta_skill, 1)

                    if meta_skill_result and meta_skill_result.get("meta_skill_content"):
                        meta_skill_result["time_s"] = meta_skill_time
                        meta_skill_result["action"] = "write_meta_skill"
                        print(
                            f"    [meta skill] memory written "
                            f"({len(meta_skill_result['meta_skill_content'])} chars), "
                            f"{meta_skill_time}s"
                        )
                    else:
                        meta_skill_result = meta_skill_result or {}
                        meta_skill_result["time_s"] = meta_skill_time
                        meta_skill_result["action"] = "no_content"
                        print(f"    [meta skill] no memory produced, {meta_skill_time}s")

                    with open(meta_skill_done_path, "w") as f:
                        json.dump(meta_skill_result, f, indent=2, ensure_ascii=False)

        # ── Save best skill ──────────────────────────────────────────────
        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
            f.write(best_skill)
        _persist_runtime_state(global_step)
        print(
            f"\n  [done] best skill from step {best_step}, "
            f"score={best_score:.4f}"
        )

        # ── Final test evaluation (valid_unseen) ─────────────────────────
        baseline_test_hard = None
        baseline_test_soft = None
        test_hard = None
        test_soft = None
        final_test_hard = None
        final_test_soft = None
        final_selection_hard = None
        final_selection_soft = None

        if cfg["eval_test"]:
            task_types = adapter.get_task_types()

            # ── Final skill validation (valid_seen) + best promotion ─────
            # The final (last) skill may carry an epoch-end slow_update that
            # was force-injected WITHOUT a val pass (use_gate=false or
            # slow_update_gate_with_selection=false), so it never competed for
            # best. Run one real val on the final skill; if its gate score
            # beats the incumbent best, PROMOTE it to best so that best is the
            # true val-argmax over all skills (including the final slow_update).
            # When final == best, reuse the existing val score (no rollout).
            try:
                if skill_hash(current_skill) == skill_hash(best_skill):
                    final_selection_hard, final_selection_soft = best_score, None
                    print(
                        "\n  [final skill == best skill] "
                        f"final_selection_hard={best_score:.4f} (reused)"
                    )
                else:
                    fval_env, fval_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    fval_dir = os.path.join(out_root, "final_selection_eval")
                    fval_results = adapter.rollout(fval_env, current_skill, fval_dir)
                    final_selection_hard, final_selection_soft = compute_score(fval_results)
                    final_gate_score = select_gate_score(
                        final_selection_hard, final_selection_soft,
                        gate_metric, gate_mixed_weight,
                        skill_content=current_skill,
                        use_semantic_density=use_semantic_density,
                        semantic_density_weight=semantic_density_weight,
                        leading_words=leading_words,
                    )
                    print(
                        f"\n  [final skill val] items={fval_n} "
                        f"final_selection_hard={final_selection_hard:.4f} "
                        f"gate={final_gate_score:.4f} "
                        f"(best={best_score:.4f})"
                    )
                    if final_gate_score > best_score:
                        # Promote: the final (slow-updated) skill is val-better
                        # than the incumbent best. Make it the new best so the
                        # subsequent BEST-skill test rollout evaluates it and
                        # best/final test scores coincide.
                        print(
                            f"  [promote] final {final_gate_score:.4f} > "
                            f"best {best_score:.4f} → final becomes new best "
                            f"(step {global_step}, origin {current_origin})"
                        )
                        best_skill = current_skill
                        best_score = final_gate_score
                        best_step = global_step
                        best_origin = current_origin
                        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                            f.write(best_skill)
                        _persist_runtime_state(global_step)
            except Exception as _e:  # noqa: BLE001
                final_selection_hard = None
                final_selection_soft = None
                print(f"\n  [final skill val FAILED: {_e!r}]")

            # Baseline: S_0 on test set (valid_unseen)
            print(f"\n{'='*60}")
            print("  BASELINE TEST — evaluate initial skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env, test_n = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n}")
            baseline_test_dir = os.path.join(out_root, "test_eval_baseline")
            os.makedirs(baseline_test_dir, exist_ok=True)
            baseline_test_results = adapter.rollout(test_env, skill_init, baseline_test_dir)
            baseline_test_hard, baseline_test_soft = compute_score(baseline_test_results)
            baseline_buckets = _compute_task_type_buckets(baseline_test_results, task_types)
            print("\n  === Baseline Test Results (S_0) ===")
            for task_type in task_types + ["overall"]:
                b = baseline_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(baseline_test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in baseline_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Best skill on test set
            print(f"\n{'='*60}")
            print("  BEST SKILL TEST — evaluate best skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env2, test_n2 = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n2}")
            test_dir = os.path.join(out_root, "test_eval")
            os.makedirs(test_dir, exist_ok=True)
            test_results = adapter.rollout(test_env2, best_skill, test_dir)
            test_hard, test_soft = compute_score(test_results)
            best_buckets = _compute_task_type_buckets(test_results, task_types)
            print("\n  === Best Skill Test Results ===")
            for task_type in task_types + ["overall"]:
                b = best_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in best_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Final skill (last skill in trajectory) on test set.
            # Distinct from best_skill: with use_gate=False every candidate is
            # force-accepted so the final skill is whatever the last step
            # produced; with use_gate=True it is the last accepted skill, which
            # may differ from the best-on-val skill. We always evaluate it so
            # every run reports baseline / best-on-val / final on test.
            # Guarded so a failure here never prevents summary.json from being
            # written (the orchestrator's post-hoc safety net fills it in).
            try:
                if skill_hash(current_skill) == skill_hash(best_skill):
                    # Final == best: reuse results, skip a redundant rollout.
                    final_test_hard, final_test_soft = test_hard, test_soft
                    final_test_dir = os.path.join(out_root, "test_eval_final")
                    os.makedirs(final_test_dir, exist_ok=True)
                    with open(os.path.join(final_test_dir, "summary.json"), "w") as f:
                        json.dump(
                            {
                                k: {
                                    "total": b["total"],
                                    "hard_acc": b["hard"] / max(b["total"], 1),
                                }
                                for k, b in best_buckets.items()
                            },
                            f, indent=2, ensure_ascii=False,
                        )
                    print(
                        "\n  [final skill == best skill] "
                        f"final_test_hard={final_test_hard:.4f} (reused)"
                    )
                else:
                    print(f"\n{'='*60}")
                    print("  FINAL SKILL TEST — evaluate last skill on Test set (valid_unseen)")
                    print(f"{'='*60}")
                    test_env3, test_n3 = _build_eval_env(
                        split="valid_unseen",
                        env_num=cfg["test_env_num"],
                        seed=seed,
                    )
                    print(f"  Test items: {test_n3}")
                    final_test_dir = os.path.join(out_root, "test_eval_final")
                    os.makedirs(final_test_dir, exist_ok=True)
                    final_test_results = adapter.rollout(test_env3, current_skill, final_test_dir)
                    final_test_hard, final_test_soft = compute_score(final_test_results)
                    final_buckets = _compute_task_type_buckets(final_test_results, task_types)
                    print("\n  === Final Skill Test Results ===")
                    for task_type in task_types + ["overall"]:
                        b = final_buckets.get(task_type, {"total": 0, "hard": 0})
                        t = max(b["total"], 1)
                        print(
                            f"    {task_type:<40s}: "
                            f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                        )
                    with open(os.path.join(final_test_dir, "summary.json"), "w") as f:
                        json.dump(
                            {
                                k: {
                                    "total": b["total"],
                                    "hard_acc": b["hard"] / max(b["total"], 1),
                                }
                                for k, b in final_buckets.items()
                            },
                            f, indent=2, ensure_ascii=False,
                        )
            except Exception as _e:  # noqa: BLE001
                final_test_hard = None
                final_test_soft = None
                print(f"\n  [final skill test FAILED: {_e!r}] "
                      "— will be filled by post-hoc eval")

            # Comparison
            delta_hard = (test_hard or 0) - (baseline_test_hard or 0)
            print(f"\n  === Improvement vs baseline (init S_0) ===")
            print(
                f"    [2] best-on-val hard: {baseline_test_hard:.4f} -> {test_hard:.4f}  "
                f"(delta={delta_hard:+.4f})"
            )
            if final_test_hard is not None:
                final_delta_hard = (final_test_hard or 0) - (baseline_test_hard or 0)
                print(
                    f"    [3] final/last  hard: {baseline_test_hard:.4f} -> {final_test_hard:.4f}  "
                    f"(delta={final_delta_hard:+.4f})"
                )

        # ── Global summary ───────────────────────────────────────────────
        total_wall = time.time() - t_loop_start
        n_accept = sum(1 for h in history if "accept" in h.get("action", ""))
        n_reject = sum(1 for h in history if h.get("action") == "reject")
        n_skip = sum(1 for h in history if h.get("action") == "skip_no_patches")

        token_summary = get_token_summary()

        # Epoch-level statistics
        epoch_stats = []
        for e in range(1, num_epochs + 1):
            epoch_records = [h for h in history if h.get("epoch") == e]
            if epoch_records:
                epoch_stats.append({
                    "epoch": e,
                    "steps": [h["step"] for h in epoch_records],
                    "accepts": sum(1 for h in epoch_records if "accept" in h.get("action", "")),
                    "rejects": sum(1 for h in epoch_records if h.get("action") == "reject"),
                    "skips": sum(1 for h in epoch_records if h.get("action") == "skip_no_patches"),
                    "best_score_at_epoch_end": epoch_records[-1].get("best_score", 0.0),
                    "current_score_at_epoch_end": epoch_records[-1].get("current_score", 0.0),
                })

        summary = {
            "version": "textualrl-0.1.0",
            "config": _redact_cfg(cfg),
            "baseline_selection_hard": sel_cache.get(
                skill_hash(skill_init), (None, None),
            )[0],
            "best_selection_hard": best_score,
            "final_selection_hard": final_selection_hard,
            "final_selection_soft": final_selection_soft,
            "best_step": best_step,
            "current_origin": current_origin,
            "best_origin": best_origin,
            "total_steps": len(history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_skips": n_skip,
            "epoch_stats": epoch_stats,
            "baseline_test_hard": baseline_test_hard,
            "baseline_test_soft": baseline_test_soft,
            "test_hard": test_hard,
            "test_soft": test_soft,
            "final_test_hard": final_test_hard,
            "final_test_soft": final_test_soft,
            "test_delta_hard": (
                (test_hard or 0) - (baseline_test_hard or 0)
                if test_hard is not None
                else None
            ),
            "final_test_delta_hard": (
                (final_test_hard or 0) - (baseline_test_hard or 0)
                if final_test_hard is not None
                else None
            ),
            "total_wall_time_s": round(total_wall, 1),
            "token_summary": token_summary,
        }
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("  Final Summary")
        print(f"{'='*60}")
        print(
            f"  steps={len(history)} accept={n_accept} "
            f"reject={n_reject} skip={n_skip}"
        )
        print(f"  best_score={best_score:.4f} (step {best_step})  wall={total_wall:.0f}s")
        if epoch_stats:
            for es in epoch_stats:
                print(
                    f"    epoch {es['epoch']}: accept={es['accepts']} reject={es['rejects']} "
                    f"best={es['best_score_at_epoch_end']:.4f}"
                )
        if baseline_test_hard is not None:
            print("\n  === TEST scores (3 skills, split=valid_unseen) ===")
            print(
                f"    [1] init/baseline (S_0)          : "
                f"test_hard={baseline_test_hard:.4f}"
            )
        if test_hard is not None:
            print(
                f"    [2] best-on-val (step {best_step})".ljust(37)
                + f": test_hard={test_hard:.4f} test_soft={test_soft:.4f}"
            )
        if final_test_hard is not None:
            print(
                f"    [3] final/last skill             : "
                f"test_hard={final_test_hard:.4f} test_soft={final_test_soft:.4f}"
            )
        if token_summary.get("_total"):
            t = token_summary["_total"]
            print(
                f"  total tokens: {t['total_tokens']:,} "
                f"(prompt={t['prompt_tokens']:,} "
                f"completion={t['completion_tokens']:,} "
                f"calls={t['calls']})"
            )

        return summary
