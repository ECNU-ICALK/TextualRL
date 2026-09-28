"""ReflACT core Reflect engine -- minibatch trajectory analysis.

Provides environment-agnostic minibatch trajectory analysis: instead of
analyzing each trajectory independently, trajectories are grouped into
minibatches of size M and analyzed together -- analogous to minibatch SGD
vs per-sample SGD in neural network training.

Two-level prompt priority system:

1. **Custom prompt** (adapter returns non-None) -- used as-is.
2. **Generic default prompt** (adapter returns None) -- built-in defaults
   that work for any environment without configuration.

Public API
----------
- :func:`fmt_trajectory`               -- format one conversation into text
- :func:`fmt_minibatch_trajectories`   -- format multiple trajectories for batch analysis
- :func:`run_error_analyst_minibatch`   -- one optimizer call for a group of failures
- :func:`run_success_analyst_minibatch` -- one optimizer call for a group of successes
- :func:`run_minibatch_reflect`         -- full reflect stage dispatcher
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.model import chat_optimizer
from skillopt.gradient.context_batching import run_context_bounded_analyst
from skillopt.optimizer.cross_group import (
    ANALYST_SUFFIX as CROSS_GROUP_ANALYST_SUFFIX,
    attach_cross_group_cards,
    cross_group_evidence_enabled,
)
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.group_relative import (
    annotate_analyst_patch,
    augment_group_relative_analyst_prompt,
    format_group_relative_analyst_context,
    get_group_relative_edit_credit_config,
    is_group_relative_edit_credit_enabled,
)
from skillopt.optimizer.skill_aware import (
    augment_error_prompt,
    augment_success_prompt,
    extract_appendix_notes,
    get_skill_aware_appendix_source,
    is_skill_aware_enabled,
)
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    truncate_payload,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


# ── Trajectory formatting ────────────────────────────────────────────────────


def _clip_text(value, limit: int | None = None) -> str:
    """Render optional trajectory fields. Truncation is disabled: the optimizer
    is given the full content so it can see exactly what the agent saw/did.

    ``limit`` is accepted for backward compatibility but ignored.
    """
    if value is None:
        return ""
    return str(value)


def fmt_trajectory(
    conversation: list[dict],
    max_chars: int | None = None,
) -> str:
    """Format a conversation list into analyst-readable text.

    Accepts two common formats:

    1. Tool-call records:   ``{"type": "tool_call", "cmd": ..., "obs": ...}``
    2. Step records:        ``{"step": N, "action": ..., "env_feedback": ..., "reasoning": ...}``

    Any other dict is rendered via its ``"content"`` key.
    """
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {_clip_text(item)}")
            continue
        if item.get("type") == "tool_call":
            cmd = _clip_text(item.get("cmd"))
            obs = _clip_text(item.get("obs"))
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = _clip_text(item.get("reasoning"))
            action = _clip_text(item.get("action"))
            feedback = _clip_text(item.get("env_feedback"))
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif item.get("role") == "system":
            # Post-execution verification / enrichment info
            msg = _clip_text(item.get("content"))
            lines.append(f"[verification] {msg}")
        else:
            msg = _clip_text(item.get("content"))
            role = item.get("role", "agent")
            lines.append(f"[{role}] {msg}")

    return "\n".join(lines)


# ── Minibatch trajectory formatting ──────────────────────────────────────────


def fmt_minibatch_trajectories(
    items: list[dict],
    prediction_dir: str,
) -> str:
    """Format multiple trajectories for minibatch analyst consumption.

    Each item is a rollout result dict with ``"id"``, ``"task_description"``,
    ``"task_type"``, ``"fail_reason"``, etc.  Reads ``conversation.json``
    for each and formats them together with trajectory headers.

    If available, includes the spreadsheet preview and target system prompt
    so the analyst can see what the agent saw.

    Parameters
    ----------
    items : list[dict]
        Rollout result dicts belonging to one minibatch.
    prediction_dir : str
        Path to ``predictions/`` directory containing per-task
        ``<task_id>/conversation.json`` files.

    Returns
    -------
    str
        Formatted text with all trajectories separated by ``---``.
    """
    parts: list[str] = []
    for idx, item in enumerate(items, 1):
        tid = str(item["id"])
        conv_path = os.path.join(prediction_dir, tid, "conversation.json")
        if not os.path.exists(conv_path):
            continue
        with open(conv_path) as f:
            conversation = json.load(f)
        if not conversation:
            continue

        traj_text = fmt_trajectory(conversation)
        stratified_route = str(
            item.get("_outcome_stratified_route") or ""
        )
        stratified_primary = bool(
            item.get("_outcome_stratified_task_primary", False)
        )
        if stratified_route:
            group_id = str(item.get("rollout_group_id") or tid)
            unique_index = int(
                item.get("_outcome_stratified_unique_index", 1) or 1
            )
            unique_count = int(
                item.get("_outcome_stratified_unique_count", 1) or 1
            )
            multiplicity = int(
                item.get("_outcome_stratified_multiplicity", 1) or 1
            )
            header = (
                f"### Task Block {group_id!r}\n"
                f"#### Unique Trajectory {unique_index}/{unique_count} "
                f"(id={tid}, multiplicity={multiplicity})\n"
                f"Outcome: hard={int(bool(item.get('hard')))}; "
                f"soft={float(item.get('soft', 0.0) or 0.0):.4f}\n"
            )
            if stratified_primary:
                header += (
                    f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
                    f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
                )
        else:
            header = (
                f"### Trajectory {idx} (id={tid})\n"
                f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
                f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
            )
        fail_reason = item.get("fail_reason", "")
        if fail_reason:
            header += f"Failure reason: {fail_reason}\n"
        header += f"Steps: {item.get('n_turns', '?')}\n"

        group = item.get("same_task_group")
        if isinstance(group, dict) and (
            not stratified_route or stratified_primary
        ):
            header += (
                "\n#### Same-Task Rollout Group\n"
                f"Source task_id: {item.get('rollout_group_id', tid)}\n"
                f"Status: {group.get('status', 'unknown')}\n"
                f"Successes: {group.get('success_count', 0)}/"
                f"{group.get('completed_count', 0)} "
                f"(expected {group.get('expected_count', '?')})\n"
                f"Group hard mean: {group.get('hard_mean', 0.0)}\n"
                f"This sample's group-relative hard score: "
                f"{item.get('group_relative_hard', 0.0)}\n"
                "Sibling outcomes from the same task and same skill snapshot:\n"
            )
            for outcome in group.get("outcomes", []):
                answer = str(outcome.get("predicted_answer", ""))
                reason = str(outcome.get("fail_reason", ""))
                response = str(outcome.get("response_excerpt", ""))
                header += (
                    f"- sample {outcome.get('rollout_index', '?')}: "
                    f"hard={outcome.get('hard', 0)} "
                    f"soft={float(outcome.get('soft', 0.0) or 0.0):.4f}; "
                    f"answer={answer!r}"
                )
                if reason:
                    header += f"; failure={reason}"
                if response:
                    header += f"; response={response!r}"
                header += "\n"

            # For a mixed group, expose one full counterfactual sibling rather
            # than asking the analyst to infer causality from final answers.
            # This is the additional information K>1 is meant to provide.
            if (
                is_group_relative_edit_credit_enabled()
                and group.get("status") == "mixed"
                and not stratified_route
            ):
                current_success = bool(item.get("hard", 0))
                opposing = [
                    outcome
                    for outcome in group.get("outcomes", [])
                    if bool(outcome.get("hard", 0)) != current_success
                ]
                if opposing:
                    opposing.sort(
                        key=lambda outcome: (
                            -float(outcome.get("soft", 0.0) or 0.0)
                            if not current_success
                            else float(outcome.get("soft", 0.0) or 0.0),
                            int(outcome.get("rollout_index", 0) or 0),
                        )
                    )
                    counterpart = opposing[0]
                    counterpart_id = str(counterpart.get("id") or "")
                    counterpart_path = os.path.join(
                        prediction_dir,
                        counterpart_id,
                        "conversation.json",
                    )
                    if counterpart_id and os.path.exists(counterpart_path):
                        with open(counterpart_path) as f:
                            counterpart_conversation = json.load(f)
                        header += (
                            "\n#### Full Counterfactual Sibling Trajectory\n"
                            f"id={counterpart_id}; "
                            f"hard={counterpart.get('hard', 0)}; "
                            f"soft={float(counterpart.get('soft', 0.0) or 0.0):.4f}\n"
                            "Compare this trajectory with the primary trajectory "
                            "below and identify their first causal behavioral "
                            "divergence.\n"
                            f"{fmt_trajectory(counterpart_conversation)}\n"
                        )

        reference_text = str(item.get("reference_text") or "").strip()
        if reference_text and (not stratified_route or stratified_primary):
            header += (
                f"\n#### Hidden Reference\n"
                f"{reference_text}\n"
            )

        # ── Append target context (what the agent saw) ──────────────
        target_prompt = item.get("target_system_prompt", "")
        if not target_prompt:
            prompt_path = os.path.join(prediction_dir, tid, "target_system_prompt.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    target_prompt = f.read()
        if target_prompt and (not stratified_route or stratified_primary):
            header += (
                f"\n#### Target System Prompt\n"
                f"{target_prompt}\n"
            )

        user_prompt = item.get("target_user_prompt", "")
        if not user_prompt:
            user_prompt_path = os.path.join(prediction_dir, tid, "target_user_prompt.txt")
            if os.path.exists(user_prompt_path):
                with open(user_prompt_path) as f:
                    user_prompt = f.read()
        if user_prompt and (not stratified_route or stratified_primary):
            header += (
                f"\n#### Target User Prompt\n"
                f"{user_prompt}\n"
            )

        if (
            os.environ.get("REFLACT_CODEX_TRACE_TO_OPTIMIZER", "0") == "1"
            and (not stratified_route or stratified_primary)
        ):
            codex_trace_summary = item.get("codex_trace_summary", "")
            if not codex_trace_summary:
                codex_trace_summary_path = os.path.join(prediction_dir, tid, "codex_trace_summary.txt")
                if os.path.exists(codex_trace_summary_path):
                    with open(codex_trace_summary_path) as f:
                        codex_trace_summary = f.read()
            if codex_trace_summary:
                header += (
                    f"\n#### Codex Trace Summary\n"
                    f"{codex_trace_summary}\n"
                )

        codex_probe_trace_steps = str(item.get("codex_probe_trace_steps") or "").strip()
        if codex_probe_trace_steps and (
            not stratified_route or stratified_primary
        ):
            header += (
                f"\n#### Codex Trace Steps\n"
                f"{codex_probe_trace_steps}\n"
            )

        preview = item.get("spreadsheet_preview", "")
        if not preview:
            preview_path = os.path.join(prediction_dir, tid, "spreadsheet_preview.txt")
            if os.path.exists(preview_path):
                with open(preview_path) as f:
                    preview = f.read()
        if preview and (not stratified_route or stratified_primary):
            header += (
                f"\n#### Spreadsheet Preview\n"
                f"{preview}\n"
            )

        parts.append(header + "\n" + traj_text)

    return "\n\n---\n\n".join(parts)


def _select_same_task_group_representatives(
    results: list[dict],
    *,
    successful: bool,
) -> tuple[list[dict], bool]:
    """Select at most one outcome of each type from every repeated task.

    The full sibling outcome summary remains attached to the representative,
    so the analyst sees the within-task contrast without counting four samples
    of one task as four independent sources of support.
    """
    grouped_mode = any(
        int(row.get("rollout_count", 1) or 1) > 1
        and row.get("rollout_group_id")
        for row in results
    )
    if not grouped_mode:
        selected = [row for row in results if bool(row.get("hard")) == successful]
        return selected, False

    groups: dict[str, list[dict]] = {}
    for row in results:
        group_id = str(row.get("rollout_group_id") or row.get("id", ""))
        groups.setdefault(group_id, []).append(row)

    representatives: list[dict] = []
    for members in groups.values():
        candidates = [
            row for row in members
            if bool(row.get("hard")) == successful
        ]
        if not candidates:
            continue
        if is_group_relative_edit_credit_enabled():
            if successful:
                candidates.sort(
                    key=lambda row: (
                        -float(row.get("group_relative_hard", 0.0) or 0.0),
                        -float(row.get("soft", 0.0) or 0.0),
                        int(row.get("n_turns", 0) or 0),
                        int(row.get("rollout_index", 0) or 0),
                    )
                )
            else:
                candidates.sort(
                    key=lambda row: (
                        float(row.get("group_relative_hard", 0.0) or 0.0),
                        float(row.get("soft", 0.0) or 0.0),
                        int(row.get("rollout_index", 0) or 0),
                    )
                )
        else:
            candidates.sort(
                key=lambda row: (
                    int(row.get("rollout_index", 0) or 0),
                    str(row.get("id", "")),
                )
            )
        representatives.append(candidates[0])
    return representatives, True


_VOLATILE_TRAJECTORY_KEYS = {
    "created_at",
    "duration",
    "elapsed",
    "latency",
    "request_id",
    "timestamp",
    "timing",
    "usage",
}


def _canonicalize_trajectory_value(value):
    """Normalize a saved conversation for behavior-level exact deduplication."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_trajectory_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _VOLATILE_TRAJECTORY_KEYS
        }
    if isinstance(value, list):
        return [_canonicalize_trajectory_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _trajectory_signature(item: dict, prediction_dir: str) -> str:
    """Return a deterministic signature without treating timing as behavior."""
    task_id = str(item.get("id") or "")
    conversation_path = os.path.join(
        prediction_dir,
        task_id,
        "conversation.json",
    )
    conversation = None
    if task_id and os.path.exists(conversation_path):
        try:
            with open(conversation_path) as f:
                conversation = json.load(f)
        except (OSError, ValueError):
            conversation = None
    if conversation is None:
        conversation = {
            "predicted_answer": item.get("predicted_answer"),
            "response": item.get("response") or item.get("response_excerpt"),
            "fail_reason": item.get("fail_reason"),
            "n_turns": item.get("n_turns"),
        }
    payload = {
        "hard": int(bool(item.get("hard"))),
        "conversation": _canonicalize_trajectory_value(conversation),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deduplicate_task_group(
    members: list[dict],
    prediction_dir: str,
    *,
    route: str,
) -> list[dict]:
    """Keep each unique sibling behavior once while preserving multiplicity."""
    ordered = sorted(
        members,
        key=lambda row: (
            int(row.get("rollout_index", 0) or 0),
            str(row.get("id") or ""),
        ),
    )
    unique: list[dict] = []
    signature_to_index: dict[str, int] = {}
    source_ids: list[list[str]] = []
    for member in ordered:
        signature = _trajectory_signature(member, prediction_dir)
        source_id = str(member.get("id") or "")
        if signature in signature_to_index:
            index = signature_to_index[signature]
            source_ids[index].append(source_id)
            continue
        signature_to_index[signature] = len(unique)
        unique.append(dict(member))
        source_ids.append([source_id])

    unique_count = len(unique)
    for index, member in enumerate(unique, 1):
        ids = [source_id for source_id in source_ids[index - 1] if source_id]
        member["_outcome_stratified_route"] = route
        member["_outcome_stratified_task_primary"] = index == 1
        member["_outcome_stratified_unique_index"] = index
        member["_outcome_stratified_unique_count"] = unique_count
        member["_outcome_stratified_multiplicity"] = max(1, len(ids))
        member["_outcome_stratified_source_rollout_ids"] = ids
    return unique


def build_outcome_stratified_task_blocks(
    results: list[dict],
    prediction_dir: str,
) -> dict[str, list[list[dict]]]:
    """Route each K-rollout task to mixed, all-success, or all-failure blocks."""
    groups: dict[str, list[dict]] = {}
    for row in results:
        group_id = str(row.get("rollout_group_id") or row.get("id") or "")
        if group_id:
            groups.setdefault(group_id, []).append(row)

    routed: dict[str, list[list[dict]]] = {
        "mixed": [],
        "stable_success": [],
        "stable_failure": [],
    }
    for group_id in sorted(groups):
        members = groups[group_id]
        success_count = sum(bool(row.get("hard")) for row in members)
        if success_count == 0:
            route = "stable_failure"
        elif success_count == len(members):
            route = "stable_success"
        else:
            route = "mixed"
        block = _deduplicate_task_group(
            members,
            prediction_dir,
            route=route,
        )
        if block:
            routed[route].append(block)
    return routed


def _shuffle_task_blocks(
    blocks: list[list[dict]],
    seed: int | None,
) -> list[list[dict]]:
    ordered = list(blocks)
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    return ordered


def _split_task_block_minibatches(
    blocks: list[list[dict]],
    batch_size: int,
) -> tuple[list[list[list[dict]]], list[list[dict]]]:
    """Batch by independent tasks and avoid one-task homogeneous batches."""
    size = max(2, int(batch_size))
    if len(blocks) < 2:
        return [], list(blocks)
    batches = [blocks[index : index + size] for index in range(0, len(blocks), size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches, []


def _flatten_task_blocks(blocks: list[list[dict]]) -> list[dict]:
    return [row for block in blocks for row in block]


def _outcome_stratified_routing_audit(
    routed: dict[str, list[list[dict]]],
    skipped: dict[str, list[list[dict]]],
) -> dict:
    def summarize(block: list[dict]) -> dict:
        first = block[0]
        source_ids = sorted(
            {
                source_id
                for row in block
                for source_id in row.get(
                    "_outcome_stratified_source_rollout_ids", []
                )
                if source_id
            }
        )
        return {
            "task_id": str(first.get("rollout_group_id") or first.get("id") or ""),
            "route": str(first.get("_outcome_stratified_route") or ""),
            "original_rollout_count": len(source_ids),
            "unique_trajectory_count": len(block),
            "multiplicities": [
                int(row.get("_outcome_stratified_multiplicity", 1) or 1)
                for row in block
            ],
            "source_rollout_ids": source_ids,
        }

    routes = {
        route: [summarize(block) for block in blocks]
        for route, blocks in routed.items()
    }
    skipped_task_ids = {
        route: [
            str(block[0].get("rollout_group_id") or block[0].get("id") or "")
            for block in blocks
        ]
        for route, blocks in skipped.items()
        if blocks
    }
    return {
        "mode": "outcome_stratified_reflection",
        "route_task_counts": {
            route: len(blocks) for route, blocks in routed.items()
        },
        "route_unique_trajectory_counts": {
            route: sum(len(block) for block in blocks)
            for route, blocks in routed.items()
        },
        "routes": routes,
        "skipped_singleton_homogeneous_task_ids": skipped_task_ids,
        "support_unit": "distinct_task_id",
    }


def _select_mixed_task_contrastive_representatives(
    results: list[dict],
    *,
    limit: int | None = None,
) -> list[dict]:
    """Return one strongest negative representative for each mixed task.

    Each selected row still carries the complete sibling outcome summary, so a
    dedicated analyst call can compare positive and negative trajectories while
    assigning provenance to exactly one independent source task.
    """
    groups: dict[str, list[dict]] = {}
    for row in results:
        group_id = str(row.get("rollout_group_id") or row.get("id", ""))
        summary = row.get("same_task_group")
        if not group_id or not isinstance(summary, dict):
            continue
        if str(summary.get("status") or "") != "mixed":
            continue
        groups.setdefault(group_id, []).append(row)

    representatives: list[dict] = []
    for group_id in sorted(groups):
        failures = [row for row in groups[group_id] if not bool(row.get("hard"))]
        if not failures:
            continue
        failures.sort(
            key=lambda row: (
                float(row.get("group_relative_hard", 0.0) or 0.0),
                float(row.get("soft", 0.0) or 0.0),
                int(row.get("rollout_index", 0) or 0),
            )
        )
        representatives.append(failures[0])
    representatives.sort(
        key=lambda row: (
            -4.0
            * float(row.get("same_task_group", {}).get("hard_mean", 0.0) or 0.0)
            * (
                1.0
                - float(
                    row.get("same_task_group", {}).get("hard_mean", 0.0)
                    or 0.0
                )
            ),
            str(row.get("rollout_group_id") or row.get("id") or ""),
        )
    )
    if limit is not None:
        return representatives[: max(0, int(limit))]
    return representatives


_CONTRASTIVE_ANALYST_SUFFIX = """

## Dedicated Same-Task Counterfactual Analysis

This call contains exactly one source task with both successful and failed
sibling rollouts. Locate their first causal behavioral divergence. Propose at
most one atomic, inference-time executable rule that would preserve the good
behavior or repair the bad behavior. If no causal and generalizable difference
is visible, return an empty patch. Do not summarize the task and do not combine
multiple behaviors. Set `evidence_kind` to `contrastive` and
`generality_scope` to `single_task`.

The edit content must contain a complete trigger, action, and observable check;
never emit a heading or topic label without its rule body. Add this object to
the edit:
`causal_trace`: {
  `observable_trigger`, `first_divergent_decision`, `successful_behavior`,
  `failed_behavior`, `verification_signal`
}.
Use only information visible in the two sibling trajectories. If any field
cannot be grounded, return an empty patch instead of guessing.
"""


def run_contrastive_analyst_group(
    skill_content: str,
    item: dict,
    prediction_dir: str,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    trajectory_memory_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool = False,
) -> dict | None:
    """Extract one atomically-provenanced edit from one mixed rollout group."""
    mode = normalize_update_mode(update_mode)
    if is_full_rewrite_minibatch_mode(mode):
        return None
    actual_system = _resolve_prompt(system_prompt, "analyst_error", mode)
    actual_system = actual_system.rstrip() + _CONTRASTIVE_ANALYST_SUFFIX
    result = run_error_analyst_minibatch(
        skill_content,
        [item],
        prediction_dir,
        edit_budget=1,
        system_prompt=actual_system,
        step_buffer_context=step_buffer_context,
        trajectory_memory_context=trajectory_memory_context,
        meta_skill_context=meta_skill_context,
        update_mode=mode,
        skill_aware_reflection=skill_aware_reflection,
    )
    if not result:
        return None
    task_id = str(item.get("rollout_group_id") or item.get("id") or "")
    result["source_type"] = "contrastive"
    for edit in get_payload_items(result.get("patch", {}), mode):
        edit["evidence_task_ids"] = [task_id] if task_id else []
        edit["evidence_kind"] = "contrastive"
        edit["generality_scope"] = "single_task"
    annotate_analyst_patch(result, [item], update_mode=mode)
    return result


_STRATIFIED_MIXED_SUFFIX = """

## Outcome-Stratified Route A: Mixed Sibling Outcomes

This call contains all unique trajectories from exactly one task whose K
sibling rollouts include both successes and failures. Compare complete
trajectories, locate the first observable behavioral divergence, and emit at
most one atomic repair or preservation rule. The rule must be supported by the
within-task contrast rather than by task literals or final evaluator labels.
If no plausible causal divergence is visible, return an empty patch.
"""


_STRATIFIED_SUCCESS_SUFFIX = """

## Outcome-Stratified Route B: Cross-Task Stable Success

Each Task Block is one independent task whose sibling rollouts all succeeded.
Multiple unique trajectories inside one Task Block are correlated alternatives,
not independent votes. First compare Task Blocks semantically. Emit a rule only
for an inference-time behavior shared by at least two distinct task_ids. Treat
the result as preservation evidence, not proof that the behavior caused
success. If no cross-task invariant is visible, return an empty patch. Set
`evidence_kind` to `consensus` and `generality_scope` to `cross_task`.
"""


_STRATIFIED_FAILURE_SUFFIX = """

## Outcome-Stratified Route C: Cross-Task Stable Failure

Each Task Block is one independent task whose sibling rollouts all failed.
Multiple unique trajectories inside one Task Block are correlated attempts,
not independent votes. First cluster Task Blocks by a shared, observable
failure mechanism. Emit a cautious repair hypothesis only when at least two
distinct task_ids expose the same mechanism. Do not infer a correct answer or
task-specific procedure from failure alone. If no cross-task mechanism is
visible, return an empty patch. Set `evidence_kind` to `consensus` and
`generality_scope` to `cross_task`.
"""


def run_outcome_stratified_analyst_group(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    *,
    route: str,
    edit_budget: int,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    trajectory_memory_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool = False,
    emit_appendix_notes: bool = True,
) -> dict | None:
    """Run one route-specific analyst call over task-blocked evidence."""
    mode = normalize_update_mode(update_mode)
    if is_full_rewrite_minibatch_mode(mode):
        return None
    cross_group = cross_group_evidence_enabled()
    if cross_group:
        default_prompt = "analyst_success" if route == "stable_success" else "analyst_error"
        system_prompt = (
            _resolve_prompt(system_prompt, default_prompt, mode).rstrip()
            + CROSS_GROUP_ANALYST_SUFFIX
        )
    if route == "mixed":
        resolved = _resolve_prompt(system_prompt, "analyst_error", mode)
        result = run_error_analyst_minibatch(
            skill_content,
            items,
            prediction_dir,
            edit_budget=1,
            system_prompt=resolved.rstrip() + _STRATIFIED_MIXED_SUFFIX,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=mode,
            skill_aware_reflection=skill_aware_reflection,
            trajectory_section_label="Mixed Sibling Trajectories",
            raise_errors=True,
        )
        source_type = "contrastive"
    elif route == "stable_success":
        resolved = _resolve_prompt(system_prompt, "analyst_success", mode)
        result = run_success_analyst_minibatch(
            skill_content,
            items,
            prediction_dir,
            edit_budget=max(1, int(edit_budget)),
            system_prompt=resolved.rstrip() + _STRATIFIED_SUCCESS_SUFFIX,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=mode,
            skill_aware_reflection=skill_aware_reflection,
            emit_appendix_notes=emit_appendix_notes,
            raise_errors=True,
        )
        source_type = "success"
    elif route == "stable_failure":
        resolved = _resolve_prompt(system_prompt, "analyst_error", mode)
        result = run_error_analyst_minibatch(
            skill_content,
            items,
            prediction_dir,
            edit_budget=max(1, int(edit_budget)),
            system_prompt=resolved.rstrip() + _STRATIFIED_FAILURE_SUFFIX,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=mode,
            skill_aware_reflection=skill_aware_reflection,
            raise_errors=True,
        )
        source_type = "failure"
    else:
        raise ValueError(f"unknown outcome-stratified route: {route!r}")

    if not result:
        return None
    if cross_group:
        result = attach_cross_group_cards(result, items, prediction_dir, route, fmt_trajectory)
    result["source_type"] = source_type
    result["reflection_route"] = route
    payload_items = get_payload_items(result.get("patch", {}), mode)
    if route == "mixed":
        task_ids = sorted(
            {
                str(item.get("rollout_group_id") or item.get("id") or "")
                for item in items
                if str(item.get("rollout_group_id") or item.get("id") or "")
            }
        )
        for edit in payload_items:
            edit["evidence_task_ids"] = task_ids[:1]
            edit["evidence_kind"] = "contrastive"
            edit["generality_scope"] = "single_task"
            edit["reflection_route"] = route
    else:
        for edit in payload_items:
            edit["reflection_route"] = route
    if is_group_relative_edit_credit_enabled():
        annotate_analyst_patch(result, items, update_mode=mode)
    else:
        result["outcome_stratified_source_task_ids"] = sorted(
            {
                str(item.get("rollout_group_id") or item.get("id") or "")
                for item in items
                if str(item.get("rollout_group_id") or item.get("id") or "")
            }
        )
        result["outcome_stratified_source_rollout_ids"] = sorted(
            {
                str(source_id)
                for item in items
                for source_id in item.get(
                    "_outcome_stratified_source_rollout_ids",
                    [item.get("id")],
                )
                if str(source_id or "")
            }
        )
    return result


# ── Prompt resolution ───────────────────────────────────────────────────────


def _resolve_prompt(custom: str | None, default_name: str, update_mode: str = "patch") -> str:
    """Return *custom* if provided (non-None), otherwise load from file."""
    if custom is not None:
        return custom
    mode = normalize_update_mode(update_mode)
    actual_name = default_name
    if is_full_rewrite_minibatch_mode(mode):
        full_name = f"{default_name}_full_rewrite"
        try:
            return load_prompt(full_name)
        except FileNotFoundError:
            actual_name = default_name
    elif mode == "rewrite_from_suggestions":
        rewrite_name = f"{default_name}_rewrite"
        try:
            return load_prompt(rewrite_name)
        except FileNotFoundError:
            actual_name = default_name
    return load_prompt(actual_name)


# ── Minibatch analysts ──────────────────────────────────────────────────────


def run_error_analyst_minibatch(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool = False,
    trajectory_section_label: str = "Failed Trajectories",
    raise_errors: bool = False,
) -> dict | None:
    """Analyze a minibatch of failed trajectories in one optimizer call.

    Parameters
    ----------
    skill_content : str
        Current skill document text.
    items : list[dict]
        Rollout result dicts (all should have ``hard=0``).
    prediction_dir : str
        Path to ``predictions/`` directory.
    edit_budget : int
        Maximum number of edits (L).
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.
    rejection_context : str
        *Deprecated* — use ``step_buffer_context``.
    trajectory_memory_context : str
        *Deprecated* — use ``step_buffer_context``.
    step_buffer_context : str
        Unified summary of previous steps (failure patterns + rejected edits).

    Returns
    -------
    dict | None
        Patch dict with ``source_type="failure"``, or ``None`` on error.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = _resolve_prompt(system_prompt, "analyst_error", mode)
    # Skill-aware reflection: augment the resolved prompt at runtime so both
    # env-specific and generic analyst prompts get the defect/lapse instruction.
    # When the toggle is off this is a no-op (prompt byte-identical to baseline).
    if skill_aware_reflection and not is_full_rewrite_minibatch_mode(mode):
        actual_system = augment_error_prompt(actual_system)
    group_relative_enabled = is_group_relative_edit_credit_enabled()
    if group_relative_enabled and not is_full_rewrite_minibatch_mode(mode):
        actual_system = augment_group_relative_analyst_prompt(actual_system)

    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
    )
    if is_full_rewrite_minibatch_mode(mode):
        user += (
            f"## Update Format\n"
            f"Produce one complete replacement skill candidate for this minibatch. "
            f"Do not output edits, patches, or revise suggestions.\n\n"
        )
    else:
        user += (
            f"## {payload_label(mode, title=True)} Budget\n"
            f"Produce at most L={edit_budget} {payload_label(mode)}.\n\n"
        )
    # Unified step buffer context (preferred)
    ctx = step_buffer_context or rejection_context or ""
    if trajectory_memory_context:
        ctx = f"{ctx}\n{trajectory_memory_context}" if ctx else trajectory_memory_context
    if ctx.strip():
        user += f"## Previous Steps in This Epoch\n{ctx}\n\n"
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    if group_relative_enabled:
        group_context = format_group_relative_analyst_context(items)
        if group_context:
            user += group_context + "\n\n"
    section_label = str(trajectory_section_label or "Failed Trajectories")
    user += f"## {section_label} ({len(items)} total)\n{trajectories_text}"

    try:
        response, _ = chat_optimizer(
            system=actual_system, user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(mode) else 16384,
            retries=3,
            stage="analyst",
        )
        result = extract_json(response)
        if not result:
            return None
        notes = extract_appendix_notes(result) if skill_aware_reflection else []
        if "patch" in result:
            result["source_type"] = "failure"
            if not is_full_rewrite_minibatch_mode(mode):
                truncate_payload(result["patch"], edit_budget, mode)
            if group_relative_enabled:
                annotate_analyst_patch(result, items, update_mode=mode)
            if skill_aware_reflection:
                result["appendix_notes"] = notes
            return result
        # Skill-aware: a batch may legitimately yield ONLY execution-lapse notes
        # (no body edit). Return a no-op patch so the notes still reach the
        # trainer via all_raw_patches; empty edits are dropped from the body
        # pipeline by _normalise_patches, so body behavior is unchanged.
        if skill_aware_reflection and notes:
            return {
                "source_type": "failure",
                "patch": {"reasoning": "execution-lapse only", "edits": []},
                "appendix_notes": notes,
            }
    except Exception:  # noqa: BLE001
        if raise_errors:
            raise
        traceback.print_exc()
    return None


def run_success_analyst_minibatch(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool = False,
    emit_appendix_notes: bool = True,
    raise_errors: bool = False,
) -> dict | None:
    """Analyze a minibatch of successful trajectories in one optimizer call.

    Parameters
    ----------
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.
    trajectory_memory_context : str
        *Deprecated* — use ``step_buffer_context``.
    step_buffer_context : str
        Unified summary of previous steps (failure patterns + rejected edits).

    Returns
    -------
    dict | None
        Patch dict with ``source_type="success"``, or ``None`` on error.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = _resolve_prompt(system_prompt, "analyst_success", mode)
    # Only augment + parse appendix notes on the success side when allowed.
    # failure_only mode (paper-faithful S_app) suppresses success-side notes.
    sa_emit = skill_aware_reflection and emit_appendix_notes
    if sa_emit and not is_full_rewrite_minibatch_mode(mode):
        actual_system = augment_success_prompt(actual_system)
    group_relative_enabled = is_group_relative_edit_credit_enabled()
    if group_relative_enabled and not is_full_rewrite_minibatch_mode(mode):
        actual_system = augment_group_relative_analyst_prompt(actual_system)

    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
    )
    if is_full_rewrite_minibatch_mode(mode):
        user += (
            f"## Update Format\n"
            f"Produce one complete replacement skill candidate for this minibatch. "
            f"Do not output edits, patches, or revise suggestions.\n\n"
        )
    else:
        user += (
            f"## {payload_label(mode, title=True)} Budget\n"
            f"Produce at most L={edit_budget} {payload_label(mode)}.\n\n"
        )
    ctx = step_buffer_context or trajectory_memory_context or ""
    if ctx.strip():
        user += f"## Previous Steps in This Epoch\n{ctx}\n\n"
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    if group_relative_enabled:
        group_context = format_group_relative_analyst_context(items)
        if group_context:
            user += group_context + "\n\n"
    user += f"## Successful Trajectories ({len(items)} total)\n{trajectories_text}"

    try:
        response, _ = chat_optimizer(
            system=actual_system, user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(mode) else 16384,
            retries=3,
            stage="analyst",
        )
        result = extract_json(response)
        if result and "patch" in result:
            result["source_type"] = "success"
            if not is_full_rewrite_minibatch_mode(mode):
                truncate_payload(result["patch"], edit_budget, mode)
            if group_relative_enabled:
                annotate_analyst_patch(result, items, update_mode=mode)
            if sa_emit:
                result["appendix_notes"] = extract_appendix_notes(result)
            return result
    except Exception:  # noqa: BLE001
        if raise_errors:
            raise
        traceback.print_exc()
    return None


# ── Minibatch reflect dispatcher ────────────────────────────────────────────


def _split_minibatches(items: list, batch_size: int) -> list[list]:
    """Split items into minibatches of at most *batch_size*."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _shuffle_for_minibatch(items: list, seed: int | None) -> list:
    """Return items in minibatch order.

    Uses a deterministic shuffle when a seed is provided so resume runs keep
    the same minibatch composition. Falls back to input order when no seed is
    available.
    """
    ordered = list(items)
    if seed is None:
        return ordered
    random.Random(seed).shuffle(ordered)
    return ordered


def _run_outcome_stratified_reflect(
    results: list[dict],
    skill_content: str,
    prediction_dir: str,
    patches_dir: str,
    workers: int,
    failure_only: bool,
    minibatch_size: int,
    edit_budget: int,
    random_seed: int | None,
    *,
    error_system: str | None,
    success_system: str | None,
    rejection_context: str,
    trajectory_memory_context: str,
    step_buffer_context: str,
    meta_skill_context: str,
    update_mode: str,
    skill_aware_reflection: bool,
    skill_aware_appendix_source: str,
    taskwise_edit_budget: int,
) -> list[dict | None]:
    """Route mixed tasks locally and homogeneous tasks across task blocks."""
    if cross_group_evidence_enabled():
        patches_dir = os.path.join(patches_dir, "cross_group_evidence")
        os.makedirs(patches_dir, exist_ok=True)
    routed = build_outcome_stratified_task_blocks(results, prediction_dir)
    mixed_blocks = _shuffle_task_blocks(routed["mixed"], random_seed)
    failure_blocks = _shuffle_task_blocks(
        routed["stable_failure"],
        None if random_seed is None else random_seed + 1,
    )
    success_blocks = _shuffle_task_blocks(
        routed["stable_success"],
        None if random_seed is None else random_seed + 2,
    )

    failure_batches, skipped_failures = _split_task_block_minibatches(
        failure_blocks,
        minibatch_size,
    )
    if failure_only:
        success_batches: list[list[list[dict]]] = []
        skipped_successes: list[list[dict]] = []
    else:
        success_batches, skipped_successes = _split_task_block_minibatches(
            success_blocks,
            minibatch_size,
        )

    routing_audit = _outcome_stratified_routing_audit(
        routed,
        {
            "stable_failure": skipped_failures,
            "stable_success": skipped_successes,
        },
    )
    routing_audit["minibatches"] = {
        "mixed": [
            [str(block[0].get("rollout_group_id") or block[0].get("id") or "")]
            for block in mixed_blocks
        ],
        "stable_failure": [
            [
                str(block[0].get("rollout_group_id") or block[0].get("id") or "")
                for block in batch
            ]
            for batch in failure_batches
        ],
        "stable_success": [
            [
                str(block[0].get("rollout_group_id") or block[0].get("id") or "")
                for block in batch
            ]
            for batch in success_batches
        ],
    }
    with open(
        os.path.join(patches_dir, "outcome_stratified_routing.json"),
        "w",
    ) as f:
        json.dump(routing_audit, f, ensure_ascii=False, indent=2)

    print(
        "    [2/6 REFLECT outcome-stratified] "
        f"mixed={len(mixed_blocks)} tasks/"
        f"{sum(len(block) for block in mixed_blocks)} unique trajectories; "
        f"all-failure={len(failure_blocks)} tasks/"
        f"{sum(len(block) for block in failure_blocks)} unique trajectories"
        f"→{len(failure_batches)} cross-task batches; "
        f"all-success={len(success_blocks)} tasks/"
        f"{sum(len(block) for block in success_blocks)} unique trajectories"
        f"→{len(success_batches)} cross-task batches "
        f"(task_M={max(2, int(minibatch_size))}, K evidence deduplicated, "
        f"workers={workers})"
    )
    if skipped_failures or skipped_successes:
        print(
            "      [outcome-stratified] skipped singleton homogeneous "
            f"tasks: failure={len(skipped_failures)} "
            f"success={len(skipped_successes)}"
        )

    jobs: list[tuple[str, int, str, list[dict], int]] = []
    for index, block in enumerate(mixed_blocks):
        jobs.append(
            (
                "route_a_mixed",
                index,
                "mixed",
                _flatten_task_blocks([block]),
                min(max(1, int(taskwise_edit_budget)), max(1, int(edit_budget))),
            )
        )
    for index, batch in enumerate(failure_batches):
        jobs.append(
            (
                "route_c_failure",
                index,
                "stable_failure",
                _flatten_task_blocks(batch),
                max(1, int(edit_budget)),
            )
        )
    for index, batch in enumerate(success_batches):
        jobs.append(
            (
                "route_b_success",
                index,
                "stable_success",
                _flatten_task_blocks(batch),
                max(1, int(edit_budget)),
            )
        )

    raw_patches: list[dict | None] = []
    pending: list[tuple[str, int, str, list[dict], int]] = []
    for tag, index, route, items, budget in jobs:
        path = os.path.join(patches_dir, f"{tag}_{index:03d}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw_patches.append(json.load(f))
        else:
            pending.append((tag, index, route, items, budget))

    def run_job(
        tag: str,
        index: int,
        route: str,
        items: list[dict],
        budget: int,
    ) -> list[tuple[str, dict | None, int, int]]:
        def analyze(child_items, child_budget):
            return run_outcome_stratified_analyst_group(
                skill_content,
                child_items,
                prediction_dir,
                route=route,
                edit_budget=child_budget,
                system_prompt=(
                    success_system if route == "stable_success" else error_system
                ),
                step_buffer_context=step_buffer_context or rejection_context,
                trajectory_memory_context=trajectory_memory_context,
                meta_skill_context=meta_skill_context,
                update_mode=update_mode,
                skill_aware_reflection=skill_aware_reflection,
                emit_appendix_notes=(
                    skill_aware_appendix_source != "failure_only"
                ),
            )

        return run_context_bounded_analyst(
            items, budget, route, analyze, patches_dir, f"{tag}_{index:03d}",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_job, *job): job
            for job in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            for tag, patch, task_count, trajectory_count in future.result():
                if patch:
                    raw_patches.append(patch)
                item_count = len(
                    get_payload_items(
                        patch.get("patch", {}) if patch else {},
                        update_mode,
                    )
                )
                print(
                    f"      [analyst] {completed}/{len(pending)} {tag} "
                    f"({task_count} tasks, {trajectory_count} unique trajectories) "
                    f"→ {item_count} {payload_label(update_mode)}"
                )
    return raw_patches


def run_minibatch_reflect(
    results: list[dict],
    skill_content: str,
    prediction_dir: str,
    patches_dir: str,
    workers: int,
    failure_only: bool,
    minibatch_size: int = 8,
    edit_budget: int = 4,
    random_seed: int | None = None,
    *,
    error_system: str | None = None,
    success_system: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool | None = None,
    skill_aware_appendix_source: str | None = None,
) -> list[dict | None]:
    """Full minibatch reflect stage: group → parallel optimizer calls → patches.

    Separates failure and success trajectories, splits each into minibatches
    of size M, runs all minibatches in parallel, and saves patch files.

    Parameters
    ----------
    results : list[dict]
        Rollout result dicts; see :class:`~skillopt.types.RolloutResult`.
    skill_content : str
        Current skill document.
    prediction_dir : str
        Path to ``predictions/`` with ``conversation.json`` files.
    patches_dir : str
        Path to save per-minibatch patch JSON files.
    workers : int
        Max parallel optimizer calls.
    failure_only : bool
        If True, skip success trajectories.
    minibatch_size : int
        Trajectories per group (M).
    edit_budget : int
        Max edits per minibatch (L).
    random_seed : int | None
        Optional seed used to shuffle trajectories before minibatch splitting.
    error_system, success_system : str | None
        Optional custom prompts. ``None`` = use generic defaults.

    Returns
    -------
    list[dict | None]
        Patch dicts (with ``source_type`` "failure" or "success").
    """
    # Resolve the skill-aware toggle: explicit kwargs win; otherwise fall back
    # to the process-wide config switch set by the trainer, so the feature is
    # env-independent and adapters need no per-benchmark wiring.
    if skill_aware_reflection is None:
        skill_aware_reflection = is_skill_aware_enabled()
    if skill_aware_appendix_source is None:
        skill_aware_appendix_source = get_skill_aware_appendix_source()

    os.makedirs(patches_dir, exist_ok=True)

    # Separate failure / success. With same-task K>1, each analyst receives at
    # most one representative per task group and reads all sibling outcomes
    # from the attached group summary.
    failures, grouped_mode = _select_same_task_group_representatives(
        results,
        successful=False,
    )
    if failure_only:
        successes = []
    else:
        successes, _ = _select_same_task_group_representatives(
            results,
            successful=True,
        )

    group_runtime_cfg = get_group_relative_edit_credit_config()
    group_relative_enabled = (
        is_group_relative_edit_credit_enabled()
        and not is_full_rewrite_minibatch_mode(update_mode)
    )
    group_cfg = (
        group_runtime_cfg
        if group_relative_enabled
        else {}
    )
    taskwise_reflection = bool(
        group_cfg.get("taskwise_reflection", False)
    ) and grouped_mode
    outcome_stratified_reflection = bool(
        group_runtime_cfg.get("outcome_stratified_reflection", False)
    ) and grouped_mode and not is_full_rewrite_minibatch_mode(update_mode)

    if outcome_stratified_reflection:
        return _run_outcome_stratified_reflect(
            results,
            skill_content,
            prediction_dir,
            patches_dir,
            workers,
            failure_only,
            minibatch_size,
            edit_budget,
            random_seed,
            error_system=error_system,
            success_system=success_system,
            rejection_context=rejection_context,
            trajectory_memory_context=trajectory_memory_context,
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            skill_aware_reflection=skill_aware_reflection,
            skill_aware_appendix_source=skill_aware_appendix_source,
            taskwise_edit_budget=int(
                group_runtime_cfg.get("taskwise_edit_budget", 1) or 1
            ),
        )

    contrastive_groups: list[dict] = []
    if group_relative_enabled:
        contrastive_groups = _select_mixed_task_contrastive_representatives(
            results,
            # Taskwise mode needs one exact-provenance update for every mixed
            # task. The legacy cap remains available for batched reflection.
            limit=(
                None
                if taskwise_reflection
                else int(group_cfg.get("contrastive_max_groups", 8) or 0)
            ),
        )

    if taskwise_reflection:
        # A mixed task is handled exactly once by the dedicated counterfactual
        # analyst. Stable groups are reflected independently, which makes the
        # existing single-record evidence fallback exact rather than heuristic.
        mixed_ids = {
            str(row.get("rollout_group_id") or row.get("id") or "")
            for row in contrastive_groups
        }
        failures = [
            row
            for row in failures
            if str(row.get("rollout_group_id") or row.get("id") or "")
            not in mixed_ids
        ]
        successes = [
            row
            for row in successes
            if str(row.get("rollout_group_id") or row.get("id") or "")
            not in mixed_ids
        ]

    failures = _shuffle_for_minibatch(failures, random_seed)
    successes = _shuffle_for_minibatch(
        successes,
        None if random_seed is None else random_seed + 1,
    )

    effective_minibatch_size = 1 if taskwise_reflection else minibatch_size
    analyst_edit_budget = (
        min(
            edit_budget,
            int(group_cfg.get("taskwise_edit_budget", 1) or 1),
        )
        if taskwise_reflection
        else edit_budget
    )

    # Split into minibatches. In taskwise mode, each patch has one and only
    # one source task, so K sibling rollouts can never become K support votes.
    fail_batches = _split_minibatches(failures, effective_minibatch_size)
    succ_batches = _split_minibatches(successes, effective_minibatch_size)

    n_fail_batches = len(fail_batches)
    n_succ_batches = len(succ_batches)
    print(
        f"    [2/6 REFLECT minibatch] "
        f"failure={len(failures)}→{n_fail_batches} groups  "
        f"success={len(successes)}→{n_succ_batches} groups  "
        f"contrastive={len(contrastive_groups)} groups  "
        f"(M={effective_minibatch_size}, L={analyst_edit_budget}, "
        f"workers={workers}, same_task_grouped={grouped_mode}, "
        f"taskwise={taskwise_reflection})"
    )

    raw_patches: list[dict | None] = []

    # Resume support: check for already-done minibatch patches
    pending_fail: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(fail_batches):
        path = os.path.join(patches_dir, f"minibatch_fail_{idx:03d}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw_patches.append(json.load(f))
        else:
            pending_fail.append((idx, batch))

    pending_succ: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(succ_batches):
        path = os.path.join(patches_dir, f"minibatch_succ_{idx:03d}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw_patches.append(json.load(f))
        else:
            pending_succ.append((idx, batch))

    pending_contrastive: list[tuple[int, dict]] = []
    for idx, item in enumerate(contrastive_groups):
        path = os.path.join(patches_dir, f"contrastive_{idx:03d}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw_patches.append(json.load(f))
        else:
            pending_contrastive.append((idx, item))

    # ── Worker functions ──────────────────────────────────────────────────
    def _do_fail(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_error_analyst_minibatch(
            skill_content, batch, prediction_dir,
            edit_budget=analyst_edit_budget,
            system_prompt=error_system,
            step_buffer_context=step_buffer_context,
            # backward compat fallback
            rejection_context=rejection_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            skill_aware_reflection=skill_aware_reflection,
        )
        return f"minibatch_fail_{idx:03d}", patch

    def _do_succ(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_success_analyst_minibatch(
            skill_content, batch, prediction_dir,
            edit_budget=analyst_edit_budget,
            system_prompt=success_system,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            skill_aware_reflection=skill_aware_reflection,
            emit_appendix_notes=(skill_aware_appendix_source != "failure_only"),
        )
        return f"minibatch_succ_{idx:03d}", patch

    def _do_contrastive(idx: int, item: dict) -> tuple[str, dict | None]:
        patch = run_contrastive_analyst_group(
            skill_content,
            item,
            prediction_dir,
            system_prompt=error_system,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            skill_aware_reflection=skill_aware_reflection,
        )
        return f"contrastive_{idx:03d}", patch

    # Run all pending minibatches in parallel
    all_pending = (
        [("fail", idx, batch) for idx, batch in pending_fail]
        + [("succ", idx, batch) for idx, batch in pending_succ]
        + [("contrastive", idx, [item]) for idx, item in pending_contrastive]
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for kind, idx, batch in all_pending:
            if kind == "fail":
                futs[ex.submit(_do_fail, idx, batch)] = (kind, idx, len(batch))
            elif kind == "contrastive":
                futs[ex.submit(_do_contrastive, idx, batch[0])] = (
                    kind,
                    idx,
                    1,
                )
            else:
                futs[ex.submit(_do_succ, idx, batch)] = (kind, idx, len(batch))

        for i, fut in enumerate(as_completed(futs), 1):
            kind, idx, batch_len = futs[fut]
            tag, patch = fut.result()
            if patch:
                path = os.path.join(patches_dir, f"{tag}.json")
                with open(path, "w") as f:
                    json.dump(patch, f, ensure_ascii=False, indent=2)
                raw_patches.append(patch)
            n_edits = len(get_payload_items(patch.get("patch", {}) if patch else {}, update_mode))
            print(
                f"      [analyst] {i}/{len(all_pending)} {tag} "
                f"({batch_len} trajs) → {n_edits} {payload_label(update_mode)}"
            )

    return raw_patches
