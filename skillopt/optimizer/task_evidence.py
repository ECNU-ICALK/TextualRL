"""Task-grounded support metadata for standalone outcome-stratified edits.

The optimizer judges behavioral relevance. Code checks citation provenance and
counts distinct tasks, not siblings. This is not a causal-effect estimator.
"""
from __future__ import annotations

from copy import deepcopy

from skillopt.optimizer.update_modes import get_payload_items


SUPPORT_UNIT = "cited_distinct_tasks"

CONDITIONAL_RULE_SUFFIX = """

## Conditional, Non-Conflicting Skill Rules

Write each edit as one coherent conditional behavior in its executable content:
when to apply it (an observable trigger), what action to take, and what visible
signal means the action is complete or further attempts should stop. Do not
turn successful one-off behavior into an unconditional rule. Do not repeatedly
verify already-established facts without a remaining uncertainty or new evidence.
Distinguish genuinely missing evidence from a recoverable search/interaction
failure; do not invent a substitute answer or assert unavailability prematurely.
Compare the proposed behavior against Current Skill. Prefer a targeted replace
of a conflicting/overlapping rule over appending a contradictory instruction.
Copy the exact target text for replace. Preserve supported exceptions and
unrelated rules; do not broaden a prohibition beyond the observed conditions.
Split unrelated behaviors with different conditions or supporting tasks into
separate edits. Do not bundle them to inflate support or evade the edit budget.
No additional fields are needed for these conditions: put them in content.
Existing protected sections and deployment-observability restrictions still apply.
"""

ANALYST_EVIDENCE_SUFFIX = """

## Per-Edit Behavioral Evidence

For EACH edit add supporting_evidence, a JSON list of objects with:
task_id, rollout_id, quote, rationale.
Use the Task Block ID and the displayed unique trajectory ID. quote is a short
verbatim span of the observed action/response/tool observation, not evaluator
labels, reference answers, or post-execution verification. rationale explains
why that behavior supports this particular whole conditional rule (for failures,
why it motivates the repair hypothesis). These are training-side audit metadata,
NEVER part of executable content. Cite only actual supporting tasks, not every
task in the input. Multiple siblings of one task still contribute ONE task.
For a mixed task cite both sides of the observed divergence. For stable success
or failure cite the shared mechanism across distinct tasks; success correlation
and failed attempts alone are not proof of a causal improvement.
Do not supply a numerical support estimate: support_count is computed from
valid citations. Missing citations receive zero support, never batch-size credit.
"""

MERGE_EVIDENCE_SUFFIX = """

## Task-Grounded Support (Overrides Source-Patch Count Heuristics)

support_count means distinct cited task IDs, NOT input trajectories, source
patches, or merge depth. Do not sum source support counts. For each output edit
return supporting_evidence with task_id, rollout_id, quote, rationale. Copy the
first three fields exactly from valid input supporting_evidence; reassess the
rationale for the OUTPUT behavior. Keep only citations actually supporting that
whole conditional behavior. Never transfer every citation in a patch to a new
clause. A materially new behavior with no supporting citation gets an empty list.
Merge equivalent conditional behaviors and deduplicate their evidence by task;
do not merge independent rules whose evidence/conditions differ. Evidence and
numeric support are metadata, not instructions to include in skill content.
"""

RANKING_EVIDENCE_SUFFIX = """

For task-grounded edits, support_count counts distinct tasks with traceable
behavioral citations, not batch size or sibling count. Use the cited rationale
to judge relevance and generality; this count is not a causal-effect estimate.
Prefer coherent, applicable rules and revisions resolving existing conflicts.
Do not add credit for input batch size or merge depth. Zero citation support
means unknown support, not demonstrated harm. Keep the existing edit budget.
"""


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _citation_key(row: dict) -> tuple[str, str, str]:
    return (_text(row.get("task_id")), _text(row.get("rollout_id")),
            _normalized(_text(row.get("quote"))))


def has_task_evidence(patches: list[dict], update_mode: str = "patch") -> bool:
    return any(
        e.get("support_unit") == SUPPORT_UNIT
        for p in patches for e in get_payload_items(p, update_mode)
        if isinstance(e, dict)
    )


def normalize_task_evidence(
    patch: dict,
    *,
    sources: dict[tuple[str, str], str] | None = None,
    parents: list[dict] | None = None,
    update_mode: str = "patch",
) -> dict:
    """Validate citations without filtering edits or claiming causal validity.

At extraction, sources contain only displayed runtime trajectory text. At merge,
the output can cite only already validated input citations. The optimizer must
still judge whether those citations support the revised behavior.
"""
    result = deepcopy(patch)
    allowed = set()
    if parents is not None:
        for parent in parents:
            for e in get_payload_items(parent, update_mode):
                if isinstance(e, dict) and e.get("support_unit") == SUPPORT_UNIT:
                    for row in e.get("supporting_evidence", []):
                        if isinstance(row, dict):
                            allowed.add(_citation_key(row))
    source_text = {key: _normalized(value) for key, value in (sources or {}).items()}
    for edit in get_payload_items(result, update_mode):
        if not isinstance(edit, dict):
            continue
        raw = edit.get("supporting_evidence", [])
        rejected = []
        kept = []
        seen = set()
        if not isinstance(raw, list):
            rejected.append({"reason": "malformed_evidence_list", "claim": raw})
            raw = []
        for row in raw:
            reason = ""
            if not isinstance(row, dict):
                rejected.append({"reason": "malformed_citation", "claim": row})
                continue
            task, rollout, quote = _citation_key(row)
            rationale = _text(row.get("rationale"))
            if not all((task, rollout, quote, rationale)):
                reason = "incomplete_citation"
            elif sources is not None:
                if (task, rollout) not in source_text:
                    reason = "unknown_task_or_rollout"
                elif quote not in source_text[task, rollout]:
                    reason = "quote_not_in_runtime_trajectory"
            elif (task, rollout, quote) not in allowed:
                reason = "citation_not_in_merge_inputs"
            if reason:
                rejected.append({"reason": reason, "claim": row})
                continue
            key = (task, rollout, quote)
            if key in seen:
                continue
            seen.add(key)
            kept.append({"task_id": task, "rollout_id": rollout,
                         "quote": _text(row["quote"]), "rationale": rationale})
        task_ids = sorted({row["task_id"] for row in kept})
        edit["task_evidence_audit"] = {
            "stage": "analyst" if sources is not None else "merge",
            "proposed_support_count": edit.get("support_count"),
            "proposed_evidence_task_ids": edit.get("evidence_task_ids"),
            "proposed_supporting_evidence": raw,
            "excluded_citations": rejected,
            "valid_citation_count": len(kept),
            "support_count": len(task_ids),
            "semantic_relevance": "optimizer_judgment_not_causal_verification",
        }
        edit["supporting_evidence"] = kept
        edit["evidence_task_ids"] = task_ids
        edit["support_count"] = len(task_ids)
        edit["support_unit"] = SUPPORT_UNIT
    return result


def merge_prompt_inputs(patches: list[dict], update_mode: str = "patch") -> list[dict]:
    """Keep invalid/raw claims on disk, but do not feed them back as evidence."""
    result = deepcopy(patches)
    for patch in result:
        for edit in get_payload_items(patch, update_mode):
            if isinstance(edit, dict):
                edit.pop("task_evidence_audit", None)
    return result
