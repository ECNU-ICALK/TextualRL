"""Group-relative credit assignment for same-task rollout skill edits.

The target policy may sample the same training task K times.  Those sibling
rollouts are correlated evidence, so they must not be counted as K independent
sources of support.  This module turns each same-task group into one unit of
evidence, records within-group outcome divergence, and ranks edits using only
training-side evidence.  Selection-set validation remains unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from typing import Any

from textualrl.optimizer.quarantine import (
    behavior_similarity,
)
from textualrl.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    set_payload_items,
)


_RUNTIME: dict[str, Any] = {
    "enabled": False,
    "semantic_similarity_threshold": 0.45,
    "direct_evidence_similarity_threshold": 0.65,
    "min_atomic_support_fraction": 1.0,
    "atomize_append_edits": False,
    "atomize_all_edit_clauses": False,
    "specificity_penalty_weight": 0.75,
    "max_specific_singletons": 1,
    "min_cross_task_support": 2,
    "max_exploratory": 1,
    "min_candidate_cross_task_edits": 0,
    "soft_cross_task_target": False,
    "max_contrastive_edits": -1,
    "max_singleton_edits": -1,
    "block_singleton_stable_hypotheses": False,
    "actionable_priority_bonus": 0.0,
    "causal_trace_priority_bonus": 0.0,
    "contrastive_max_groups": 8,
    "taskwise_reflection": False,
    "taskwise_edit_budget": 1,
    "outcome_stratified_reflection": False,
    "min_credit_score": 0.0,
    "require_actionable_units": True,
    "require_causal_trace": True,
    "output_contract_require_direct_causal_evidence": False,
    "credit_aggregation": "weakest",
    "model_semantic_matching": False,
}


def configure_group_relative_edit_credit(
    enabled: bool,
    *,
    semantic_similarity_threshold: float = 0.45,
    direct_evidence_similarity_threshold: float = 0.65,
    min_atomic_support_fraction: float = 1.0,
    atomize_append_edits: bool = False,
    atomize_all_edit_clauses: bool = False,
    specificity_penalty_weight: float = 0.75,
    max_specific_singletons: int = 1,
    min_cross_task_support: int = 2,
    max_exploratory: int = 1,
    min_candidate_cross_task_edits: int = 0,
    soft_cross_task_target: bool = False,
    max_contrastive_edits: int = -1,
    max_singleton_edits: int = -1,
    block_singleton_stable_hypotheses: bool = False,
    actionable_priority_bonus: float = 0.0,
    causal_trace_priority_bonus: float = 0.0,
    contrastive_max_groups: int = 8,
    taskwise_reflection: bool = False,
    taskwise_edit_budget: int = 1,
    outcome_stratified_reflection: bool = False,
    min_credit_score: float = 0.0,
    require_actionable_units: bool = True,
    require_causal_trace: bool = True,
    output_contract_require_direct_causal_evidence: bool = False,
    credit_aggregation: str = "weakest",
    model_semantic_matching: bool = False,
) -> None:
    """Configure the process-wide training-side credit policy."""
    normalized_credit_aggregation = str(credit_aggregation).strip().lower()
    if normalized_credit_aggregation not in {"weakest", "mean"}:
        raise ValueError(
            "credit_aggregation must be 'weakest' or 'mean', got "
            f"{credit_aggregation!r}"
        )
    _RUNTIME.update(
        {
            "enabled": bool(enabled),
            "semantic_similarity_threshold": min(
                1.0, max(0.0, float(semantic_similarity_threshold))
            ),
            "direct_evidence_similarity_threshold": min(
                1.0,
                max(0.0, float(direct_evidence_similarity_threshold)),
            ),
            "min_atomic_support_fraction": min(
                1.0, max(0.0, float(min_atomic_support_fraction))
            ),
            "atomize_append_edits": bool(atomize_append_edits),
            "atomize_all_edit_clauses": bool(atomize_all_edit_clauses),
            "specificity_penalty_weight": max(
                0.0, float(specificity_penalty_weight)
            ),
            "max_specific_singletons": max(
                0, int(max_specific_singletons)
            ),
            "min_cross_task_support": max(
                2, int(min_cross_task_support)
            ),
            "max_exploratory": max(0, int(max_exploratory)),
            "min_candidate_cross_task_edits": max(
                0, int(min_candidate_cross_task_edits)
            ),
            "soft_cross_task_target": bool(soft_cross_task_target),
            "max_contrastive_edits": max(-1, int(max_contrastive_edits)),
            "max_singleton_edits": max(-1, int(max_singleton_edits)),
            "block_singleton_stable_hypotheses": bool(
                block_singleton_stable_hypotheses
            ),
            "actionable_priority_bonus": max(
                0.0, float(actionable_priority_bonus)
            ),
            "causal_trace_priority_bonus": max(
                0.0, float(causal_trace_priority_bonus)
            ),
            "contrastive_max_groups": max(
                0, int(contrastive_max_groups)
            ),
            "taskwise_reflection": bool(taskwise_reflection),
            "taskwise_edit_budget": max(1, int(taskwise_edit_budget)),
            "outcome_stratified_reflection": bool(
                outcome_stratified_reflection
            ),
            "min_credit_score": float(min_credit_score),
            "require_actionable_units": bool(require_actionable_units),
            "require_causal_trace": bool(require_causal_trace),
            "output_contract_require_direct_causal_evidence": bool(
                output_contract_require_direct_causal_evidence
            ),
            "credit_aggregation": normalized_credit_aggregation,
            "model_semantic_matching": bool(model_semantic_matching),
        }
    )


def is_group_relative_edit_credit_enabled() -> bool:
    return bool(_RUNTIME["enabled"])


def get_group_relative_edit_credit_config() -> dict[str, Any]:
    return dict(_RUNTIME)


def _group_id(item: dict) -> str:
    return str(item.get("rollout_group_id") or item.get("id") or "")


def _task_description(item: dict) -> str:
    return str(
        item.get("task_description")
        or item.get("instruction")
        or item.get("question")
        or ""
    ).strip()


def _group_record(item: dict) -> dict[str, Any]:
    summary = item.get("same_task_group")
    if not isinstance(summary, dict):
        summary = {}
    status = str(summary.get("status") or "unknown")
    hard_mean = float(summary.get("hard_mean", item.get("hard", 0.0)) or 0.0)
    hard_mean = min(1.0, max(0.0, hard_mean))
    contrast_strength = 4.0 * hard_mean * (1.0 - hard_mean)
    confidence = {
        "mixed": 1.0,
        "stable_success": 0.35,
        "stable_failure": 0.25,
    }.get(status, 0.2)
    return {
        "task_id": _group_id(item),
        "task_description": _task_description(item),
        "status": status,
        "hard_mean": round(hard_mean, 6),
        "contrast_strength": round(contrast_strength, 6),
        "confidence": confidence,
        "success_count": int(summary.get("success_count", bool(item.get("hard")))),
        "completed_count": int(summary.get("completed_count", 1) or 1),
    }


def collect_group_records(items: list[dict]) -> dict[str, dict[str, Any]]:
    """Return one evidence record per source task, never per rollout."""
    records: dict[str, dict[str, Any]] = {}
    for item in items:
        record = _group_record(item)
        if record["task_id"]:
            records.setdefault(record["task_id"], record)
    return records


def format_group_relative_analyst_context(items: list[dict]) -> str:
    """Build compact task-normalized evidence instructions for an analyst."""
    records = collect_group_records(items)
    if not records:
        return ""
    rows = []
    for record in records.values():
        rows.append(
            "- task_id={task_id!r}: status={status}, hard_mean={hard_mean:.4f}, "
            "contrast_strength={contrast_strength:.4f}, outcomes={success_count}/"
            "{completed_count} successful".format(**record)
        )
    return (
        "## Group-Relative Credit Assignment\n"
        "The K sibling rollouts of one task are one correlated evidence unit, "
        "not K independent supports. For a mixed group, compare the positive-"
        "advantage and negative-advantage trajectories and base the edit on their "
        "first causal behavioral divergence. Stable-failure groups provide only "
        "low-confidence repair hypotheses; stable-success groups provide only "
        "low-confidence preservation evidence.\n"
        "For every proposed edit, add `evidence_task_ids` containing only exact "
        "task_id values below that directly support the rule. Also add "
        "`evidence_kind` (`contrastive`, `consensus`, or `exploratory`) and "
        "`generality_scope` (`cross_task` or `single_task`). Do not copy document "
        "names, entities, dates, identifiers, or numeric answers into a rule when "
        "the underlying behavior can be stated abstractly.\n"
        + "\n".join(rows)
    )


ANALYST_SUFFIX = """

## Same-Task Group-Relative Evidence Contract

Treat sibling rollouts with the same task_id as one evidence source. A repeated
failure from one task is not cross-task prevalence. For mixed groups, compare
the strongest positive-advantage outcome with the strongest negative-advantage
outcome and identify the first causal behavioral divergence. Prefer a rule only
when it describes an inference-time observable decision. Phrase the behavior at
the broadest scope justified by the evidence, without copying task literals.

Each proposed item must include:
- `evidence_task_ids`: exact source task_id strings that independently support it;
- `evidence_kind`: `contrastive`, `consensus`, or `exploratory`;
- `generality_scope`: `cross_task` or `single_task`.

The executable content must be a complete behavior rule, not a heading, topic
label, sentence fragment, or numbered lead-in. State an inference-time trigger,
the action to take, and an observable way to check the result. Keep a Markdown
heading attached to the rule body it describes.

For an item supported by a mixed-success sibling group, also include:
`causal_trace`: {
  `observable_trigger`, `first_divergent_decision`, `successful_behavior`,
  `failed_behavior`, `verification_signal`
}.
Every field must describe evidence visible in the supplied trajectories. Do not
use evaluator labels, hidden references, or post-hoc correctness as a trigger.

Do not claim support from every task in the minibatch. Use only tasks whose
trajectory evidence directly supports that specific item.
"""


V20_ANALYST_SUFFIX = """

## Evidence-Atomic Policy Contract

Write one independently testable condition-action rule per Markdown bullet. Do
not hide several normalization, formatting, or decision rules inside one edit.
Cross-task support is a ranking signal, not a minimum candidate quota: keep the
best directly supported rule even when fewer than two cross-task rules exist.

Separate reasoning policy from answer rendering. An edit that changes the final
answer string, units, symbols, articles, titles, suffixes, capitalization, or
other output formatting must be justified by a mixed same-task trajectory whose
complete causal trace shows that this rendering decision changed the outcome.
When the task exposes an exact answer span, preserve that span verbatim unless
the supplied trajectory evidence directly demonstrates a required transform.
"""


def augment_group_relative_analyst_prompt(system_prompt: str) -> str:
    prompt = system_prompt.rstrip() + "\n" + ANALYST_SUFFIX
    if any(
        (
            bool(_RUNTIME.get("soft_cross_task_target", False)),
            bool(_RUNTIME.get("atomize_all_edit_clauses", False)),
            bool(
                _RUNTIME.get(
                    "output_contract_require_direct_causal_evidence", False
                )
            ),
        )
    ):
        prompt += V20_ANALYST_SUFFIX
    return prompt


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_LONG_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d{3,}(?:\.\d+)?(?![\w.])")
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z]{2,}(?:[-_]\d{2,})+(?:[-_][A-Za-z0-9]+)*|"
    r"[A-Z]{2,}\d{2,})\b"
)
_QUOTED_LITERAL_RE = re.compile(r"[`\"'][^`\"'\n]{5,}[`\"']")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for",
    "from", "if", "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "then", "this", "to", "use", "using", "when", "with",
}


def _behavior_text(item: dict, update_mode: str) -> str:
    if is_full_rewrite_minibatch_mode(update_mode):
        return str(item.get("new_skill") or "")
    if is_rewrite_mode(update_mode):
        return str(item.get("instruction") or "")
    return str(item.get("content") or item.get("target") or "")


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _has_task_phrase_overlap(text: str, task_descriptions: list[str]) -> bool:
    edit_tokens = _tokens(text)
    if len(edit_tokens) < 4:
        return False
    edit_ngrams = {
        tuple(edit_tokens[i : i + 4])
        for i in range(len(edit_tokens) - 3)
    }
    for description in task_descriptions:
        task_tokens = _tokens(description)
        task_ngrams = {
            tuple(task_tokens[i : i + 4])
            for i in range(max(0, len(task_tokens) - 3))
        }
        if edit_ngrams & task_ngrams:
            return True
    return False


def compute_specificity_penalty(
    item: dict,
    *,
    update_mode: str = "patch",
    task_descriptions: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Estimate whether an edit copies instance literals instead of a policy."""
    text = _behavior_text(item, update_mode)
    flags: list[str] = []
    penalty = 0.0
    if _YEAR_RE.search(text):
        flags.append("literal_year")
        penalty += 0.2
    if _LONG_NUMBER_RE.search(text):
        flags.append("literal_number")
        penalty += 0.25
    if _IDENTIFIER_RE.search(text):
        flags.append("literal_identifier")
        penalty += 0.35
    if _QUOTED_LITERAL_RE.search(text):
        flags.append("quoted_literal")
        penalty += 0.15
    if _has_task_phrase_overlap(text, task_descriptions or []):
        flags.append("task_phrase_overlap")
        penalty += 0.35
    return min(1.0, round(penalty, 6)), flags


def _validated_evidence_ids(
    item: dict,
    records: dict[str, dict[str, Any]],
) -> list[str]:
    raw = item.get("evidence_task_ids")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    evidence_ids = []
    for value in raw:
        task_id = str(value)
        if task_id in records and task_id not in evidence_ids:
            evidence_ids.append(task_id)
    if not evidence_ids and len(records) == 1:
        evidence_ids = list(records)
    return evidence_ids


def annotate_analyst_patch(
    result: dict,
    items: list[dict],
    *,
    update_mode: str = "patch",
) -> dict:
    """Attach task-normalized evidence to each analyst-proposed item."""
    if not isinstance(result, dict):
        return result
    patch = result.get("patch")
    if not isinstance(patch, dict):
        return result
    records = collect_group_records(items)
    descriptions = [record["task_description"] for record in records.values()]
    provenance_task_ids = sorted(records)
    provenance_rollout_ids_set: set[str] = set()
    for source in items:
        source_ids = source.get("_outcome_stratified_source_rollout_ids")
        if not isinstance(source_ids, list):
            source_ids = [source.get("id")]
        provenance_rollout_ids_set.update(
            str(source_id)
            for source_id in source_ids
            if str(source_id or "")
        )
    provenance_rollout_ids = sorted(provenance_rollout_ids_set)
    for item in get_payload_items(patch, update_mode):
        evidence_ids = _validated_evidence_ids(item, records)
        evidence = [records[task_id] for task_id in evidence_ids]
        specificity, specificity_flags = compute_specificity_penalty(
            item,
            update_mode=update_mode,
            task_descriptions=descriptions,
        )
        mixed_count = sum(record["status"] == "mixed" for record in evidence)
        contrast = (
            sum(record["contrast_strength"] for record in evidence) / len(evidence)
            if evidence
            else 0.0
        )
        confidence_support = sum(record["confidence"] for record in evidence)
        causal_trace = normalize_causal_trace(item.get("causal_trace"))
        causal_complete = causal_trace_is_complete(causal_trace)
        if causal_trace:
            item["causal_trace"] = causal_trace
        item["evidence_task_ids"] = evidence_ids
        item["provenance_task_ids"] = provenance_task_ids
        item["provenance_rollout_ids"] = provenance_rollout_ids
        item["task_support_count"] = len(evidence_ids)
        item["_group_relative_credit"] = {
            "task_support_count": len(evidence_ids),
            "mixed_task_support_count": mixed_count,
            "mean_contrast_strength": round(contrast, 6),
            "confidence_support": round(confidence_support, 6),
            "specificity_penalty": specificity,
            "specificity_flags": specificity_flags,
            "evidence_statuses": {
                task_id: records[task_id]["status"] for task_id in evidence_ids
            },
            "evidence_contrast_strengths": {
                task_id: records[task_id]["contrast_strength"]
                for task_id in evidence_ids
            },
            "causal_trace_complete": causal_complete,
            "causal_trace_task_ids": (
                [
                    task_id
                    for task_id in evidence_ids
                    if records[task_id]["status"] == "mixed"
                ]
                if causal_complete
                else []
            ),
            "provenance_task_ids": provenance_task_ids,
            "provenance_rollout_ids": provenance_rollout_ids,
        }
    result["group_relative_source_tasks"] = list(records.values())
    return result


def _source_items(patches: list[dict], update_mode: str) -> list[dict]:
    items = []
    for patch in patches:
        items.extend(get_payload_items(patch, update_mode))
    return [item for item in items if isinstance(item, dict)]


_MARKDOWN_HEADING_RE = re.compile(
    r"(?m)^\s{0,3}#{1,6}\s+.+?\s*$"
)
_MARKDOWN_LIST_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_MARKDOWN_LIST_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:[-*+]|\d+[.)])\s+"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\s*;\s+")
_DANGLING_ENUM_RE = re.compile(r"(?:^|\s)\d+[.)]\s*$")
_ACTION_CUE_RE = re.compile(
    r"\b(?:if|when|whenever|before|after|unless|until|must|should|ensure|"
    r"verify|check|compare|confirm|preserve|avoid|use|read|write|select|"
    r"inspect|compute|derive|infer|identify|extract|return|report|answer|"
    r"open|close|move|place|search|retry|stop|validate|test|keep|remove|"
    r"insert|delete|update|match|normalize|parse|calculate|execute)\b",
    re.IGNORECASE,
)
_CAUSAL_TRACE_FIELDS = (
    "observable_trigger",
    "first_divergent_decision",
    "successful_behavior",
    "failed_behavior",
    "verification_signal",
)

_OUTPUT_CONTRACT_PATTERNS = (
    re.compile(
        r"\b(?:final answer|answer string|output format|response format|"
        r"exact string|verbatim span|return only|report only|raw number|"
        r"bare number|full name|short form)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:answer|output|response|result)\b.{0,80}"
        r"\b(?:unit|units|symbol|symbols|percent sign|percentage sign|"
        r"currency|article|articles|honorific|title|suffix|prefix|"
        r"capitalization|punctuation|comma|commas|decimal|formatting)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:include|omit|strip|remove|preserve|normalize)\b.{0,80}"
        r"\b(?:unit|units|symbol|symbols|percent sign|percentage sign|"
        r"currency symbol|article|articles|honorific|title|suffix|prefix|"
        r"capitalization|punctuation|comma|commas)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


def classify_behavior_kind(text: str) -> str:
    """Classify rendering rules separately from task reasoning rules."""
    value = str(text or "")
    if any(pattern.search(value) for pattern in _OUTPUT_CONTRACT_PATTERNS):
        return "output_contract"
    return "reasoning_policy"


def _split_top_level_markdown_list(text: str) -> list[str]:
    """Split sibling Markdown items while preserving nested continuations."""
    value = str(text or "").strip()
    if not value or "```" in value:
        return []
    lines = value.splitlines()
    markers: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = _MARKDOWN_LIST_LINE_RE.match(line)
        if match is None:
            continue
        indent = len(match.group("indent").expandtabs(4))
        markers.append((index, indent))
    if len(markers) < 2:
        return []
    base_indent = min(indent for _, indent in markers)
    starts = [index for index, indent in markers if indent == base_indent]
    if len(starts) < 2:
        return []

    prefix = "\n".join(lines[: starts[0]]).strip()
    units: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        item = "\n".join(lines[start:end]).strip()
        if prefix:
            item = f"{prefix}\n{item}"
        if item:
            units.append(item)
    return units


def _split_unheaded_behavior(text: str) -> list[str]:
    """Split prose without breaking Markdown lists or dependent procedures."""
    units: list[str] = []
    for raw_block in re.split(r"\n\s*\n", str(text or "").strip()):
        block = raw_block.strip()
        if not block:
            continue
        if (
            bool(_RUNTIME.get("atomize_all_edit_clauses", False))
            and _MARKDOWN_LIST_RE.search(block)
            and "```" not in block
        ):
            list_units = _split_top_level_markdown_list(block)
            if list_units:
                units.extend(list_units)
                continue
        if _MARKDOWN_LIST_RE.search(block) or "```" in block:
            units.append(block)
            continue
        block_units: list[str] = []
        flattened = " ".join(line.strip() for line in block.splitlines())
        pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY_RE.split(flattened)]
        pending = ""
        for piece in pieces:
            if not piece:
                continue
            if pending:
                piece = f"{pending} {piece}".strip()
                pending = ""
            if piece.endswith(":") or len(_tokens(piece)) < 3:
                pending = piece
                continue
            block_units.append(piece)
        if pending:
            if block_units:
                block_units[-1] = f"{block_units[-1]} {pending}".strip()
            else:
                block_units.append(pending)
        units.extend(block_units)
    return units


def split_semantic_behavior_units(text: str) -> list[str]:
    """Return complete behavior units while preserving Markdown structure.

    A heading always stays attached to its body. In evidence-atomic mode,
    sibling list clauses become separate units carrying the same heading;
    nested procedures and fenced examples remain intact.
    """
    normalized = str(text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []
    headings = list(_MARKDOWN_HEADING_RE.finditer(normalized))
    if not headings:
        return _split_unheaded_behavior(normalized) or [normalized]

    units = _split_unheaded_behavior(normalized[: headings[0].start()])
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(normalized)
        heading = match.group(0).strip()
        body = normalized[match.end() : end].strip()
        if body and bool(_RUNTIME.get("atomize_all_edit_clauses", False)):
            body_units = _split_unheaded_behavior(body)
            if len(body_units) > 1:
                units.extend(f"{heading}\n{unit}" for unit in body_units)
                continue
        units.append(f"{heading}\n{body}".strip() if body else heading)
    return units


def _join_semantic_behavior_units(units: list[str]) -> str:
    """Rebuild content after unsupported clauses are removed."""
    sections: list[str] = []
    current_heading = ""
    current_bodies: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_bodies
        if current_heading:
            body = "\n".join(part for part in current_bodies if part).strip()
            sections.append(
                f"{current_heading}\n{body}".strip() if body else current_heading
            )
        elif current_bodies:
            sections.extend(part for part in current_bodies if part)
        current_heading = ""
        current_bodies = []

    for unit in units:
        lines = str(unit or "").strip().splitlines()
        heading = (
            lines[0].strip()
            if lines and _MARKDOWN_HEADING_RE.fullmatch(lines[0])
            else ""
        )
        body = "\n".join(lines[1:] if heading else lines).strip()
        if heading != current_heading:
            flush()
            current_heading = heading
        current_bodies.append(body)
    flush()
    return "\n\n".join(section for section in sections if section).strip()


def assess_behavior_actionability(text: str) -> dict[str, Any]:
    """Reject structural fragments without encoding benchmark-specific rules."""
    value = str(text or "").strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    body_lines = [line for line in lines if not _MARKDOWN_HEADING_RE.fullmatch(line)]
    body = " ".join(body_lines).strip()
    plain = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", body).strip()
    tokens = _tokens(plain)
    issues: list[str] = []
    if not value:
        issues.append("empty_behavior")
    if lines and not body_lines:
        issues.append("heading_without_body")
    if body and len(tokens) < 4:
        issues.append("too_short_to_execute")
    if body.endswith(":"):
        issues.append("dangling_lead_in")
    if _DANGLING_ENUM_RE.search(body):
        issues.append("dangling_enumerator")

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", plain)
    title_ratio = (
        sum(word[:1].isupper() for word in words) / len(words) if words else 0.0
    )
    has_terminal_sentence = bool(re.search(r"[.!?]\s*$", plain))
    has_action_cue = bool(_ACTION_CUE_RE.search(plain))
    if (
        2 <= len(words) <= 12
        and title_ratio >= 0.7
        and not has_terminal_sentence
        and not has_action_cue
    ):
        issues.append("title_like_fragment")
    if (
        body
        and len(tokens) <= 12
        and not has_terminal_sentence
        and not has_action_cue
        and "\n" not in body
    ):
        issues.append("no_executable_predicate")
    return {
        "accepted": not issues,
        "issues": sorted(set(issues)),
        "token_count": len(tokens),
        "has_action_cue": has_action_cue,
    }


def normalize_causal_trace(value: Any) -> dict[str, str]:
    """Normalize the generic mixed-trajectory causal evidence contract."""
    if not isinstance(value, dict):
        return {}
    return {
        field: str(value.get(field) or "").strip()
        for field in _CAUSAL_TRACE_FIELDS
        if str(value.get(field) or "").strip()
    }


def causal_trace_is_complete(value: Any) -> bool:
    trace = normalize_causal_trace(value)
    return all(trace.get(field) for field in _CAUSAL_TRACE_FIELDS)


def split_atomic_behavior_clauses(text: str) -> list[str]:
    """Backward-compatible name for the semantic-unit parser."""
    return split_semantic_behavior_units(text)


def _item_with_behavior(item: dict, behavior: str, update_mode: str) -> dict:
    atom = dict(item)
    if is_full_rewrite_minibatch_mode(update_mode):
        atom["new_skill"] = behavior
    elif is_rewrite_mode(update_mode):
        atom["instruction"] = behavior
    else:
        atom["content"] = behavior
        atom.pop("target", None)
    return atom


def _atomize_append_items(
    items: list[dict],
    *,
    update_mode: str,
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Split multi-clause append edits into independently executable edits.

    Replacement and insertion edits retain their original structure because
    duplicating their anchors would change patch semantics. Append edits have
    no such coupling, so clause-level splitting lets each rule earn provenance,
    selection, probing, and quarantine decisions independently.
    """
    if is_rewrite_mode(update_mode) or is_full_rewrite_minibatch_mode(
        update_mode
    ):
        return list(items), []

    expanded: list[dict] = []
    events: list[dict[str, Any]] = []
    for source_index, item in enumerate(items):
        clauses = split_atomic_behavior_clauses(
            _behavior_text(item, update_mode)
        )
        if str(item.get("op") or "").strip().lower() != "append" or len(
            clauses
        ) <= 1:
            expanded.append(item)
            continue
        atomized_indices = []
        for atom_index, clause in enumerate(clauses):
            atom = _item_with_behavior(item, clause, update_mode)
            atom["_group_relative_actionability"] = (
                assess_behavior_actionability(clause)
            )
            atom["_group_relative_original_merged_index"] = source_index
            atom["_group_relative_atom_index"] = atom_index
            atom["_group_relative_atom_count"] = len(clauses)
            atomized_indices.append(len(expanded))
            expanded.append(atom)
        events.append(
            {
                "original_merged_index": source_index,
                "original_behavior": _behavior_text(item, update_mode),
                "atomic_behaviors": clauses,
                "output_indices": atomized_indices,
            }
        )
    return expanded, events


def _prune_unsupported_patch_clauses(
    item: dict,
    credit: dict[str, Any],
    *,
    update_mode: str,
) -> tuple[dict, dict[str, Any] | None]:
    """Prune unsupported clauses without duplicating replace/insert anchors."""
    if is_rewrite_mode(update_mode) or is_full_rewrite_minibatch_mode(
        update_mode
    ):
        return item, None
    if str(item.get("op") or "").strip().lower() == "append":
        return item, None

    clauses = split_atomic_behavior_clauses(
        _behavior_text(item, update_mode)
    )
    atomic_support = credit.get("atomic_support", [])
    if (
        len(clauses) <= 1
        or not isinstance(atomic_support, list)
        or len(atomic_support) != len(clauses)
    ):
        return item, None

    kept: list[str] = []
    rejected: list[dict[str, Any]] = []
    output_contract_gate = bool(
        _RUNTIME.get(
            "output_contract_require_direct_causal_evidence", False
        )
    )
    for clause, atom in zip(clauses, atomic_support):
        reasons: list[str] = []
        if int(atom.get("task_support_count", 0) or 0) <= 0:
            reasons.append("no_direct_training_evidence")
        if not atom.get("actionability", {}).get("accepted", False):
            reasons.append("unactionable_clause")
        if (
            output_contract_gate
            and atom.get("behavior_kind") == "output_contract"
            and not atom.get("output_contract_evidence_accepted", False)
        ):
            reasons.append("output_contract_without_causal_divergence")
        if reasons:
            rejected.append(
                {
                    "behavior": clause,
                    "behavior_kind": atom.get("behavior_kind"),
                    "reasons": reasons,
                    "evidence_task_ids": atom.get("evidence_task_ids", []),
                    "causal_trace_task_ids": atom.get(
                        "causal_trace_task_ids", []
                    ),
                }
            )
        else:
            kept.append(clause)

    if not rejected:
        return item, None
    event = {
        "op": str(item.get("op") or ""),
        "target": str(item.get("target") or ""),
        "original_behavior": _behavior_text(item, update_mode),
        "original_clause_count": len(clauses),
        "retained_clause_count": len(kept),
        "rejected_clause_count": len(rejected),
        "retained_behaviors": kept,
        "rejected": rejected,
        "applied": bool(kept),
    }
    if not kept:
        return item, event

    pruned = dict(item)
    pruned["content"] = _join_semantic_behavior_units(kept)
    pruned["_group_relative_original_behavior"] = event[
        "original_behavior"
    ]
    pruned["_group_relative_pruned_clause_count"] = len(rejected)
    return pruned, event


def _source_atomic_records(
    source_items: list[dict],
    *,
    update_mode: str,
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for source_index, source in enumerate(source_items):
        source_credit = source.get("_group_relative_credit")
        if not isinstance(source_credit, dict):
            source_credit = {}
        evidence_ids = source.get("evidence_task_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        provenance_ids = source.get("provenance_task_ids", [])
        if isinstance(provenance_ids, str):
            provenance_ids = [provenance_ids]
        causal_complete = bool(source_credit.get("causal_trace_complete", False))
        causal_task_ids = {
            str(value)
            for value in source_credit.get("causal_trace_task_ids", [])
        }
        contrast_by_task = {
            str(task_id): float(value or 0.0)
            for task_id, value in source_credit.get(
                "evidence_contrast_strengths", {}
            ).items()
        } if isinstance(
            source_credit.get("evidence_contrast_strengths", {}), dict
        ) else {}
        for atom_index, behavior in enumerate(
            split_atomic_behavior_clauses(_behavior_text(source, update_mode))
        ):
            actionability = assess_behavior_actionability(behavior)
            if (
                bool(_RUNTIME.get("require_actionable_units", True))
                and not actionability["accepted"]
            ):
                continue
            atom_item = _item_with_behavior(source, behavior, update_mode)
            specificity, specificity_flags = compute_specificity_penalty(
                atom_item,
                update_mode=update_mode,
            )
            atom_key = f"{source_index}:{atom_index}:{behavior.casefold()}"
            atoms.append(
                {
                    "atom_id": hashlib.sha1(atom_key.encode("utf-8")).hexdigest()[:12],
                    "source_edit_index": source_index,
                    "source_atom_index": atom_index,
                    "behavior": behavior,
                    "item": atom_item,
                    "evidence_task_ids": sorted({str(v) for v in evidence_ids}),
                    "provenance_task_ids": sorted({str(v) for v in provenance_ids}),
                    "evidence_statuses": dict(
                        source_credit.get("evidence_statuses", {})
                    ),
                    "mean_contrast_strength": float(
                        source_credit.get("mean_contrast_strength", 0.0) or 0.0
                    ),
                    "evidence_contrast_strengths": contrast_by_task,
                    "causal_trace_complete": causal_complete,
                    "causal_trace_task_ids": sorted(causal_task_ids),
                    "actionability": actionability,
                    "specificity_penalty": specificity,
                    "specificity_flags": specificity_flags,
                }
            )
    return atoms


def _merged_atom_id(merged_index: int, atom_index: int) -> str:
    return f"merged_{merged_index:04d}_atom_{atom_index:03d}"


def _optimizer_model_semantic_matches(
    merged_items: list[dict],
    source_atoms: list[dict[str, Any]],
    *,
    update_mode: str,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Ask the optimizer model to attribute merged clauses to source atoms.

    The model may only select opaque IDs supplied by the framework. Task IDs
    are recovered from those source records after parsing, so generated or
    copied provenance claims can never create support credit.
    """
    merged_atoms = [
        {
            "merged_atom_id": _merged_atom_id(merged_index, atom_index),
            "behavior": behavior,
        }
        for merged_index, item in enumerate(merged_items)
        for atom_index, behavior in enumerate(
            split_atomic_behavior_clauses(_behavior_text(item, update_mode))
        )
    ]
    source_payload = [
        {
            "source_atom_id": atom["atom_id"],
            "behavior": atom["behavior"],
        }
        for atom in source_atoms
    ]
    empty_audit = {
        "enabled": True,
        "matcher": "optimizer_model",
        "status": "no_matchable_atoms",
        "merged_atom_count": len(merged_atoms),
        "source_atom_count": len(source_atoms),
        "accepted_match_count": 0,
        "accepted_matches": [],
        "invalid_rows": [],
        "ambiguous_source_atom_ids": [],
    }
    if not merged_atoms or not source_atoms:
        return {}, empty_audit

    system = """You perform semantic evidence attribution for skill edits.
Return JSON only. For every merged behavior clause, select source_atom_ids that
directly support the entire clause. A direct match requires the same observable
trigger or precondition, a compatible action, and the same intended outcome or
verification behavior. Topic similarity alone is not support. Do not match a
source clause when the merged clause adds an unsupported action, exception, or
output requirement. Do not invent IDs, and assign each source_atom_id to at
most one merged_atom_id.

Return this schema:
{
  "matches": [
    {
      "merged_atom_id": "one supplied merged ID",
      "source_atom_ids": ["zero or more supplied source IDs"],
      "rationale": "brief semantic justification"
    }
  ]
}
"""
    user = (
        "## Merged clauses\n"
        + json.dumps(merged_atoms, ensure_ascii=False, indent=2)
        + "\n\n## Source taskwise clauses\n"
        + json.dumps(source_payload, ensure_ascii=False, indent=2)
    )
    try:
        from textualrl.model import chat_optimizer
        from textualrl.utils import extract_json

        response, _ = chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="group_relative_semantic_match",
        )
        parsed = extract_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("optimizer matcher returned no JSON object")
    except Exception as exc:  # noqa: BLE001
        audit = dict(empty_audit)
        audit.update(
            {
                "status": "model_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return {}, audit

    valid_merged_ids = {row["merged_atom_id"] for row in merged_atoms}
    valid_source_ids = {row["source_atom_id"] for row in source_payload}
    normalized_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    source_claims: dict[str, set[str]] = defaultdict(set)
    raw_rows = parsed.get("matches", [])
    if not isinstance(raw_rows, list):
        raw_rows = []
        invalid_rows.append({"reason": "matches_not_a_list"})
    for row_index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            invalid_rows.append(
                {"row_index": row_index, "reason": "match_not_an_object"}
            )
            continue
        merged_id = str(row.get("merged_atom_id") or "")
        source_ids = row.get("source_atom_ids", [])
        if not isinstance(source_ids, list):
            invalid_rows.append(
                {"row_index": row_index, "reason": "source_ids_not_a_list"}
            )
            continue
        unknown_source_ids = sorted(
            {
                str(source_id)
                for source_id in source_ids
                if str(source_id) not in valid_source_ids
            }
        )
        if merged_id not in valid_merged_ids or unknown_source_ids:
            invalid_rows.append(
                {
                    "row_index": row_index,
                    "reason": (
                        "unknown_merged_atom_id"
                        if merged_id not in valid_merged_ids
                        else "unknown_source_atom_ids"
                    ),
                    "merged_atom_id": merged_id,
                    "unknown_source_atom_ids": unknown_source_ids,
                }
            )
            continue
        accepted_source_ids = sorted(
            {str(source_id) for source_id in source_ids}
        )
        normalized_rows.append(
            {
                "merged_atom_id": merged_id,
                "source_atom_ids": accepted_source_ids,
                "rationale": str(row.get("rationale") or "").strip(),
            }
        )
        for source_id in accepted_source_ids:
            source_claims[source_id].add(merged_id)

    ambiguous_source_ids = {
        source_id
        for source_id, merged_ids in source_claims.items()
        if len(merged_ids) > 1
    }
    matches: dict[str, set[str]] = defaultdict(set)
    accepted_rows = []
    for row in normalized_rows:
        accepted_source_ids = [
            source_id
            for source_id in row["source_atom_ids"]
            if source_id not in ambiguous_source_ids
        ]
        if accepted_source_ids:
            matches[row["merged_atom_id"]].update(accepted_source_ids)
        accepted_rows.append(
            {
                **row,
                "source_atom_ids": accepted_source_ids,
                "removed_ambiguous_source_atom_ids": sorted(
                    set(row["source_atom_ids"]) & ambiguous_source_ids
                ),
            }
        )

    audit = {
        "enabled": True,
        "matcher": "optimizer_model",
        "status": "ok",
        "merged_atom_count": len(merged_atoms),
        "source_atom_count": len(source_atoms),
        "accepted_match_count": sum(len(ids) for ids in matches.values()),
        "accepted_matches": accepted_rows,
        "invalid_rows": invalid_rows,
        "ambiguous_source_atom_ids": sorted(ambiguous_source_ids),
        "model_output": parsed,
    }
    return dict(matches), audit


def _cluster_source_atoms(
    atoms: list[dict[str, Any]],
    *,
    update_mode: str,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Cluster equivalent source clauses while retaining task provenance."""
    clusters: list[dict[str, Any]] = []
    for atom in atoms:
        best_cluster = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = max(
                behavior_similarity(
                    atom["item"], member["item"], update_mode=update_mode
                )
                for member in cluster["members"]
            )
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_cluster = cluster
                best_similarity = similarity
        if best_cluster is None:
            best_cluster = {
                "cluster_id": f"atomic_{len(clusters):04d}",
                "members": [],
            }
            clusters.append(best_cluster)
        best_cluster["members"].append(atom)

    for cluster in clusters:
        members = cluster["members"]
        cluster["evidence_task_ids"] = sorted(
            {
                task_id
                for member in members
                for task_id in member["evidence_task_ids"]
            }
        )
        cluster["provenance_task_ids"] = sorted(
            {
                task_id
                for member in members
                for task_id in member["provenance_task_ids"]
            }
        )
        cluster["source_edit_indices"] = sorted(
            {member["source_edit_index"] for member in members}
        )
    return clusters


def _credit_for_merged_item(
    item: dict,
    source_clusters: list[dict[str, Any]],
    *,
    merged_item_index: int,
    update_mode: str,
    similarity_threshold: float,
    direct_evidence_similarity_threshold: float,
    specificity_penalty_weight: float,
    model_semantic_matches: dict[str, set[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    provenance_ids: list[str] = []
    statuses: dict[str, str] = {}
    confidence_by_task: dict[str, float] = {}
    contrast_by_task: dict[str, float] = {}
    specificity_flags: set[str] = set()
    source_specificities: list[float] = []
    atomic_support: list[dict[str, Any]] = []

    merged_atoms = split_atomic_behavior_clauses(_behavior_text(item, update_mode))
    for merged_atom_index, merged_behavior in enumerate(merged_atoms):
        merged_atom_id = _merged_atom_id(merged_item_index, merged_atom_index)
        model_source_ids = (
            model_semantic_matches.get(merged_atom_id, set())
            if model_semantic_matches is not None
            else None
        )
        merged_atom = _item_with_behavior(item, merged_behavior, update_mode)
        actionability = assess_behavior_actionability(merged_behavior)
        behavior_kind = classify_behavior_kind(merged_behavior)
        atom_matches: list[dict[str, Any]] = []
        atom_evidence: set[str] = set()
        atom_provenance: set[str] = set()
        atom_causal_trace_tasks: set[str] = set()
        atom_statuses: dict[str, str] = {}
        atom_contrast: dict[str, float] = {}
        for cluster in (
            source_clusters
            if actionability["accepted"]
            or not bool(_RUNTIME.get("require_actionable_units", True))
            else []
        ):
            if model_source_ids is not None:
                member_similarities = [
                    (source_atom, None)
                    for source_atom in cluster["members"]
                    if source_atom["atom_id"] in model_source_ids
                ]
                if not member_similarities:
                    continue
                cluster_similarity = None
            else:
                member_similarities = [
                    (
                        source_atom,
                        behavior_similarity(
                            merged_atom,
                            source_atom["item"],
                            update_mode=update_mode,
                        ),
                    )
                    for source_atom in cluster["members"]
                ]
                cluster_similarity = max(
                    (similarity for _, similarity in member_similarities),
                    default=0.0,
                )
                if cluster_similarity < similarity_threshold:
                    continue
            for source_atom, similarity in member_similarities:
                direct_match = (
                    model_source_ids is not None
                    or similarity >= direct_evidence_similarity_threshold
                )
                if not direct_match:
                    atom_matches.append(
                        {
                            "source_atom_id": source_atom["atom_id"],
                            "source_cluster_id": cluster["cluster_id"],
                            "source_edit_index": source_atom["source_edit_index"],
                            "similarity": round(similarity, 6),
                            "cluster_similarity": round(cluster_similarity, 6),
                            "credit_granted": False,
                            "match_kind": "cluster_only",
                            "evidence_task_ids": [],
                            "provenance_task_ids": [],
                            "behavior": source_atom["behavior"],
                        }
                    )
                    continue
                atom_evidence.update(source_atom["evidence_task_ids"])
                atom_provenance.update(source_atom["provenance_task_ids"])
                for task_id in source_atom["evidence_task_ids"]:
                    status = str(
                        source_atom["evidence_statuses"].get(task_id, "unknown")
                    )
                    atom_statuses[task_id] = status
                    statuses[task_id] = status
                    confidence_by_task[task_id] = max(
                        confidence_by_task.get(task_id, 0.0),
                        {
                            "mixed": 1.0,
                            "stable_success": 0.35,
                            "stable_failure": 0.25,
                        }.get(status, 0.2),
                    )
                    contrast = float(
                        source_atom.get("evidence_contrast_strengths", {}).get(
                            task_id,
                            source_atom["mean_contrast_strength"],
                        )
                        or 0.0
                    )
                    atom_contrast[task_id] = max(
                        atom_contrast.get(task_id, 0.0), contrast
                    )
                    contrast_by_task[task_id] = max(
                        contrast_by_task.get(task_id, 0.0), contrast
                    )
                    if task_id in set(
                        source_atom.get("causal_trace_task_ids", [])
                    ):
                        atom_causal_trace_tasks.add(task_id)
                source_specificities.append(
                    float(source_atom["specificity_penalty"] or 0.0)
                )
                specificity_flags.update(source_atom["specificity_flags"])
                atom_matches.append(
                    {
                        "source_atom_id": source_atom["atom_id"],
                        "source_cluster_id": cluster["cluster_id"],
                        "source_edit_index": source_atom["source_edit_index"],
                        "similarity": (
                            round(similarity, 6)
                            if similarity is not None
                            else None
                        ),
                        "cluster_similarity": (
                            round(cluster_similarity, 6)
                            if cluster_similarity is not None
                            else None
                        ),
                        "credit_granted": True,
                        "match_kind": (
                            "optimizer_model"
                            if model_source_ids is not None
                            else "direct"
                        ),
                        "evidence_task_ids": source_atom["evidence_task_ids"],
                        "provenance_task_ids": source_atom["provenance_task_ids"],
                        "behavior": source_atom["behavior"],
                    }
                )
        evidence_ids.extend(
            task_id for task_id in sorted(atom_evidence) if task_id not in evidence_ids
        )
        provenance_ids.extend(
            task_id
            for task_id in sorted(atom_provenance)
            if task_id not in provenance_ids
        )
        matches.extend(atom_matches)
        atomic_support.append(
            {
                "merged_atom_id": merged_atom_id,
                "merged_atom_index": merged_atom_index,
                "behavior": merged_behavior,
                "evidence_task_ids": sorted(atom_evidence),
                "provenance_task_ids": sorted(atom_provenance),
                "task_support_count": len(atom_evidence),
                "behavior_kind": behavior_kind,
                "actionability": actionability,
                "direct_evidence_task_ids": sorted(atom_evidence),
                "direct_evidence_task_support_count": len(atom_evidence),
                "causal_trace_task_ids": sorted(atom_causal_trace_tasks),
                "causal_trace_task_support_count": len(
                    atom_causal_trace_tasks
                ),
                "mixed_task_support_count": sum(
                    status == "mixed" for status in atom_statuses.values()
                ),
                "stable_failure_task_support_count": sum(
                    status == "stable_failure"
                    for status in atom_statuses.values()
                ),
                "stable_success_task_support_count": sum(
                    status == "stable_success"
                    for status in atom_statuses.values()
                ),
                "confidence_support": round(
                    sum(
                        {
                            "mixed": 1.0,
                            "stable_success": 0.35,
                            "stable_failure": 0.25,
                        }.get(status, 0.2)
                        for status in atom_statuses.values()
                    ),
                    6,
                ),
                "mean_contrast_strength": round(
                    sum(atom_contrast.values()) / len(atom_contrast)
                    if atom_contrast else 0.0,
                    6,
                ),
                "evidence_contrast_strengths": {
                    task_id: round(value, 6)
                    for task_id, value in sorted(atom_contrast.items())
                },
                "source_matches": atom_matches,
                "output_contract_evidence_accepted": (
                    behavior_kind != "output_contract"
                    or bool(atom_causal_trace_tasks)
                ),
            }
        )

    # The merge model may copy or invent task IDs. Keep those claims for the
    # audit, but never turn them into credit without a direct clause match.
    claimed_evidence_ids = item.get("evidence_task_ids", [])
    if isinstance(claimed_evidence_ids, str):
        claimed_evidence_ids = [claimed_evidence_ids]
    if not isinstance(claimed_evidence_ids, list):
        claimed_evidence_ids = []
    claimed_provenance_ids = item.get("provenance_task_ids", [])
    if isinstance(claimed_provenance_ids, str):
        claimed_provenance_ids = [claimed_provenance_ids]
    if not isinstance(claimed_provenance_ids, list):
        claimed_provenance_ids = []

    merged_specificity, merged_flags = compute_specificity_penalty(
        item,
        update_mode=update_mode,
    )
    specificity_flags.update(merged_flags)
    specificity = max([merged_specificity, *source_specificities], default=0.0)
    for atom in atomic_support:
        atom_raw_score = (
            float(atom.get("confidence_support", 0.0) or 0.0)
            + 0.35 * max(0, int(atom.get("task_support_count", 0) or 0) - 1)
            + 0.5 * float(atom.get("mean_contrast_strength", 0.0) or 0.0)
        )
        atom["atomic_credit_score"] = round(
            atom_raw_score - specificity_penalty_weight * specificity,
            6,
        )
    mixed_count = sum(status == "mixed" for status in statuses.values())
    max_atomic_support = max(
        (atom["task_support_count"] for atom in atomic_support), default=0
    )
    max_atomic_mixed_support = max(
        (atom["mixed_task_support_count"] for atom in atomic_support), default=0
    )
    output_contract_gate = bool(
        _RUNTIME.get(
            "output_contract_require_direct_causal_evidence", False
        )
    )

    def _atom_has_accepted_evidence(atom: dict[str, Any]) -> bool:
        if int(atom.get("task_support_count", 0) or 0) <= 0:
            return False
        if (
            output_contract_gate
            and atom.get("behavior_kind") == "output_contract"
            and not atom.get("output_contract_evidence_accepted", False)
        ):
            return False
        return True

    supported_atomic_count = sum(
        _atom_has_accepted_evidence(atom) for atom in atomic_support
    )
    supported_atomic_fraction = (
        supported_atomic_count / len(atomic_support) if atomic_support else 0.0
    )
    actionable_atomic_count = sum(
        bool(atom.get("actionability", {}).get("accepted", False))
        for atom in atomic_support
    )
    actionable_atomic_fraction = (
        actionable_atomic_count / len(atomic_support) if atomic_support else 0.0
    )
    causal_trace_task_ids = sorted(
        {
            str(task_id)
            for atom in atomic_support
            for task_id in atom.get("causal_trace_task_ids", [])
        }
    )
    causal_trace_supported_atomic_count = sum(
        int(atom.get("causal_trace_task_support_count", 0) or 0) > 0
        for atom in atomic_support
    )
    output_contract_atomic_count = sum(
        atom.get("behavior_kind") == "output_contract"
        for atom in atomic_support
    )
    output_contract_supported_atomic_count = sum(
        atom.get("behavior_kind") == "output_contract"
        and _atom_has_accepted_evidence(atom)
        for atom in atomic_support
    )
    cross_threshold = max(
        1, int(_RUNTIME.get("min_cross_task_support", 2) or 2)
    )
    cross_supported_atomic_count = sum(
        int(atom.get("task_support_count", 0) or 0) >= cross_threshold
        for atom in atomic_support
    )
    contrastive_supported_atomic_count = sum(
        int(atom.get("mixed_task_support_count", 0) or 0) > 0
        for atom in atomic_support
    )
    stable_failure_supported_atomic_count = sum(
        int(atom.get("stable_failure_task_support_count", 0) or 0) > 0
        for atom in atomic_support
    )
    stable_success_supported_atomic_count = sum(
        int(atom.get("stable_success_task_support_count", 0) or 0) > 0
        for atom in atomic_support
    )

    def _atomic_fraction(count: int) -> float:
        return count / len(atomic_support) if atomic_support else 0.0

    mean_contrast = (
        sum(contrast_by_task.values()) / len(contrast_by_task)
        if contrast_by_task
        else 0.0
    )
    confidence_support = sum(confidence_by_task.values())
    task_support_count = len(evidence_ids)
    atomic_scores = [
        float(atom.get("atomic_credit_score", 0.0) or 0.0)
        for atom in atomic_support
    ]
    weakest_atomic_score = min(atomic_scores, default=0.0)
    mean_atomic_score = (
        sum(atomic_scores) / len(atomic_scores) if atomic_scores else 0.0
    )
    credit_aggregation = str(
        _RUNTIME.get("credit_aggregation", "weakest")
    )
    if credit_aggregation == "mean":
        score = mean_atomic_score
    else:
        score = weakest_atomic_score + 0.25 * supported_atomic_fraction
    credit = {
        "evidence_task_ids": sorted(evidence_ids),
        "provenance_task_ids": sorted(provenance_ids),
        "claimed_evidence_task_ids": sorted(
            {str(value) for value in claimed_evidence_ids}
        ),
        "claimed_provenance_task_ids": sorted(
            {str(value) for value in claimed_provenance_ids}
        ),
        "task_support_count": task_support_count,
        "mixed_task_support_count": mixed_count,
        "max_atomic_task_support_count": max_atomic_support,
        "max_atomic_mixed_task_support_count": max_atomic_mixed_support,
        "supported_atomic_count": supported_atomic_count,
        "atomic_clause_count": len(atomic_support),
        "supported_atomic_fraction": round(supported_atomic_fraction, 6),
        "actionable_atomic_count": actionable_atomic_count,
        "actionable_atomic_fraction": round(actionable_atomic_fraction, 6),
        "actionability_issues": sorted(
            {
                str(issue)
                for atom in atomic_support
                for issue in atom.get("actionability", {}).get("issues", [])
            }
        ),
        "causal_trace_task_ids": causal_trace_task_ids,
        "causal_trace_task_support_count": len(causal_trace_task_ids),
        "causal_trace_supported_atomic_count": (
            causal_trace_supported_atomic_count
        ),
        "causal_trace_supported_atomic_fraction": round(
            _atomic_fraction(causal_trace_supported_atomic_count), 6
        ),
        "output_contract_evidence_gate": output_contract_gate,
        "output_contract_atomic_count": output_contract_atomic_count,
        "output_contract_supported_atomic_count": (
            output_contract_supported_atomic_count
        ),
        "output_contract_supported_atomic_fraction": round(
            (
                output_contract_supported_atomic_count
                / output_contract_atomic_count
            )
            if output_contract_atomic_count
            else 1.0,
            6,
        ),
        "cross_task_support_threshold": cross_threshold,
        "cross_supported_atomic_count": cross_supported_atomic_count,
        "cross_supported_atomic_fraction": round(
            _atomic_fraction(cross_supported_atomic_count), 6
        ),
        "contrastive_supported_atomic_count": (
            contrastive_supported_atomic_count
        ),
        "contrastive_supported_atomic_fraction": round(
            _atomic_fraction(contrastive_supported_atomic_count), 6
        ),
        "stable_failure_supported_atomic_count": (
            stable_failure_supported_atomic_count
        ),
        "stable_failure_supported_atomic_fraction": round(
            _atomic_fraction(stable_failure_supported_atomic_count), 6
        ),
        "stable_success_supported_atomic_count": (
            stable_success_supported_atomic_count
        ),
        "stable_success_supported_atomic_fraction": round(
            _atomic_fraction(stable_success_supported_atomic_count), 6
        ),
        "weakest_atomic_credit_score": round(weakest_atomic_score, 6),
        "mean_atomic_credit_score": round(mean_atomic_score, 6),
        "credit_aggregation": credit_aggregation,
        "mean_contrast_strength": round(mean_contrast, 6),
        "evidence_contrast_strengths": {
            task_id: round(value, 6)
            for task_id, value in sorted(contrast_by_task.items())
        },
        "confidence_support": round(confidence_support, 6),
        "specificity_penalty": round(specificity, 6),
        "specificity_flags": sorted(specificity_flags),
        "evidence_statuses": statuses,
        "credit_score": round(score, 6),
        "matched_source_edit_count": len(
            {match["source_edit_index"] for match in matches}
        ),
        "atomic_support": atomic_support,
    }
    return credit, matches


def annotate_merged_patch(
    merged_patch: dict,
    source_patches: list[dict],
    *,
    update_mode: str = "patch",
    semantic_similarity_threshold: float | None = None,
    direct_evidence_similarity_threshold: float | None = None,
    specificity_penalty_weight: float | None = None,
    atomize_append_edits: bool | None = None,
    atomize_all_edit_clauses: bool | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Recover cross-task support after LLM merge and return an audit."""
    threshold = float(
        _RUNTIME["semantic_similarity_threshold"]
        if semantic_similarity_threshold is None
        else semantic_similarity_threshold
    )
    direct_threshold = float(
        _RUNTIME["direct_evidence_similarity_threshold"]
        if direct_evidence_similarity_threshold is None
        else direct_evidence_similarity_threshold
    )
    direct_threshold = max(threshold, direct_threshold)
    penalty_weight = float(
        _RUNTIME["specificity_penalty_weight"]
        if specificity_penalty_weight is None
        else specificity_penalty_weight
    )
    atomize_append = bool(
        _RUNTIME["atomize_append_edits"]
        if atomize_append_edits is None
        else atomize_append_edits
    )
    atomize_all = bool(
        _RUNTIME["atomize_all_edit_clauses"]
        if atomize_all_edit_clauses is None
        else atomize_all_edit_clauses
    )
    # Clause-level evidence accounting and patch-shape rewriting are separate
    # decisions.  ``atomize_all`` controls how source and merged behaviors are
    # scored; only ``atomize_append`` is allowed to expand one append edit into
    # several independently applied edits.  Keeping the bundle intact lets a
    # coherent multi-clause procedure consume one edit-budget slot while every
    # clause still has to earn its own evidence credit.
    atomize_appends_effective = atomize_append
    sources = _source_items(source_patches, update_mode)
    source_semantic_units = [
        behavior
        for source in sources
        for behavior in split_atomic_behavior_clauses(
            _behavior_text(source, update_mode)
        )
    ]
    rejected_source_units = [
        {
            "behavior": behavior,
            "actionability": assess_behavior_actionability(behavior),
        }
        for behavior in source_semantic_units
        if not assess_behavior_actionability(behavior)["accepted"]
    ]
    source_atoms = _source_atomic_records(sources, update_mode=update_mode)
    cluster_threshold = max(0.65, threshold)
    model_semantic_matching = bool(
        _RUNTIME.get("model_semantic_matching", False)
    )
    if model_semantic_matching:
        source_clusters = [
            {
                "cluster_id": f"model_source_{atom['atom_id']}",
                "members": [atom],
                "evidence_task_ids": list(atom["evidence_task_ids"]),
                "provenance_task_ids": list(atom["provenance_task_ids"]),
                "source_edit_indices": [atom["source_edit_index"]],
            }
            for atom in source_atoms
        ]
    else:
        source_clusters = _cluster_source_atoms(
            source_atoms,
            update_mode=update_mode,
            similarity_threshold=cluster_threshold,
        )
    original_items = list(get_payload_items(merged_patch, update_mode))
    merged_items, atomization_events = (
        _atomize_append_items(original_items, update_mode=update_mode)
        if atomize_appends_effective
        else (original_items, [])
    )
    payload_changed = bool(atomization_events)
    if model_semantic_matching:
        model_matches, model_match_audit = _optimizer_model_semantic_matches(
            merged_items,
            source_atoms,
            update_mode=update_mode,
        )
    else:
        model_matches = None
        model_match_audit = {
            "enabled": False,
            "matcher": "deterministic_similarity",
            "status": "disabled",
        }
    clause_pruning_events: list[dict[str, Any]] = []
    audit_items = []
    for index, item in enumerate(merged_items):
        credit, matches = _credit_for_merged_item(
            item,
            source_clusters,
            merged_item_index=index,
            update_mode=update_mode,
            similarity_threshold=threshold,
            direct_evidence_similarity_threshold=direct_threshold,
            specificity_penalty_weight=penalty_weight,
            model_semantic_matches=model_matches,
        )
        if atomize_all:
            pruned_item, prune_event = _prune_unsupported_patch_clauses(
                item,
                credit,
                update_mode=update_mode,
            )
            if prune_event is not None:
                prune_event["merged_index"] = index
                clause_pruning_events.append(prune_event)
            if pruned_item is not item:
                item = pruned_item
                merged_items[index] = item
                payload_changed = True
                credit, matches = _credit_for_merged_item(
                    item,
                    source_clusters,
                    merged_item_index=index,
                    update_mode=update_mode,
                    similarity_threshold=threshold,
                    direct_evidence_similarity_threshold=direct_threshold,
                    specificity_penalty_weight=penalty_weight,
                    model_semantic_matches=model_matches,
                )
        evidence_ids = list(credit["evidence_task_ids"])
        item["evidence_task_ids"] = evidence_ids
        item["provenance_task_ids"] = list(credit["provenance_task_ids"])
        item["task_support_count"] = len(evidence_ids)
        item["_group_relative_credit"] = credit
        audit_items.append(
            {
                "merged_index": index,
                "behavior": _behavior_text(item, update_mode),
                "evidence_task_ids": evidence_ids,
                "credit": credit,
                "source_matches": matches,
            }
        )
    if payload_changed:
        set_payload_items(merged_patch, merged_items, update_mode)
    audit = {
        "enabled": True,
        "scope": "training_rollouts_only",
        "validation_changed": False,
        "semantic_similarity_threshold": threshold,
        "direct_evidence_similarity_threshold": direct_threshold,
        "atomic_cluster_similarity_threshold": cluster_threshold,
        "semantic_matcher": (
            "optimizer_model"
            if model_semantic_matching
            else "deterministic_similarity"
        ),
        "model_semantic_match_audit": model_match_audit,
        "credit_aggregation": str(
            _RUNTIME.get("credit_aggregation", "weakest")
        ),
        "specificity_penalty_weight": penalty_weight,
        "atomize_append_edits": atomize_appends_effective,
        "configured_atomize_append_edits": atomize_append,
        "atomize_all_edit_clauses": atomize_all,
        "bundle_preserving_clause_credit": bool(
            atomize_all and not atomize_appends_effective
        ),
        "output_contract_require_direct_causal_evidence": bool(
            _RUNTIME.get(
                "output_contract_require_direct_causal_evidence", False
            )
        ),
        "original_merged_edit_count": len(original_items),
        "atomized_append_edit_count": len(atomization_events),
        "atomization_events": atomization_events,
        "clause_pruned_edit_count": sum(
            bool(event.get("applied")) for event in clause_pruning_events
        ),
        "rejected_clause_count": sum(
            int(event.get("rejected_clause_count", 0) or 0)
            for event in clause_pruning_events
        ),
        "clause_pruning_events": clause_pruning_events,
        "source_edit_count": len(sources),
        "source_semantic_unit_count": len(source_semantic_units),
        "rejected_source_semantic_unit_count": len(rejected_source_units),
        "rejected_source_semantic_units": rejected_source_units,
        "source_atomic_clause_count": len(source_atoms),
        "source_atomic_cluster_count": len(source_clusters),
        "merged_edit_count": len(audit_items),
        "source_atomic_clusters": [
            {
                "cluster_id": cluster["cluster_id"],
                "member_count": len(cluster["members"]),
                "behaviors": [member["behavior"] for member in cluster["members"]],
                "evidence_task_ids": cluster["evidence_task_ids"],
                "provenance_task_ids": cluster["provenance_task_ids"],
                "source_edit_indices": cluster["source_edit_indices"],
            }
            for cluster in source_clusters
        ],
        "items": audit_items,
    }
    merged_patch["group_relative_credit_audit"] = audit
    return merged_patch, audit


def select_group_relative_candidate(
    patch: dict,
    *,
    max_edits: int,
    update_mode: str = "patch",
    max_specific_singletons: int | None = None,
    min_cross_task_support: int | None = None,
    max_exploratory: int | None = None,
    min_atomic_support_fraction: float | None = None,
    min_credit_score: float | None = None,
    min_candidate_cross_task_edits: int | None = None,
    soft_cross_task_target: bool | None = None,
    max_contrastive_edits: int | None = None,
    max_singleton_edits: int | None = None,
    block_singleton_stable_hypotheses: bool | None = None,
    actionable_priority_bonus: float | None = None,
    causal_trace_priority_bonus: float | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Select evidence-backed edits without forcing the candidate to be full."""
    limit = int(
        _RUNTIME["max_specific_singletons"]
        if max_specific_singletons is None
        else max_specific_singletons
    )
    cross_threshold = int(
        _RUNTIME["min_cross_task_support"]
        if min_cross_task_support is None
        else min_cross_task_support
    )
    exploratory_limit = int(
        _RUNTIME["max_exploratory"]
        if max_exploratory is None
        else max_exploratory
    )
    atomic_support_threshold = float(
        _RUNTIME["min_atomic_support_fraction"]
        if min_atomic_support_fraction is None
        else min_atomic_support_fraction
    )
    atomic_support_threshold = min(
        1.0, max(0.0, atomic_support_threshold)
    )
    credit_floor = float(
        _RUNTIME["min_credit_score"]
        if min_credit_score is None
        else min_credit_score
    )
    required_cross_edits = int(
        _RUNTIME["min_candidate_cross_task_edits"]
        if min_candidate_cross_task_edits is None
        else min_candidate_cross_task_edits
    )
    required_cross_edits = max(0, required_cross_edits)
    soft_cross_target = bool(
        _RUNTIME["soft_cross_task_target"]
        if soft_cross_task_target is None
        else soft_cross_task_target
    )
    contrastive_limit = int(
        _RUNTIME["max_contrastive_edits"]
        if max_contrastive_edits is None
        else max_contrastive_edits
    )
    contrastive_limit = max(-1, contrastive_limit)
    singleton_limit = int(
        _RUNTIME["max_singleton_edits"]
        if max_singleton_edits is None
        else max_singleton_edits
    )
    singleton_limit = max(-1, singleton_limit)
    block_singleton_stable = bool(
        _RUNTIME["block_singleton_stable_hypotheses"]
        if block_singleton_stable_hypotheses is None
        else block_singleton_stable_hypotheses
    )
    actionable_bonus = max(
        0.0,
        float(
            _RUNTIME["actionable_priority_bonus"]
            if actionable_priority_bonus is None
            else actionable_priority_bonus
        ),
    )
    causal_bonus = max(
        0.0,
        float(
            _RUNTIME["causal_trace_priority_bonus"]
            if causal_trace_priority_bonus is None
            else causal_trace_priority_bonus
        ),
    )
    items = list(get_payload_items(patch, update_mode))

    def quality_fractions(item: dict) -> tuple[float, float]:
        credit = item.get("_group_relative_credit", {})
        actionable_fraction = float(
            credit.get("actionable_atomic_fraction", 1.0) or 0.0
        )
        causal_fraction = float(
            credit.get("causal_trace_supported_atomic_fraction", 0.0)
            or 0.0
        )
        atomic = credit.get("atomic_support", [])
        if isinstance(atomic, list) and atomic:
            causal_fraction = sum(
                int(atom.get("causal_trace_task_support_count", 0) or 0) > 0
                for atom in atomic
            ) / len(atomic)
        return actionable_fraction, causal_fraction

    def category(item: dict) -> str:
        credit = item.get("_group_relative_credit", {})
        coverage = float(credit.get("supported_atomic_fraction", 1.0) or 0.0)
        actionable_fraction = float(
            credit.get("actionable_atomic_fraction", 1.0) or 0.0
        )
        if (
            bool(_RUNTIME.get("require_actionable_units", True))
            and actionable_fraction + 1e-12 < 1.0
        ):
            return "unactionable"
        atomic = credit.get("atomic_support", [])
        if (
            bool(
                _RUNTIME.get(
                    "output_contract_require_direct_causal_evidence", False
                )
            )
            and isinstance(atomic, list)
            and any(
                atom.get("behavior_kind") == "output_contract"
                and not atom.get(
                    "output_contract_evidence_accepted", False
                )
                for atom in atomic
            )
        ):
            return "unsupported_output_contract"
        if coverage + 1e-12 < atomic_support_threshold:
            return "unsupported_atomic"
        if isinstance(atomic, list) and atomic:
            cross_fraction = sum(
                int(atom.get("task_support_count", 0) or 0) >= cross_threshold
                for atom in atomic
            ) / len(atomic)
            contrastive_fraction = sum(
                int(atom.get("mixed_task_support_count", 0) or 0) > 0
                for atom in atomic
            ) / len(atomic)
            causal_trace_fraction = sum(
                int(atom.get("causal_trace_task_support_count", 0) or 0) > 0
                for atom in atomic
            ) / len(atomic)
            stable_failure_fraction = sum(
                int(atom.get("stable_failure_task_support_count", 0) or 0) > 0
                for atom in atomic
            ) / len(atomic)
            stable_success_fraction = sum(
                int(atom.get("stable_success_task_support_count", 0) or 0) > 0
                for atom in atomic
            ) / len(atomic)
        else:
            # Backward-compatible fallback for old audits and hand-built tests.
            cross_fraction = (
                coverage
                if int(
                    credit.get(
                        "max_atomic_task_support_count",
                        credit.get("task_support_count", 0),
                    )
                    or 0
                )
                >= cross_threshold
                else 0.0
            )
            contrastive_fraction = (
                coverage
                if int(
                    credit.get(
                        "max_atomic_mixed_task_support_count",
                        credit.get("mixed_task_support_count", 0),
                    )
                    or 0
                )
                > 0
                else 0.0
            )
            causal_trace_fraction = float(
                credit.get("causal_trace_supported_atomic_fraction", 0.0)
                or 0.0
            )
            statuses = set(
                str(value)
                for value in credit.get("evidence_statuses", {}).values()
            )
            stable_failure_fraction = (
                coverage if "stable_failure" in statuses else 0.0
            )
            stable_success_fraction = (
                coverage if "stable_success" in statuses else 0.0
            )
        if cross_fraction + 1e-12 >= atomic_support_threshold:
            return "cross_task"
        if contrastive_fraction + 1e-12 >= atomic_support_threshold:
            if (
                bool(_RUNTIME.get("require_causal_trace", True))
                and causal_trace_fraction + 1e-12 < atomic_support_threshold
            ):
                return "uncorroborated_contrast"
            return "contrastive"
        if stable_failure_fraction + 1e-12 >= atomic_support_threshold:
            return "stable_hypothesis"
        if stable_success_fraction + 1e-12 >= atomic_support_threshold:
            return "preservation"
        return "exploratory"

    def priority(item: dict) -> tuple[float, int, float, float, int]:
        credit = item.get("_group_relative_credit", {})
        actionable_fraction, causal_trace_fraction = quality_fractions(item)
        evidence_quality_score = (
            float(credit.get("credit_score", 0.0) or 0.0)
            + actionable_bonus * actionable_fraction
            + causal_bonus * causal_trace_fraction
        )
        return (
            evidence_quality_score,
            int(credit.get("task_support_count", 0) or 0),
            causal_trace_fraction,
            actionable_fraction,
            -int(credit.get("specificity_penalty", 0.0) >= 0.5),
        )

    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        item_category = category(item)
        item["_group_relative_selection_category"] = item_category
        buckets[item_category].append(item)
    for bucket in buckets.values():
        bucket.sort(key=priority, reverse=True)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    specific_singletons = 0
    singleton_edits = 0
    contrastive_edits = 0
    dropped = []
    item_indices = {id(item): index for index, item in enumerate(items)}

    def try_add(item: dict) -> bool:
        nonlocal contrastive_edits, singleton_edits, specific_singletons
        if id(item) in selected_ids or len(selected) >= max_edits:
            return False
        credit = item.get("_group_relative_credit", {})
        item_category = str(
            item.get("_group_relative_selection_category") or category(item)
        )
        is_singleton = int(credit.get("task_support_count", 0) or 0) <= 1
        is_specific_singleton = (
            is_singleton
            and float(credit.get("specificity_penalty", 0.0) or 0.0) >= 0.5
        )
        if (
            block_singleton_stable
            and is_singleton
            and item_category == "stable_hypothesis"
        ):
            dropped.append(
                {
                    "input_index": item_indices[id(item)],
                    "reason": "singleton_stable_hypothesis_blocked",
                    "category": item_category,
                    "credit": credit,
                    "behavior": _behavior_text(item, update_mode),
                }
            )
            return False
        if (
            item_category == "contrastive"
            and contrastive_limit >= 0
            and contrastive_edits >= contrastive_limit
        ):
            dropped.append(
                {
                    "input_index": item_indices[id(item)],
                    "reason": "contrastive_edit_quota",
                    "category": item_category,
                    "credit": credit,
                    "behavior": _behavior_text(item, update_mode),
                }
            )
            return False
        if is_singleton and singleton_limit >= 0:
            if singleton_edits >= singleton_limit:
                dropped.append(
                    {
                        "input_index": item_indices[id(item)],
                        "reason": "singleton_edit_quota",
                        "category": item_category,
                        "credit": credit,
                        "behavior": _behavior_text(item, update_mode),
                    }
                )
                return False
        elif is_specific_singleton and specific_singletons >= limit:
            dropped.append(
                {
                    "input_index": item_indices[id(item)],
                    "reason": "specific_singleton_quota",
                    "category": item_category,
                    "credit": credit,
                    "behavior": _behavior_text(item, update_mode),
                }
            )
            return False
        selected.append(item)
        selected_ids.add(id(item))
        if item_category == "contrastive":
            contrastive_edits += 1
        if is_singleton:
            singleton_edits += 1
        if is_specific_singleton:
            specific_singletons += 1
        return True

    def has_positive_credit(item: dict) -> bool:
        return (
            float(
                item.get("_group_relative_credit", {}).get(
                    "credit_score", 0.0
                )
                or 0.0
            )
            > credit_floor
        )

    cross_candidates = [
        item for item in buckets.get("cross_task", []) if has_positive_credit(item)
    ]
    contrastive_candidates = [
        item
        for item in buckets.get("contrastive", [])
        if has_positive_credit(item)
    ]
    candidate_rejection_reasons: list[str] = []
    if not soft_cross_target and required_cross_edits > max_edits:
        candidate_rejection_reasons.append(
            "cross_task_quota_exceeds_edit_budget"
        )
    elif (
        not soft_cross_target
        and len(cross_candidates) < required_cross_edits
    ):
        candidate_rejection_reasons.append(
            "insufficient_cross_task_edits"
        )

    seeded_categories: list[str] = []
    exploratory_added = 0
    lower_confidence = sorted(
        buckets.get("stable_hypothesis", [])
        + buckets.get("exploratory", []),
        key=priority,
        reverse=True,
    )
    if not candidate_rejection_reasons:
        if required_cross_edits > 0:
            # Reserve up to the requested number of slots for the strongest
            # cross-task edits. A shortfall is not a rejection in soft mode;
            # every remaining eligible category then competes by credit.
            for item in cross_candidates[:required_cross_edits]:
                if try_add(item):
                    seeded_categories.append("cross_task")
            fallback_candidates = sorted(
                cross_candidates[required_cross_edits:]
                + contrastive_candidates
                + lower_confidence,
                key=priority,
                reverse=True,
            )
            for item in fallback_candidates:
                if len(selected) >= max_edits:
                    break
                item_category = str(
                    item.get("_group_relative_selection_category")
                    or category(item)
                )
                if (
                    item_category in {"stable_hypothesis", "exploratory"}
                    and exploratory_added >= exploratory_limit
                ):
                    continue
                credit = item.get("_group_relative_credit", {})
                if (
                    int(credit.get("task_support_count", 0) or 0) < 1
                    or not has_positive_credit(item)
                ):
                    continue
                if try_add(item):
                    if item_category in {
                        "stable_hypothesis",
                        "exploratory",
                    }:
                        exploratory_added += 1
        else:
            # Backward-compatible V1.8 behavior when no candidate-level
            # cross-task quota is configured.
            evidence_backed = sorted(
                cross_candidates + contrastive_candidates,
                key=priority,
                reverse=True,
            )
            for category_name, candidates in (
                ("cross_task", cross_candidates),
                ("contrastive", contrastive_candidates),
            ):
                if candidates and len(selected) < max_edits:
                    if try_add(candidates[0]):
                        seeded_categories.append(category_name)
            for item in evidence_backed:
                if len(selected) >= max_edits:
                    break
                try_add(item)
            for item in lower_confidence:
                if (
                    len(selected) >= max_edits
                    or exploratory_added >= exploratory_limit
                ):
                    break
                credit = item.get("_group_relative_credit", {})
                if (
                    int(credit.get("task_support_count", 0) or 0) < 1
                    or not has_positive_credit(item)
                ):
                    continue
                if try_add(item):
                    exploratory_added += 1

    selected_cross_task_count = sum(
        item.get("_group_relative_selection_category") == "cross_task"
        for item in selected
    )
    if (
        not soft_cross_target
        and selected_cross_task_count < required_cross_edits
    ):
        if "insufficient_cross_task_edits" not in candidate_rejection_reasons:
            candidate_rejection_reasons.append(
                "selected_cross_task_quota_not_met"
            )

    for item in items:
        if id(item) in selected_ids:
            continue
        item_category = category(item)
        credit = item.get("_group_relative_credit", {})
        if candidate_rejection_reasons:
            reason = candidate_rejection_reasons[0]
        elif float(credit.get("credit_score", 0.0) or 0.0) <= credit_floor:
            reason = "non_positive_advantage_credit"
        elif item_category == "exploratory" and (
            int(credit.get("task_support_count", 0) or 0) < 1
        ):
            reason = "insufficient_independent_support"
        elif item_category == "unsupported_atomic":
            reason = "unsupported_atomic_clause"
        elif item_category == "unsupported_output_contract":
            reason = "output_contract_without_direct_causal_evidence"
        elif item_category == "unactionable":
            reason = "unactionable_semantic_unit"
        elif item_category == "uncorroborated_contrast":
            reason = "missing_causal_trace"
        elif item_category == "preservation":
            reason = "stable_success_not_causal"
        elif (
            item_category == "contrastive"
            and contrastive_limit >= 0
            and contrastive_edits >= contrastive_limit
        ):
            reason = "contrastive_edit_quota"
        elif (
            int(credit.get("task_support_count", 0) or 0) <= 1
            and singleton_limit >= 0
            and singleton_edits >= singleton_limit
        ):
            reason = "singleton_edit_quota"
        elif item_category in {
            "stable_hypothesis",
            "exploratory",
        } and exploratory_added >= exploratory_limit:
            reason = "exploratory_limit"
        elif len(selected) >= max_edits:
            reason = "edit_budget"
        else:
            reason = "adaptive_evidence_gate"
        if not any(
            entry.get("input_index") == item_indices[id(item)]
            for entry in dropped
        ):
            dropped.append(
                {
                    "input_index": item_indices[id(item)],
                    "reason": reason,
                    "category": item_category,
                    "credit": credit,
                    "behavior": _behavior_text(item, update_mode),
                }
            )

    result = dict(patch)
    set_payload_items(result, selected, update_mode)
    strict_composition_enabled = any(
        (
            required_cross_edits > 0,
            contrastive_limit >= 0,
            singleton_limit >= 0,
            block_singleton_stable,
            actionable_bonus > 0.0,
            causal_bonus > 0.0,
        )
    )
    evidence_atomic_policy = any(
        (
            soft_cross_target,
            bool(_RUNTIME.get("atomize_all_edit_clauses", False)),
            bool(
                _RUNTIME.get(
                    "output_contract_require_direct_causal_evidence", False
                )
            ),
        )
    )
    audit = {
        "policy": (
            "evidence_atomic_progressive_retention_v20"
            if evidence_atomic_policy
            else (
                "cross_task_retention_candidate_gate_v19"
                if strict_composition_enabled
                else "benchmark_isolated_causal_semantic_gate_v18"
            )
        ),
        "validation_changed": False,
        "candidate_eligible": not candidate_rejection_reasons,
        "candidate_rejection_reasons": candidate_rejection_reasons,
        "input_count": len(items),
        "selected_count": len(selected),
        "max_edits": max_edits,
        "max_specific_singletons": limit,
        "max_singleton_edits": singleton_limit,
        "selected_singleton_count": singleton_edits,
        "min_cross_task_support": cross_threshold,
        "min_candidate_cross_task_edits": required_cross_edits,
        "soft_cross_task_target": soft_cross_target,
        "cross_task_target_met": (
            selected_cross_task_count >= required_cross_edits
        ),
        "cross_task_target_shortfall": max(
            0, required_cross_edits - selected_cross_task_count
        ),
        "available_cross_task_edit_count": len(cross_candidates),
        "selected_cross_task_edit_count": selected_cross_task_count,
        "max_contrastive_edits": contrastive_limit,
        "selected_contrastive_edit_count": contrastive_edits,
        "max_exploratory": exploratory_limit,
        "block_singleton_stable_hypotheses": block_singleton_stable,
        "actionable_priority_bonus": actionable_bonus,
        "causal_trace_priority_bonus": causal_bonus,
        "min_atomic_support_fraction": atomic_support_threshold,
        "min_credit_score": credit_floor,
        "input_category_counts": {
            name: len(buckets.get(name, []))
            for name in (
                "cross_task",
                "contrastive",
                "stable_hypothesis",
                "preservation",
                "exploratory",
                "unsupported_atomic",
                "unsupported_output_contract",
                "unactionable",
                "uncorroborated_contrast",
            )
        },
        "selected_categories": [category(item) for item in selected],
        "seeded_evidence_categories": seeded_categories,
        "dropped": dropped,
    }
    result["group_relative_selection"] = audit
    return result, audit
