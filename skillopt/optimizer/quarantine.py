"""Candidate quarantine, fuzzy edit families, and edit audit persistence."""
from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import os
import re
from typing import Any

from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    set_payload_items,
)


CANDIDATE_SCOPE = "candidate_edit_set"
PERMANENT_EDIT_SCOPE = "permanent_edit_family"
EDIT_HISTORY_VERSION = "edit_history_v2"

UNOBSERVABLE_RUNTIME_PATTERNS = (
    (
        "gold_target",
        re.compile(
            r"\b(?:gold|golden)\s+(?:answer|answers|label|labels|solution|solutions|output|response)\b",
            re.IGNORECASE,
        ),
    ),
    ("gold_standard", re.compile(r"\bgold\s+standard\b", re.IGNORECASE)),
    (
        "ground_truth",
        re.compile(
            r"\bground[\s-]*truth(?:\s+(?:answer|answers|label|labels|solution|solutions|output|response))?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reference_target",
        re.compile(
            r"\breference\s+(?:answer|answers|label|labels|solution|solutions|output|response)\b",
            re.IGNORECASE,
        ),
    ),
    ("answer_key", re.compile(r"\banswer\s+key\b", re.IGNORECASE)),
    (
        "evaluator_feedback",
        re.compile(
            r"\b(?:evaluator|grader|judge)(?:\s+s)?\s+(?:feedback|result|results|score|scores|decision|verdict)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "evaluation_signal",
        re.compile(
            r"\b(?:evaluation|training|validation|test)\s+"
            r"(?:answer|answers|feedback|label|labels|result|results|score|scores|metric|metrics)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "evaluation_expectation",
        re.compile(
            r"\b(?:expected|required|preferred)\s+by\s+(?:the\s+)?"
            r"(?:evaluation|evaluator|grader|judge)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reward_signal",
        re.compile(r"\breward\s+(?:signal|signals|score|scores)\b", re.IGNORECASE),
    ),
    (
        "hidden_evaluation",
        re.compile(
            r"\bhidden\s+(?:answer|answers|label|labels|test|tests|evaluation|evaluator)\b",
            re.IGNORECASE,
        ),
    ),
)

NEGATED_DEPENDENCY_RE = re.compile(
    r"\b(?:avoid|cannot|do\s+not|does\s+not|ignore|must\s+not|never|"
    r"should\s+not|without)\b(?:\W+\w+){0,8}\W*$",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").lower()))


def edit_semantic_payload(
    item: dict, update_mode: str = "patch"
) -> dict[str, Any]:
    """Keep behavior-defining fields and discard support/priority metadata."""
    if not isinstance(item, dict):
        return {}
    if is_full_rewrite_minibatch_mode(update_mode):
        return {
            "op": "full_rewrite",
            "title": _normalize_text(item.get("title")),
            "change_summary": _normalize_text(" ".join(item.get("change_summary", []))),
            "new_skill": _normalize_text(item.get("new_skill")),
        }
    if is_rewrite_mode(update_mode):
        return {
            "op": _normalize_text(item.get("type") or "suggestion"),
            "title": _normalize_text(item.get("title")),
            "instruction": _normalize_text(item.get("instruction")),
        }

    op = _normalize_text(item.get("op") or "unknown")
    payload: dict[str, Any] = {"op": op}
    if op in {"replace", "delete"} or item.get("target"):
        payload["target"] = _normalize_text(item.get("target"))
    if op in {"append", "replace"} or item.get("content"):
        payload["content"] = _normalize_text(item.get("content"))
    return payload


def edit_behavior_payload(
    item: dict, update_mode: str = "patch"
) -> dict[str, str]:
    """Describe what an edit asks the agent to do, independent of patch syntax."""
    if not isinstance(item, dict):
        return {}
    if is_full_rewrite_minibatch_mode(update_mode):
        text = _normalize_text(item.get("new_skill"))
        return {"kind": "full_rewrite", "text": text} if text else {}
    if is_rewrite_mode(update_mode):
        text = _normalize_text(item.get("instruction"))
        return {"kind": "instruction", "text": text} if text else {}

    content = _normalize_text(item.get("content"))
    if content:
        return {"kind": "instruction", "text": content}
    target = _normalize_text(item.get("target"))
    if target:
        return {"kind": "deletion", "text": target}
    return {}


def _raw_behavior_text(item: dict, update_mode: str) -> str:
    if not isinstance(item, dict):
        return ""
    if is_full_rewrite_minibatch_mode(update_mode):
        return str(item.get("new_skill") or "")
    if is_rewrite_mode(update_mode):
        return str(item.get("instruction") or "")
    return str(item.get("content") or "")


def _dependency_is_negated(text: str, match_start: int) -> bool:
    clause_prefix = re.split(r"[.!?;:\n]", text[:match_start])[-1]
    return bool(NEGATED_DEPENDENCY_RE.search(clause_prefix))


def normalize_edit(item: dict, update_mode: str = "patch") -> str:
    return json.dumps(
        edit_semantic_payload(item, update_mode),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_edit(item: dict, update_mode: str = "patch") -> str:
    payload = edit_semantic_payload(item, update_mode)
    return " | ".join(
        f"{key}={value}" for key, value in payload.items() if value
    )


def fingerprint_edit(item: dict, update_mode: str = "patch") -> str:
    normalized = normalize_edit(item, update_mode)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _text_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_union = left_tokens | right_tokens
    jaccard = (
        len(left_tokens & right_tokens) / len(token_union)
        if token_union
        else 0.0
    )
    sequence = SequenceMatcher(None, left, right).ratio()
    return max(jaccard, sequence)


def edit_similarity(
    left: dict,
    right: dict,
    *,
    update_mode: str = "patch",
) -> float:
    """Compare semantic edit content while ignoring non-behavioral metadata."""
    left_payload = edit_semantic_payload(left, update_mode)
    right_payload = edit_semantic_payload(right, update_mode)
    if not left_payload or not right_payload:
        return 0.0
    if left_payload.get("op") != right_payload.get("op"):
        return 0.0
    fields = sorted((set(left_payload) | set(right_payload)) - {"op"})
    if not fields:
        return 1.0
    return min(
        _text_similarity(
            str(left_payload.get(field, "")),
            str(right_payload.get(field, "")),
        )
        for field in fields
    )


def behavior_similarity(
    left: dict,
    right: dict,
    *,
    update_mode: str = "patch",
) -> float:
    """Compare behavioral instructions while ignoring patch op and target."""
    left_payload = edit_behavior_payload(left, update_mode)
    right_payload = edit_behavior_payload(right, update_mode)
    if not left_payload or not right_payload:
        return 0.0
    if left_payload.get("kind") != right_payload.get("kind"):
        return 0.0
    return _text_similarity(left_payload.get("text", ""), right_payload.get("text", ""))


def find_unobservable_runtime_edits(
    items: list[dict],
    *,
    update_mode: str = "patch",
) -> list[dict[str, Any]]:
    """Find hard dependencies on signals unavailable to the deployed policy."""
    hits = []
    for item in items:
        behavior = edit_behavior_payload(item, update_mode)
        if behavior.get("kind") == "deletion":
            continue
        behavior_text = _raw_behavior_text(item, update_mode)
        matched_reasons = []
        for name, pattern in UNOBSERVABLE_RUNTIME_PATTERNS:
            matches = [
                match
                for match in pattern.finditer(behavior_text)
                if not _dependency_is_negated(behavior_text, match.start())
            ]
            if matches:
                matched_reasons.append(name)
        if matched_reasons:
            hits.append(
                {
                    "item": item,
                    "semantic": edit_semantic_payload(item, update_mode),
                    "behavior": behavior,
                    "matched_reasons": matched_reasons,
                }
            )
    return hits


def partition_runtime_observable_edits(
    items: list[dict],
    *,
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Return deployable edits and individually rejected hard dependencies."""
    hits = find_unobservable_runtime_edits(items, update_mode=update_mode)
    blocked_ids = {id(hit["item"]) for hit in hits}
    observable = [item for item in items if id(item) not in blocked_ids]
    return observable, hits


def filter_observable_runtime_edits(
    items: list[dict],
    *,
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Backward-compatible name for hard-only per-edit filtering."""
    return partition_runtime_observable_edits(items, update_mode=update_mode)


def filter_observable_runtime_patches(
    patches: list[dict],
    *,
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Remove hard-unobservable edits without discarding valid siblings."""
    filtered = []
    all_hits: list[dict[str, Any]] = []
    for patch in patches:
        result = dict(patch)
        items = get_payload_items(result, update_mode)
        observable, hits = partition_runtime_observable_edits(
            items,
            update_mode=update_mode,
        )
        set_payload_items(result, observable, update_mode)
        if hits and result.get("reasoning"):
            result["reasoning"] = (
                "Training-only analysis omitted; deployable edits retained."
            )
        filtered.append(result)
        all_hits.extend(hits)
    return filtered, all_hits


def format_runtime_observability_context(hits: list[dict] | None = None) -> str:
    """Tell the optimizer to express skills using inference-time evidence only."""
    rejected = ""
    if hits:
        rows = []
        for hit in hits:
            behavior = hit.get("behavior", {})
            reasons = ", ".join(hit.get("matched_reasons", []))
            rows.append(f"- {behavior.get('text', '')} (blocked: {reasons})")
        rejected = "\n\nBlocked behaviors from the prior attempt:\n" + "\n".join(rows)
    return (
        "## Hard runtime-observability constraint\n"
        "A deployable skill may use only the task instruction, provided context, "
        "tool observations, and other state visible at inference time. Never require "
        "gold or reference answers, ground-truth labels, answer keys, evaluator or "
        "grader feedback, evaluation scores, reward signals, or hidden tests. Terms "
        "such as canonical form, official name, core identifier, and proper casing are "
        "allowed when they do not require a hidden target. Rewrite hard dependencies "
        "as source-grounded procedures with observable checks."
        f"{rejected}"
    )


def normalize_candidate(
    items: list[dict], update_mode: str = "patch"
) -> list[str]:
    return sorted(normalize_edit(item, update_mode) for item in items)


def fingerprint_candidate(
    items: list[dict], update_mode: str = "patch"
) -> str:
    payload = json.dumps(
        normalize_candidate(items, update_mode),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def candidates_equivalent(
    left: list[dict],
    right: list[dict],
    *,
    update_mode: str = "patch",
    similarity_threshold: float = 0.88,
) -> bool:
    """Require an order-independent one-to-one fuzzy edit matching."""
    if len(left) != len(right):
        return False
    if not left:
        return True
    edges = {
        left_index: [
            right_index
            for right_index, right_item in enumerate(right)
            if edit_similarity(
                left_item,
                right_item,
                update_mode=update_mode,
            )
            >= similarity_threshold
        ]
        for left_index, left_item in enumerate(left)
    }
    if any(not matches for matches in edges.values()):
        return False
    order = sorted(edges, key=lambda index: len(edges[index]))

    def match(position: int, used: set[int]) -> bool:
        if position == len(order):
            return True
        for right_index in edges[order[position]]:
            if right_index not in used and match(position + 1, used | {right_index}):
                return True
        return False

    return match(0, set())


def find_quarantined_candidate(
    items: list[dict],
    records: list[dict],
    *,
    update_mode: str = "patch",
    similarity_threshold: float = 0.88,
) -> dict | None:
    for record in records:
        if record.get("scope") != CANDIDATE_SCOPE:
            continue
        if record.get("update_mode", "patch") != update_mode:
            continue
        recorded_items = record.get("items", [])
        if recorded_items and candidates_equivalent(
            items,
            recorded_items,
            update_mode=update_mode,
            similarity_threshold=similarity_threshold,
        ):
            return record
        if record.get("fingerprint") == fingerprint_candidate(items, update_mode):
            return record
    return None


def add_quarantined_candidate(
    records: list[dict],
    items: list[dict],
    *,
    update_mode: str = "patch",
    step: int | str = "",
    reason: str = "validation_reject",
    similarity_threshold: float = 0.88,
) -> dict | None:
    if not items or find_quarantined_candidate(
        items,
        records,
        update_mode=update_mode,
        similarity_threshold=similarity_threshold,
    ):
        return None
    record = {
        "scope": CANDIDATE_SCOPE,
        "fingerprint": fingerprint_candidate(items, update_mode),
        "normalized_items": normalize_candidate(items, update_mode),
        "items": items,
        "update_mode": update_mode,
        "step": step,
        "reason": reason,
    }
    records.append(record)
    return record


def load_edit_history(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid edit history JSON at {path}:{line_number}"
                ) from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def append_edit_history_event(path: str, event: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": EDIT_HISTORY_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_edit_history_event(
    items: list[dict],
    *,
    update_mode: str = "patch",
    **fields: Any,
) -> dict[str, Any]:
    return {
        **fields,
        "update_mode": update_mode,
        "candidate_fingerprint": fingerprint_candidate(items, update_mode),
        "edits": [
            {
                "edit_fingerprint": fingerprint_edit(item, update_mode),
                "semantic": edit_semantic_payload(item, update_mode),
                "behavior": edit_behavior_payload(item, update_mode),
                "raw": item,
            }
            for item in items
        ],
    }


def _event_items(event: dict) -> list[dict]:
    return [
        row.get("raw", {})
        for row in event.get("edits", [])
        if isinstance(row, dict) and isinstance(row.get("raw"), dict)
    ]


def _find_family(
    families: list[dict],
    item: dict,
    *,
    update_mode: str,
    similarity_threshold: float,
) -> dict | None:
    best = None
    best_score = -1.0
    for family in families:
        if family["update_mode"] != update_mode:
            continue
        score = max(
            behavior_similarity(item, variant, update_mode=update_mode)
            for variant in family["variant_items"]
        )
        if score >= similarity_threshold and score > best_score:
            best = family
            best_score = score
    return best


def _build_edit_families(
    records: list[dict],
    edit_history: list[dict],
    *,
    similarity_threshold: float,
) -> list[dict]:
    rejected: dict[str, tuple[str, list[dict]]] = {}
    accepted: dict[str, tuple[str, list[dict]]] = {}
    for record in records:
        if record.get("scope") == CANDIDATE_SCOPE:
            rejected[str(record.get("fingerprint", ""))] = (
                str(record.get("update_mode", "patch")),
                list(record.get("items", [])),
            )
    for event in edit_history:
        if not event.get("validation_ran"):
            continue
        fingerprint = str(event.get("candidate_fingerprint", ""))
        update_mode = str(event.get("update_mode", "patch"))
        items = _event_items(event)
        outcome = str(event.get("outcome", ""))
        if outcome in {"accept", "accept_new_best", "force_accept"}:
            accepted[fingerprint] = (update_mode, items)
        elif outcome == "reject":
            rejected[fingerprint] = (update_mode, items)

    families: list[dict] = []
    for outcome, candidates in (("rejected", rejected), ("accepted", accepted)):
        for candidate_fingerprint, (update_mode, items) in candidates.items():
            used_families: set[str] = set()
            for item in items:
                family = _find_family(
                    families,
                    item,
                    update_mode=update_mode,
                    similarity_threshold=similarity_threshold,
                )
                if family is None:
                    family = {
                        "family_id": hashlib.sha256(
                            json.dumps(
                                edit_behavior_payload(item, update_mode),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()[:20],
                        "update_mode": update_mode,
                        "representative_item": item,
                        "variant_items": [item],
                        "rejected_candidates": set(),
                        "accepted_candidates": set(),
                    }
                    families.append(family)
                elif all(
                    normalize_edit(item, update_mode)
                    != normalize_edit(variant, update_mode)
                    for variant in family["variant_items"]
                ) and len(family["variant_items"]) < 8:
                    family["variant_items"].append(item)
                if family["family_id"] in used_families:
                    continue
                family[f"{outcome}_candidates"].add(candidate_fingerprint)
                used_families.add(family["family_id"])
    return families


def promote_repeated_edits(
    records: list[dict],
    edit_history: list[dict] | None = None,
    *,
    threshold: int = 3,
    behavior_similarity_threshold: float = 0.72,
) -> list[dict]:
    """Block a fuzzy edit family after repeated rejects and zero acceptances."""
    if threshold <= 0:
        return []
    families = _build_edit_families(
        records,
        edit_history or [],
        similarity_threshold=behavior_similarity_threshold,
    )
    permanent = [
        record
        for record in records
        if record.get("scope") == PERMANENT_EDIT_SCOPE
    ]
    added = []
    for family in families:
        rejected_count = len(family["rejected_candidates"])
        accepted_count = len(family["accepted_candidates"])
        existing = next(
            (
                record
                for record in permanent
                if record.get("update_mode", "patch") == family["update_mode"]
                and any(
                    behavior_similarity(
                        family["representative_item"],
                        variant,
                        update_mode=family["update_mode"],
                    )
                    >= behavior_similarity_threshold
                    for variant in (
                        record.get("variant_items", [])
                        or [record.get("representative_item", {})]
                    )
                )
            ),
            None,
        )
        if existing is not None:
            existing["rejected_candidate_count"] = max(
                int(existing.get("rejected_candidate_count", 0)),
                rejected_count,
            )
            existing["accepted_candidate_count"] = max(
                int(existing.get("accepted_candidate_count", 0)),
                accepted_count,
            )
            existing_variants = list(existing.get("variant_items", []))
            for variant in family["variant_items"]:
                if all(
                    normalize_edit(variant, family["update_mode"])
                    != normalize_edit(prior, family["update_mode"])
                    for prior in existing_variants
                ) and len(existing_variants) < 8:
                    existing_variants.append(variant)
            existing["variant_items"] = existing_variants
            existing["behavior"] = edit_behavior_payload(
                family["representative_item"], family["update_mode"]
            )
            existing["behavior_similarity_threshold"] = (
                behavior_similarity_threshold
            )
            continue
        if rejected_count < threshold or accepted_count != 0:
            continue
        record = {
            "scope": PERMANENT_EDIT_SCOPE,
            "family_id": family["family_id"],
            "representative_item": family["representative_item"],
            "variant_items": family["variant_items"],
            "semantic": edit_semantic_payload(
                family["representative_item"], family["update_mode"]
            ),
            "behavior": edit_behavior_payload(
                family["representative_item"], family["update_mode"]
            ),
            "update_mode": family["update_mode"],
            "rejected_candidate_count": rejected_count,
            "accepted_candidate_count": accepted_count,
            "threshold": threshold,
            "behavior_similarity_threshold": behavior_similarity_threshold,
            "candidate_fingerprints": sorted(family["rejected_candidates"]),
            "reason": "repeated_across_rejected_candidates_without_acceptance",
        }
        records.append(record)
        permanent.append(record)
        added.append(record)
    return added


def find_permanently_quarantined_edits(
    items: list[dict],
    records: list[dict],
    *,
    update_mode: str = "patch",
    behavior_similarity_threshold: float = 0.72,
) -> list[dict]:
    blocked = []
    for item in items:
        for record in records:
            if record.get("scope") != PERMANENT_EDIT_SCOPE:
                continue
            if record.get("update_mode", "patch") != update_mode:
                continue
            variants = record.get("variant_items", []) or [
                record.get("representative_item", {})
            ]
            variants = [variant for variant in variants if isinstance(variant, dict)]
            if not variants:
                continue
            score = max(
                behavior_similarity(item, variant, update_mode=update_mode)
                for variant in variants
            )
            if score >= behavior_similarity_threshold:
                blocked.append({
                    "item": item,
                    "semantic": edit_semantic_payload(item, update_mode),
                    "behavior": edit_behavior_payload(item, update_mode),
                    "similarity": round(score, 6),
                    "permanent_record": record,
                })
                break
    return blocked


def format_permanent_edit_context(records: list[dict]) -> str:
    permanent = [
        record
        for record in records
        if record.get("scope") == PERMANENT_EDIT_SCOPE
    ]
    if not permanent:
        return ""
    rendered = "\n".join(
        f"- {_render_edit(record.get('representative_item', {}), record.get('update_mode', 'patch'))} "
        f"(rejected={record.get('rejected_candidate_count', '?')}, "
        f"accepted={record.get('accepted_candidate_count', '?')})"
        for record in permanent
    )
    return (
        "## Permanent bad-edit quarantine\n"
        "The edit families below repeatedly occurred in distinct candidates "
        "that failed validation and never appeared in an accepted candidate. "
        "Do not propose, merge, select, or paraphrase them.\n\n"
        f"{rendered}"
    )


def format_candidate_resample_context(
    records: list[dict],
    permanent_records: list[dict] | None = None,
) -> str:
    candidates = [
        record
        for record in records
        if record.get("scope") == CANDIDATE_SCOPE
    ]
    permanent_context = format_permanent_edit_context(permanent_records or [])
    if not candidates:
        return permanent_context
    blocks = []
    for index, record in enumerate(candidates, start=1):
        rendered = "\n".join(
            f"- {_render_edit(item, record.get('update_mode', 'patch'))}"
            for item in record.get("items", [])
        )
        blocks.append(f"Rejected candidate family {index}:\n{rendered}")
    candidate_context = (
        "## Candidate-level quarantine\n"
        "The following edit combinations already failed validation. Do not "
        "return an exact or near-equivalent complete combination. Individual "
        "edits may be reused when the overall selected set is different and "
        "the edit is not permanently quarantined.\n\n"
        + "\n\n".join(blocks)
    )
    return "\n\n".join(
        part for part in (permanent_context, candidate_context) if part
    )


def load_quarantine(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"quarantine file must contain a list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def save_quarantine(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
