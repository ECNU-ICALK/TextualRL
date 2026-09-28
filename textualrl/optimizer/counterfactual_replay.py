"""Bounded counterfactual replay for rejected skill edits.

The replay buffer is built only from candidates that completed validation and
were rejected. Edits are grouped into fuzzy semantic families so superficial
rewrites do not reset their rejection frequency. Sampling is novelty-first:
untested, low-frequency families receive probe budget before repeatedly
rejected or already evaluated families.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any

from textualrl.optimizer.quarantine import (
    behavior_similarity,
    edit_behavior_payload,
)


COUNTERFACTUAL_REPLAY_VERSION = "counterfactual_replay_v1"


def normalize_replay_item(item: dict) -> dict:
    """Normalize legacy edit payloads before applying them to a skill."""
    if not isinstance(item, dict):
        return {}
    normalized = dict(item)
    if normalized.get("target") is None:
        normalized["target"] = ""
    return normalized


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
) -> dict | None:
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
    return best


def build_rejected_edit_families(
    edit_history: list[dict],
    *,
    current_epoch: int,
    update_mode: str = "patch",
    similarity_threshold: float = 0.82,
) -> list[dict]:
    """Group edits from validated rejected candidates into semantic families."""
    families: list[dict] = []
    for event in edit_history:
        if not event.get("validation_ran"):
            continue
        if str(event.get("outcome", "")) != "reject":
            continue
        event_mode = str(event.get("update_mode", "patch"))
        if event_mode != update_mode:
            continue
        items = _event_items(event)
        if not items:
            continue
        candidate_key = str(event.get("candidate_fingerprint", "")) or (
            f"step:{event.get('step', 0)}"
        )
        event_epoch = int(event.get("epoch", 0) or 0)
        used_family_ids: set[str] = set()
        for item in items:
            family = _match_family(
                item,
                families,
                update_mode=update_mode,
                similarity_threshold=similarity_threshold,
            )
            if family is None:
                family = {
                    "family_id": _family_id(item, update_mode),
                    "update_mode": update_mode,
                    "behavior": edit_behavior_payload(item, update_mode),
                    "representative_item": item,
                    "variant_items": [item],
                    "candidate_keys": [],
                    "rejection_steps": [],
                    "current_epoch_steps": [],
                    "current_epoch_multi_edit_steps": [],
                    "analyst_support": 0,
                }
                families.append(family)
            family_id = str(family["family_id"])
            if family_id in used_family_ids:
                continue
            used_family_ids.add(family_id)
            if candidate_key not in family["candidate_keys"]:
                family["candidate_keys"].append(candidate_key)
            step = int(event.get("step", 0) or 0)
            if step and step not in family["rejection_steps"]:
                family["rejection_steps"].append(step)
            if (
                event_epoch == int(current_epoch)
                and step
                and step not in family["current_epoch_steps"]
            ):
                family["current_epoch_steps"].append(step)
            if (
                event_epoch == int(current_epoch)
                and len(items) > 1
                and step
                and step not in family["current_epoch_multi_edit_steps"]
            ):
                family["current_epoch_multi_edit_steps"].append(step)
            family["analyst_support"] = max(
                int(family.get("analyst_support", 0) or 0),
                int(item.get("support_count", 0) or 0),
            )
            if len(family["variant_items"]) < 8 and all(
                edit_behavior_payload(item, update_mode)
                != edit_behavior_payload(variant, update_mode)
                for variant in family["variant_items"]
            ):
                family["variant_items"].append(item)

    for family in families:
        family["rejection_frequency"] = len(family.pop("candidate_keys"))
        family["eligible_in_current_epoch"] = bool(
            family.get("current_epoch_multi_edit_steps")
        )
    return families


def load_counterfactual_memory(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {
            "version": COUNTERFACTUAL_REPLAY_VERSION,
            "records": [],
        }
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {
            "version": COUNTERFACTUAL_REPLAY_VERSION,
            "records": [],
        }
    if isinstance(payload, list):
        payload = {"records": payload}
    if not isinstance(payload, dict):
        payload = {}
    records = payload.get("records", [])
    return {
        "version": COUNTERFACTUAL_REPLAY_VERSION,
        "records": records if isinstance(records, list) else [],
    }


def save_counterfactual_memory(path: str, memory: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": COUNTERFACTUAL_REPLAY_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": list(memory.get("records", [])),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def append_counterfactual_record(
    path: str,
    memory: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    saved = dict(record)
    saved.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
    records = memory.setdefault("records", [])
    record_id = str(saved.get("record_id", ""))
    if record_id:
        for index, existing in enumerate(records):
            if str(existing.get("record_id", "")) == record_id:
                records[index] = saved
                save_counterfactual_memory(path, memory)
                return saved
    records.append(saved)
    save_counterfactual_memory(path, memory)
    return saved


def prioritize_replay_families(
    families: list[dict],
    memory: dict[str, Any],
    *,
    base_skill_hash: str,
    limit: int,
) -> list[dict]:
    """Select current-epoch families with low rejection frequency first."""
    records = list(memory.get("records", []))
    eligible = []
    for family in families:
        if not family.get("eligible_in_current_epoch"):
            continue
        family_id = str(family.get("family_id", ""))
        family_records = [
            row for row in records if str(row.get("family_id", "")) == family_id
        ]
        state_records = [
            row
            for row in family_records
            if str(row.get("base_skill_hash", "")) == str(base_skill_hash)
        ]
        candidate = dict(family)
        candidate["counterfactual_eval_count"] = len(family_records)
        candidate["current_state_eval_count"] = len(state_records)
        candidate["priority_key"] = [
            int(bool(state_records)),
            int(bool(family_records)),
            int(candidate.get("rejection_frequency", 0) or 0),
            len(family_records),
            -int(candidate.get("analyst_support", 0) or 0),
            family_id,
        ]
        eligible.append(candidate)

    eligible.sort(key=lambda row: tuple(row["priority_key"]))
    return eligible[: max(0, int(limit))]


def summarize_replay_reports(report_paths: list[str]) -> dict[str, Any]:
    reports = []
    for path in report_paths:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            reports.append(payload)
    probe_count = sum(
        len(report.get("probe_results", [])) for report in reports
    )
    positive = sum(
        int(row.get("probe_net_gain", 0) or 0) > 0
        for report in reports
        for row in report.get("probe_results", [])
    )
    negative = sum(
        int(row.get("probe_net_gain", 0) or 0) < 0
        for report in reports
        for row in report.get("probe_results", [])
    )
    return {
        "epoch_report_count": len(reports),
        "counterfactual_probe_count": probe_count,
        "positive_probe_count": positive,
        "neutral_probe_count": probe_count - positive - negative,
        "negative_probe_count": negative,
        "full_verification_count": sum(
            int(bool(report.get("full_verification"))) for report in reports
        ),
        "recovered_edit_count": sum(
            int(report.get("action") in {"accept", "accept_new_best"})
            for report in reports
        ),
    }
