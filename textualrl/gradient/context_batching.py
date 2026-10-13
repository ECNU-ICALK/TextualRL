"""Retry context-rejected analyst calls without splitting sibling evidence."""

from __future__ import annotations

import json
from pathlib import Path


def is_context_length_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "http 400" in message
        and any(term in message for term in (
            "maximum context length", "context_length_exceeded",
            "exceeds the context window",
        ))
    )


def run_context_bounded_analyst(items, budget, route, call, cache_dir, tag):
    """Return leaf patches, retaining task blocks and the parent edit budget.

    Only an explicit API context rejection triggers splitting. Cached split
    plans avoid resending a known oversized parent when a later child fails.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)

    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def run(rows, allowance, name):
        patch_path = directory / f"{name}.json"
        plan_path = directory / f"{name}_context_split.json"
        blocks = {}
        for row in rows:
            task = str(row.get("rollout_group_id") or row.get("id") or "")
            blocks.setdefault(task, []).append(row)
        identity = {
            "route": route,
            "task_ids": list(blocks),
            "rollout_ids": [str(row.get("id") or "") for row in rows],
            "edit_budget": allowance,
        }
        if patch_path.exists():
            return [(name, json.loads(patch_path.read_text()), len(blocks), len(rows))]

        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if any(plan.get(key) != value for key, value in identity.items()):
                raise ValueError(f"Analyst split cache does not match evidence: {name}")
        else:
            try:
                patch = call(rows, allowance)
            except Exception as error:
                if not is_context_length_error(error):
                    raise
                # Homogeneous analysts still require two independent tasks.
                if (route not in {"stable_success", "stable_failure"}
                        or "" in blocks or len(blocks) < 4 or allowance < 2):
                    raise RuntimeError(
                        f"Cannot split context-rejected analyst {name} while preserving "
                        f"sibling groups, two-task support and edit budget "
                        f"(tasks={len(blocks)}, budget={allowance})"
                    ) from error
                midpoint = len(blocks) // 2
                task_ids = list(blocks)
                children = [task_ids[:midpoint], task_ids[midpoint:]]
                budgets = [allowance // 2, allowance - allowance // 2]
                plan = {
                    **identity,
                    "reason": "api_context_length_exceeded",
                    "api_error": str(error),
                    "child_task_ids": children,
                    "child_edit_budgets": budgets,
                }
                save(plan_path, plan)
                print(
                    f"      [analyst context split] {name}: {len(blocks)} tasks "
                    f"-> {[len(child) for child in children]}; "
                    f"edit budgets={budgets}; all trajectories retained",
                    flush=True,
                )
            else:
                if patch:
                    save(patch_path, patch)
                return [(name, patch, len(blocks), len(rows))]

        children = plan["child_task_ids"]
        budgets = plan["child_edit_budgets"]
        if (len(children) != 2 or len(budgets) != 2
                or any(len(child) < 2 for child in children)
                or [task for child in children for task in child] != list(blocks)
                or any(value < 1 for value in budgets) or sum(budgets) != allowance):
            raise ValueError(f"Invalid analyst split plan: {name}")
        leaves = []
        for index, (tasks, child_budget) in enumerate(zip(children, budgets)):
            child_rows = [row for task in tasks for row in blocks[task]]
            leaves.extend(run(child_rows, child_budget, f"{name}_part{index:02d}"))
        return leaves

    return run(items, budget, tag)
