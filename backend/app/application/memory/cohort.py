from __future__ import annotations

import re

from app.application.planning.plan_schema import ExecutionPlan


def should_update_last_employee_ids(
    plan: ExecutionPlan,
    question: str,
    ids: list[str],
) -> bool:
    """Avoid treating org-wide headcounts / nl2sql dumps as the active cohort."""
    if not ids:
        return False

    has_resume = any(n.name == "resume_search" for n in plan.nodes)
    has_scoped_ids = False
    has_unscoped_nl2sql = False
    has_unscoped_count = False

    has_attribute_filter = False
    for node in plan.nodes:
        if node.name != "sql":
            continue
        params = node.params or {}
        filters = params.get("filters") or {}
        bindings = node.input_bindings or {}
        scoped = bool(filters.get("employee_ids") or bindings.get("employee_ids"))
        attr = any(
            filters.get(k)
            for k in ("department", "city", "country", "position", "hire_date_gt")
        )
        if scoped:
            has_scoped_ids = True
        if attr:
            has_attribute_filter = True
        if params.get("mode") == "nl2sql" and not scoped and not attr:
            has_unscoped_nl2sql = True
        if params.get("count_only") and not scoped and not attr:
            has_unscoped_count = True

    # Org-wide count / unrestricted nl2sql must not redefine "them"
    if has_unscoped_count and not has_resume and not has_scoped_ids:
        return False
    if has_unscoped_nl2sql and not has_resume:
        return False

    q = question.lower()
    if (
        re.search(r"\bhow many employees\b", q)
        and not has_resume
        and not has_scoped_ids
        and not has_attribute_filter
    ):
        return False

    # Hard cap: a cohort larger than this is almost certainly a full table dump
    if len(ids) > 50 and not has_resume:
        return False

    return True
