from __future__ import annotations

from typing import Any
from uuid import UUID

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.domain.tools.base import ToolResult


def extract_cohort_ids(state: GraphState, plan: ExecutionPlan | None = None) -> list[str]:
    """Pick the active employee cohort from the plan DAG (not a union of all nodes).

    Preference order:
    1. plan.active_cohort_node result (including empty — empty intersect must win)
    2. last intersect_ids operator output
    3. last extract_employee_ids operator output
    4. last SQL rows with id (non-count)
    5. last resume_search employee_ids
    """
    if plan and plan.active_cohort_node:
        # If the preferred node ran, honor its output even when empty. Falling
        # through would pick pre-intersect location ids and silently widen "them".
        if plan.active_cohort_node in state.node_results:
            return _dedupe(_from_value(state.node_results.get(plan.active_cohort_node)))

    # Walk nodes in plan order; keep last matching producer
    last: list[str] = []
    nodes = list(plan.nodes) if plan else []
    if not nodes:
        # Fallback: scan all results (legacy)
        for result in state.node_results.values():
            ids = _from_value(result)
            if ids:
                last = ids
        return _dedupe(last)

    last_intersect: list[str] = []
    last_extract: list[str] = []
    last_sql_rows: list[str] = []
    last_resume: list[str] = []

    for node in nodes:
        result = state.node_results.get(node.id)
        if result is None:
            continue
        if node.kind == "operator" and node.name == "intersect_ids":
            ids = _from_value(result)
            if ids is not None:
                last_intersect = ids
        elif node.kind == "operator" and node.name == "extract_employee_ids":
            ids = _from_value(result)
            if ids is not None:
                last_extract = ids
        elif node.name == "sql":
            data = result.data if isinstance(result, ToolResult) else result
            if isinstance(data, dict) and isinstance(data.get("rows"), list) and data["rows"]:
                row_ids = _from_value(data)
                if row_ids:
                    last_sql_rows = row_ids
        elif node.name == "resume_search":
            ids = _from_value(result)
            if ids:
                last_resume = ids

    for candidate in (last_intersect, last_extract, last_sql_rows, last_resume):
        if candidate:
            return _dedupe(candidate)
    return []


def extract_employee_ids_from_state(state: GraphState) -> list[str]:
    """Legacy helper — prefer extract_cohort_ids(state, plan) in chat flow."""
    return extract_cohort_ids(state, plan=None)


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, ToolResult):
        return _from_value(value.data)
    if isinstance(value, list):
        if value and all(isinstance(x, str) for x in value):
            return [str(x) for x in value if _looks_uuid(x)]
        out: list[str] = []
        for item in value:
            out.extend(_from_value(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        if "employee_ids" in value and isinstance(value["employee_ids"], list):
            out.extend(str(x) for x in value["employee_ids"] if x)
        if "employee_id" in value and value["employee_id"]:
            out.append(str(value["employee_id"]))
        if "id" in value and value["id"] and _looks_uuid(str(value["id"])):
            out.append(str(value["id"]))
        if "hits" in value and isinstance(value["hits"], list):
            for hit in value["hits"]:
                out.extend(_from_value(hit))
        if "rows" in value and isinstance(value["rows"], list):
            for row in value["rows"]:
                out.extend(_from_value(row))
        if "employees" in value and isinstance(value["employees"], list):
            for row in value["employees"]:
                out.extend(_from_value(row))
        if "employee" in value and isinstance(value["employee"], dict):
            out.extend(_from_value(value["employee"]))
        if "candidates" in value and isinstance(value["candidates"], list):
            for row in value["candidates"]:
                out.extend(_from_value(row))
        if "managers" in value and isinstance(value["managers"], list):
            for row in value["managers"]:
                out.extend(_from_value(row))
        return out
    return []


def _looks_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except Exception:
        return False
