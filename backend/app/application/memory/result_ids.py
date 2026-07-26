from __future__ import annotations

from typing import Any
from uuid import UUID

from app.application.execution.graph_state import GraphState
from app.domain.tools.base import ToolResult


def extract_employee_ids_from_state(state: GraphState) -> list[str]:
    """Collect employee IDs produced by tools/operators in this turn."""
    ids: list[str] = []
    for result in state.node_results.values():
        ids.extend(_from_value(result))
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _from_value(value: Any) -> list[str]:
    if isinstance(value, ToolResult):
        return _from_value(value.data)
    if isinstance(value, list):
        out: list[str] = []
        if value and all(isinstance(x, str) for x in value):
            # operator extract_employee_ids returns list[str]
            for x in value:
                if _looks_uuid(x):
                    out.append(str(x))
            return out
        for item in value:
            out.extend(_from_value(item))
        return out
    if isinstance(value, dict):
        out = []
        if "employee_ids" in value and isinstance(value["employee_ids"], list):
            out.extend(str(x) for x in value["employee_ids"] if x)
        if "employee_id" in value and value["employee_id"]:
            out.append(str(value["employee_id"]))
        if "id" in value and value["id"] and _looks_uuid(str(value["id"])):
            # SQL employee rows
            out.append(str(value["id"]))
        if "hits" in value and isinstance(value["hits"], list):
            for hit in value["hits"]:
                out.extend(_from_value(hit))
        if "rows" in value and isinstance(value["rows"], list):
            for row in value["rows"]:
                out.extend(_from_value(row))
        return out
    return []


def _looks_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except Exception:
        return False
