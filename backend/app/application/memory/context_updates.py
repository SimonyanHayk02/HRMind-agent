from __future__ import annotations

import re
from uuid import UUID

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.domain.session import ConstraintRef, EntityRef, LastFocus
from app.domain.tools.base import ToolResult

_DEPARTMENTS = [
    "Engineering",
    "People",
    "Sales",
    "Finance",
    "Product",
    "Operations",
]
_SKILLS = (
    "Python|Java|Go|React|Kubernetes|NLP|Machine Learning|SQL|"
    "AWS|Docker|Salesforce|Recruiting|Accounting|Product Strategy"
)
_SKILL_RE = re.compile(rf"\b({_SKILLS})\b", re.I)
_CITY_RE = re.compile(r"\b(Berlin|Dubai|London|Paris|New York)\b", re.I)

# Constraints that can be applied as SQL filters (not skills — those need RAG).
_SQL_CONSTRAINT_FIELDS = frozenset(
    {"department", "city", "country", "position", "education", "employment_status"}
)
_FACET_DIMS = ("country", "city", "department", "position", "education", "employment_status")


def infer_constraints_from_question(question: str) -> list[ConstraintRef]:
    out: list[ConstraintRef] = []
    lower = question.lower()
    for dept in _DEPARTMENTS:
        if dept.lower() in lower:
            out.append(ConstraintRef(field="department", op="eq", value=dept))
            break
    city = _CITY_RE.search(question)
    if city:
        out.append(ConstraintRef(field="city", op="eq", value=city.group(1)))
    skill = _SKILL_RE.search(question)
    if skill:
        out.append(ConstraintRef(field="skill", op="contains", value=skill.group(1)))
    return out


def filters_from_constraint_memory(constraints: list[ConstraintRef] | None) -> dict[str, str]:
    """Map remembered structured filters for the next SQL/RAG handoff."""
    if not constraints:
        return {}
    filters: dict[str, str] = {}
    for c in constraints:
        if c.field in _SQL_CONSTRAINT_FIELDS and c.op == "eq" and c.value:
            filters[c.field] = str(c.value)
    return filters


def extract_entities_from_state(state: GraphState) -> list[EntityRef]:
    entities: list[EntityRef] = []
    seen: set[UUID] = set()

    for result in state.node_results.values():
        data = result.data if isinstance(result, ToolResult) else result
        if not isinstance(data, dict):
            continue

        rows = data.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or not row.get("id"):
                    continue
                try:
                    eid = UUID(str(row["id"]))
                except Exception:
                    continue
                if eid in seen:
                    continue
                seen.add(eid)
                first = str(row.get("first_name") or "").strip()
                last = str(row.get("last_name") or "").strip()
                name = f"{first} {last}".strip() or str(eid)
                entities.append(EntityRef(employee_id=eid, display_name=name, confidence=0.9))

        hits = data.get("hits")
        if isinstance(hits, list):
            for hit in hits:
                if not isinstance(hit, dict) or not hit.get("employee_id"):
                    continue
                try:
                    eid = UUID(str(hit["employee_id"]))
                except Exception:
                    continue
                if eid in seen:
                    continue
                seen.add(eid)
                name = str(hit.get("employee_name") or "").strip() or str(eid)
                entities.append(EntityRef(employee_id=eid, display_name=name, confidence=0.8))

        # Employee tool profile / candidates / nested employee
        people: list[dict] = []
        if data.get("id") and (data.get("full_name") or data.get("first_name")):
            people.append(data)
        if isinstance(data.get("employee"), dict):
            people.append(data["employee"])
        if isinstance(data.get("candidates"), list):
            people.extend(x for x in data["candidates"] if isinstance(x, dict))
        if isinstance(data.get("employees"), list):
            people.extend(x for x in data["employees"] if isinstance(x, dict))
        for person in people:
            if not person.get("id"):
                continue
            try:
                eid = UUID(str(person["id"]))
            except Exception:
                continue
            if eid in seen:
                continue
            seen.add(eid)
            name = str(person.get("full_name") or "").strip()
            if not name:
                first = str(person.get("first_name") or "").strip()
                last = str(person.get("last_name") or "").strip()
                name = f"{first} {last}".strip() or str(eid)
            entities.append(EntityRef(employee_id=eid, display_name=name, confidence=0.9))

    return entities


def extract_last_focus(
    state: GraphState,
    plan: ExecutionPlan,
    *,
    employee_ids: list[str] | None = None,
) -> LastFocus | None:
    """Derive discourse focus from the plan/results for short follow-ups."""
    # Prefer explicit distinct / count_distinct facet plans
    for node in plan.nodes:
        if node.name != "sql":
            continue
        params = node.params or {}
        dim = params.get("count_distinct")
        if not dim and params.get("distinct"):
            cols = params.get("columns") or []
            if cols:
                dim = cols[0]
        if not dim or dim not in _FACET_DIMS:
            continue
        values = _facet_values_from_state(state, str(dim), prefer_node=node.id)
        return LastFocus(kind="facet", dimension=str(dim), values=values)

    # Distinct-looking rows without employee id (nl2sql fallback)
    for node in plan.nodes:
        result = state.node_results.get(node.id)
        data = result.data if isinstance(result, ToolResult) else result
        if not isinstance(data, dict):
            continue
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        sample = rows[0] if isinstance(rows[0], dict) else None
        if not sample or sample.get("id") or sample.get("first_name"):
            continue
        for dim in _FACET_DIMS:
            if dim in sample:
                values = _unique_row_values(rows, dim)
                if values:
                    return LastFocus(kind="facet", dimension=dim, values=values)

    if employee_ids:
        return LastFocus(kind="cohort", dimension="employees", values=[])
    return None


def _facet_values_from_state(
    state: GraphState, dimension: str, *, prefer_node: str | None = None
) -> list[str]:
    order = []
    if prefer_node:
        order.append(prefer_node)
    order.extend(k for k in state.node_results if k != prefer_node)
    for key in order:
        result = state.node_results.get(key)
        data = result.data if isinstance(result, ToolResult) else result
        if not isinstance(data, dict):
            continue
        rows = data.get("rows")
        if isinstance(rows, list) and rows:
            vals = _unique_row_values(rows, dimension)
            if vals:
                return vals
    return []


def _unique_row_values(rows: list, dimension: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or dimension not in row:
            continue
        val = str(row.get(dimension) or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def plan_mentions_result_set(plan: ExecutionPlan) -> bool:
    for node in plan.nodes:
        filters = (node.params or {}).get("filters") or {}
        if filters.get("employee_ids"):
            return True
        if node.name == "intersect_ids":
            return True
        if node.name == "resume_search" and (node.params or {}).get("employee_ids"):
            return True
    return False
