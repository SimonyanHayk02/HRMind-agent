from __future__ import annotations

import re
from uuid import UUID

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.domain.places import CITY_ALT, COUNTRY_ALT, canon_city, canon_country
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
_CITY_RE = re.compile(rf"\b({CITY_ALT})\b", re.IGNORECASE)
_COUNTRY_RE = re.compile(rf"\b({COUNTRY_ALT})\b", re.IGNORECASE)

# Constraints that can be applied as SQL filters. City and country are absent:
# they live in resume text, so a remembered place becomes a retrieval scope
# instead (see place_from_constraint_memory), and skills need RAG too.
_SQL_CONSTRAINT_FIELDS = frozenset(
    {"department", "position", "education", "employment_status"}
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
        out.append(
            ConstraintRef(field="city", op="eq", value=canon_city(city.group(1)) or city.group(1))
        )
    else:
        country = _COUNTRY_RE.search(question)
        if country:
            out.append(
                ConstraintRef(
                    field="country",
                    op="eq",
                    value=canon_country(country.group(1)) or country.group(1),
                )
            )
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


def place_from_constraint_memory(
    constraints: list[ConstraintRef] | None,
) -> tuple[str | None, str | None]:
    """The remembered place, for the planner to turn into a retrieval node.

    Kept separate from `filters_from_constraint_memory` so a city remembered from
    an earlier turn can never quietly become a SQL filter.
    """
    city: str | None = None
    country: str | None = None
    for c in constraints or []:
        if c.op != "eq" or not c.value:
            continue
        if c.field == "city":
            city = canon_city(str(c.value)) or str(c.value)
        elif c.field == "country":
            country = canon_country(str(c.value)) or str(c.value)
    return city, country


def _row_as_entity(row: dict) -> EntityRef | None:
    if not row.get("id"):
        return None
    # Facet rows have a dimension value but no person name — never list-anchor them.
    has_name = bool(
        row.get("first_name")
        or row.get("last_name")
        or row.get("full_name")
        or row.get("employee_name")
    )
    if not has_name:
        return None
    try:
        eid = UUID(str(row["id"]))
    except Exception:
        return None
    name = str(row.get("full_name") or row.get("employee_name") or "").strip()
    if not name:
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        name = f"{first} {last}".strip() or str(eid)
    return EntityRef(employee_id=eid, display_name=name, confidence=0.9)


def _entities_from_resume_payload(data: dict) -> list[EntityRef]:
    """Ordered people from resume_search hits/facts when SQL did not list names."""
    out: list[EntityRef] = []
    seen: set[UUID] = set()

    facts = data.get("facts")
    if isinstance(facts, list):
        for fact in facts:
            if not isinstance(fact, dict) or not fact.get("employee_id"):
                continue
            try:
                eid = UUID(str(fact["employee_id"]))
            except Exception:
                continue
            if eid in seen:
                continue
            name = str(fact.get("name") or fact.get("employee_name") or "").strip()
            if not name:
                continue
            seen.add(eid)
            out.append(EntityRef(employee_id=eid, display_name=name, confidence=0.85))
        if out:
            return out

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
            name = str(
                hit.get("employee_name")
                or (hit.get("metadata") or {}).get("employee_name")
                or ""
            ).strip()
            if not name:
                continue
            seen.add(eid)
            out.append(EntityRef(employee_id=eid, display_name=name, confidence=0.8))
    return out


def extract_listed_employees(state: GraphState, plan: ExecutionPlan) -> list[EntityRef]:
    """Ordered employees from the answer-driving name list, if any.

    Used for ordinals ("the first person"). Facet value lists and count-only
    payloads are ignored so they cannot poison display order.
    """
    # Prefer the active cohort / last sql or employee node that returned name rows.
    prefer_ids: list[str] = []
    if plan.active_cohort_node:
        prefer_ids.append(plan.active_cohort_node)
    for node in plan.nodes:
        if node.name in {"sql", "employee"} and node.id not in prefer_ids:
            prefer_ids.append(node.id)

    def _from_data(data: object) -> list[EntityRef]:
        if not isinstance(data, dict):
            return []
        # Count-only — no names shown.
        if data.get("count") is not None and not data.get("rows") and not data.get("employees"):
            return []
        out: list[EntityRef] = []
        seen: set[UUID] = set()
        for key in ("rows", "employees"):
            rows = data.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ent = _row_as_entity(row)
                if ent is None or ent.employee_id in seen:
                    continue
                seen.add(ent.employee_id)
                out.append(ent)
        if out:
            return out
        # Attribute / skill RAG answers that never passed through SQL.
        return _entities_from_resume_payload(data)

    for nid in prefer_ids:
        result = state.node_results.get(nid)
        if result is None:
            continue
        data = result.data if isinstance(result, ToolResult) else result
        listed = _from_data(data)
        if listed:
            return listed

    # Fallback: first payload in node order that looks like a name list.
    for node in plan.nodes:
        result = state.node_results.get(node.id)
        if result is None:
            continue
        data = result.data if isinstance(result, ToolResult) else result
        listed = _from_data(data)
        if listed:
            return listed
    return []


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
                if not isinstance(row, dict):
                    continue
                ent = _row_as_entity(row)
                if ent is None or ent.employee_id in seen:
                    continue
                seen.add(ent.employee_id)
                entities.append(ent)

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
            ent = _row_as_entity(person)
            if ent is None or ent.employee_id in seen:
                continue
            seen.add(ent.employee_id)
            entities.append(ent)

    return entities


def extract_last_focus(
    state: GraphState,
    plan: ExecutionPlan,
    *,
    employee_ids: list[str] | None = None,
) -> LastFocus | None:
    """Derive discourse focus from the plan/results for short follow-ups."""
    # Resume-sourced facets (city/country) are aggregated by retrieval.
    for node in plan.nodes:
        if node.name != "resume_search":
            continue
        params = node.params or {}
        dim = params.get("facet")
        if not dim or dim not in _FACET_DIMS:
            continue
        values = _facet_values_from_state(state, str(dim), prefer_node=node.id)
        return LastFocus(kind="facet", dimension=str(dim), values=values)

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
