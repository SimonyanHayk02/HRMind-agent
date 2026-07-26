from __future__ import annotations

import re
from uuid import UUID

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan
from app.domain.session import ConstraintRef, EntityRef
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
_SQL_CONSTRAINT_FIELDS = frozenset({"department", "city", "country", "position"})


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

    return entities


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
