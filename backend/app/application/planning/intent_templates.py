"""Expand high-level intents into validated multi-tool ExecutionPlans."""
from __future__ import annotations

import re
from typing import Any

from app.application.planning.heuristic_planner import (
    _JOINED_RE,
    _hire_date_filters,
    _hire_date_list_plan,
    _longest_tenured_plan,
    _tenure_agg_plan,
)
from app.application.planning.location_plans import location_cohort_plan, location_person_plan
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.response.refusal import RefusalCode
from app.application.understanding.plan_from_query_state import (
    _birthday_plan,
    _certifications_plan,
    _facet_count_plan,
    _facet_list_plan,
    _filtered_count_plan,
    _languages_plan,
    _plain_count_plan,
    _set_status_plan,
)
from app.domain.places import CITIES, COUNTRIES, canon_city, canon_country
from app.domain.query_state import QueryState
from app.domain.session import SessionMemory


_NAME_COLUMNS = ["id", "first_name", "last_name", "department", "position"]

_DEPARTMENT_CANON = {
    "engineering": "Engineering",
    "eng": "Engineering",
    "engeneering": "Engineering",
    "people": "People",
    "hr": "People",
    "sales": "Sales",
    "finance": "Finance",
    "product": "Product",
    "operations": "Operations",
    "ops": "Operations",
}


def _canon_department(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return _DEPARTMENT_CANON.get(raw.lower(), raw)


def re_status_active(q_l: str) -> bool:
    return bool(
        re.search(
            r"\b(?:status\s+active|active\s+status|employment\s+status\s+active|"
            r"have\s+status\s+active|with\s+status\s+active)\b",
            q_l,
        )
    )


def re_status_inactive(q_l: str) -> bool:
    return bool(
        re.search(
            r"\b(?:status\s+inactive|inactive\s+status|employment\s+status\s+inactive|"
            r"have\s+status\s+inactive|with\s+status\s+inactive)\b",
            q_l,
        )
    )


def expand_intent(
    intent: str,
    *,
    question: str,
    slots: dict[str, Any],
    memory: SessionMemory | None,
) -> ExecutionPlan | None:
    """Compile a known intent + slots into a safe DAG. Returns None if unknown."""
    intent = (intent or "").strip().lower()
    prior_ids = list(memory.last_employee_ids) if memory and memory.last_employee_ids else []
    if len(prior_ids) > 50:
        prior_ids = []

    department = _canon_department(slots.get("department"))
    raw_city = str(slots.get("city") or "").strip()
    raw_country = str(slots.get("country") or "").strip()
    city = canon_city(raw_city) if raw_city else None
    country = canon_country(raw_country) if raw_country else None
    skill = slots.get("skill")
    name = slots.get("name") or slots.get("person_name")
    count_only = bool(slots.get("count_only") or slots.get("want_count"))
    refers = bool(slots.get("refers_to_prior"))
    employee_ids = [str(x) for x in (slots.get("employee_ids") or []) if x]
    if not employee_ids and refers and prior_ids:
        employee_ids = list(prior_ids)

    q_l = (question or "").lower()
    any_of_them = bool(re.search(r"\bany(?:one)?\s+of\s+them\b", q_l))

    # Recover common skills when the model chose count/list but mentioned a skill.
    if not skill:
        for token, label in (
            ("python", "Python"),
            ("kubernetes", "Kubernetes"),
            ("kubernetees", "Kubernetes"),
            ("docker", "Docker"),
            ("react", "React"),
            ("java", "Java"),
            ("aws", "AWS"),
        ):
            if token in q_l:
                skill = label
                break

    # Recover department / place from the question when the model omitted slots.
    if not department and intent in {
        "count",
        "headcount",
        "list",
        "roster",
        "skill_search",
        "skill_count",
    }:
        for alias, canon in _DEPARTMENT_CANON.items():
            if alias in q_l:
                department = canon
                break
    if not city:
        for c in CITIES:
            if c.lower() in q_l:
                city = c
                break
    if not country:
        for c in COUNTRIES:
            if c.lower() in q_l:
                country = c
                break

    # Fresh department/location scopes must not inherit a prior cohort unless
    # the user explicitly referred to it ("of them").
    explicit_prior = any(
        phrase in q_l
        for phrase in (
            "of them",
            "among them",
            "among those",
            "of those",
            "from them",
            "from those",
        )
    ) or (
        refers and any(w in q_l.split() for w in ("them", "those"))
    )
    if (department or city or country) and not explicit_prior:
        employee_ids = []
        refers = False

    # Unknown place slot must clarify — never search with a dropped city/country.
    if (raw_city and not city) or (raw_country and not country):
        unknown = raw_city if (raw_city and not city) else raw_country
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                f'I don\'t recognize "{unknown}" as a known city or country. '
                "Try one of the places we track (for example Berlin, Dubai, or USA)."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

    # LLM often maps facet asks to plain count/list — rewrite before expanding.
    if intent in {"count", "headcount", "list", "roster"}:
        from app.application.planning.heuristic_planner import (
            _FACET_COUNT_RE,
            _FACET_LIST_RE,
            _detect_facet_dimension,
        )

        dim = slots.get("facet_dimension") or slots.get("facet") or _detect_facet_dimension(
            question
        )
        if dim and _FACET_COUNT_RE.search(question or ""):
            intent = "facet_count"
            slots = {**slots, "facet_dimension": dim}
        elif dim and _FACET_LIST_RE.search(question or ""):
            intent = "facet_list"
            slots = {**slots, "facet_dimension": dim}

    if intent in {"unsupported", "out_of_scope"}:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "I don't have that information in the HR data I can access "
                "(PTO, benefits, payroll, equity, visa, and performance reviews need "
                "dedicated HRIS sources that are not connected yet)."
            ),
            refusal_code=RefusalCode.OUT_OF_SCOPE.value,
        )

    if intent == "clarify":
        q = slots.get("clarify_question") or (
            "Could you clarify who or what you mean?"
        )
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=str(q),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

    if intent in {"count", "headcount"}:
        # Skill (+ optional place) counts must not collapse to location-only SQL.
        if skill:
            skill_ids = employee_ids if explicit_prior else []
            return _skill_plan(
                question=f"who knows {skill}",
                count_only=True,
                intersect_with=skill_ids or None,
                city=city,
                country=country,
                skill=str(skill),
            )
        filters: dict[str, Any] = {}
        if department:
            filters["department"] = department
        if employee_ids:
            filters["employee_ids"] = employee_ids
        emp_status = slots.get("employment_status")
        if not emp_status and re_status_active(q_l):
            emp_status = "active"
        elif not emp_status and re_status_inactive(q_l):
            emp_status = "inactive"
        if emp_status:
            filters["employment_status"] = str(emp_status)
        # Agent flag employees.status (bool) — only when clearly true/false, not "active".
        agent_flag = slots.get("status")
        if isinstance(agent_flag, bool) and "employment_status" not in filters:
            filters["status"] = agent_flag
        if city or country:
            return location_cohort_plan(
                city=city,
                country=country,
                count_only=True,
                intersect_with=employee_ids or None,
                extra_filters={
                    k: v
                    for k, v in (
                        ("department", department),
                        ("employment_status", filters.get("employment_status")),
                        ("status", filters.get("status")),
                    )
                    if v is not None
                }
                or None,
            )
        if filters:
            return _filtered_count_plan(filters)
        return _plain_count_plan({})

    if intent in {"list", "roster"}:
        focus = memory.last_focus if memory else None
        if (
            focus is not None
            and getattr(focus, "kind", None) == "facet"
            and getattr(focus, "dimension", None)
            and not department
            and not city
            and not country
            and not employee_ids
        ):
            return _facet_list_plan(str(focus.dimension))
        filters = {}
        if department:
            filters["department"] = department
        if employee_ids:
            filters["employee_ids"] = employee_ids
        if not filters and not (city or country):
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=(
                    "Which names should I list — countries, cities, departments, "
                    "or employees from a previous search?"
                ),
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        if city or country:
            return location_cohort_plan(
                city=city,
                country=country,
                count_only=count_only or any_of_them,
                intersect_with=employee_ids or None,
                extra_filters={"department": department} if department else None,
            )
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="sql1",
                    kind="tool",
                    name="sql",
                    params={
                        "mode": "constrained",
                        "count_only": False,
                        "filters": filters,
                        "columns": _NAME_COLUMNS,
                    },
                )
            ],
            response_strategy="template",
            active_cohort_node="sql1",
        )

    if intent in {"skill_search", "skill_count"}:
        # Fresh skill asks ("who knows X?") must not inherit a prior cohort.
        skill_ids = employee_ids if explicit_prior else []
        # Never embed the count phrasing — it collapses retrieval recall.
        rag_question = f"who knows {skill}" if skill else question
        return _skill_plan(
            question=rag_question,
            count_only=count_only or intent == "skill_count",
            intersect_with=skill_ids or None,
            city=city,
            country=country,
            skill=str(skill) if skill else None,
        )

    if intent == "birthday_person":
        return _birthday_plan(
            QueryState(
                intent="birthday",
                birthday_scope="person",
                person_name=name,
                person_employee_ids=employee_ids,
                wants_age=bool(slots.get("wants_age")),
                wants_wish=bool(slots.get("wants_wish")),
                confidence=0.9,
            )
        )

    if intent in {"birthday_cohort", "birthday_today", "birthday_upcoming", "birthday_month"}:
        scope = {
            "birthday_today": "today",
            "birthday_upcoming": "upcoming",
            "birthday_month": "month",
            "birthday_cohort": slots.get("scope") or "upcoming",
        }.get(intent, "upcoming")
        return _birthday_plan(
            QueryState(
                intent="birthday",
                birthday_scope=scope,
                birthday_month=slots.get("month"),
                person_employee_ids=employee_ids if refers else [],
                refers_to_prior=refers,
                wants_wish=bool(slots.get("wants_wish")),
                confidence=0.9,
            )
        )

    if intent == "location_person":
        return location_person_plan(name or "that employee", employee_ids=employee_ids or None)

    if intent == "location_cohort":
        return location_cohort_plan(
            city=city,
            country=country,
            count_only=count_only or any_of_them,
            intersect_with=employee_ids or None,
        )

    if intent == "tenure_agg":
        # Don't let the model route join windows into tenure aggregates.
        if _JOINED_RE.search(question or ""):
            hire = _hire_date_filters(question or "")
            if hire:
                if count_only or any(
                    w in q_l for w in ("how many", "count", "number of")
                ):
                    return _filtered_count_plan(hire)
                return _hire_date_list_plan(hire)
        return _tenure_agg_plan(department=department)

    if intent == "longest_tenured":
        if _JOINED_RE.search(question or ""):
            hire = _hire_date_filters(question or "")
            if hire:
                return _hire_date_list_plan(hire)
        return _longest_tenured_plan(department=department, limit=5)

    if intent == "facet_count":
        dim = str(slots.get("facet_dimension") or slots.get("facet") or "country")
        if dim in {"city", "country"}:
            from app.application.planning.location_plans import location_facet_plan

            return location_facet_plan(dim, count=True)
        return _facet_count_plan(dim)

    if intent == "facet_list":
        dim = str(slots.get("facet_dimension") or slots.get("facet") or "country")
        if dim in {"city", "country"}:
            from app.application.planning.location_plans import location_facet_plan

            return location_facet_plan(dim, count=False)
        return _facet_list_plan(dim)

    if intent == "languages":
        return _languages_plan(
            QueryState(
                intent="languages",
                language=slots.get("language"),
                person_name=name,
                person_employee_ids=employee_ids,
                refers_to_prior=refers,
                confidence=0.9,
            )
        )

    if intent == "certifications":
        return _certifications_plan(
            QueryState(
                intent="certifications",
                certification=slots.get("certification"),
                person_name=name,
                person_employee_ids=employee_ids,
                refers_to_prior=refers,
                confidence=0.9,
            )
        )

    if intent == "profile":
        if not name and not employee_ids:
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question="Which employee did you mean?",
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        if employee_ids and len(employee_ids) == 1:
            return ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="e1",
                        kind="tool",
                        name="employee",
                        params={"action": "by_id", "employee_id": employee_ids[0]},
                    )
                ],
                response_strategy="template",
                active_cohort_node="e1",
            )
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="e1",
                    kind="tool",
                    name="employee",
                    params={"action": "by_name", "name": name},
                )
            ],
            response_strategy="template",
            active_cohort_node="e1",
        )

    if intent == "manager":
        params: dict[str, Any] = {"action": "manager", "question": question}
        if name:
            params["name"] = name
        if employee_ids and len(employee_ids) == 1:
            params["employee_id"] = employee_ids[0]
        return ExecutionPlan(
            nodes=[PlanNode(id="e1", kind="tool", name="employee", params=params)],
            response_strategy="template",
            active_cohort_node="e1",
        )

    if intent in {"status_list", "status_resolve", "status_read"}:
        # Read path: list status columns for a prior cohort (or named person).
        # status_resolve is a resume purpose for *writes* — never use it for reads.
        ids = employee_ids or (list(prior_ids) if explicit_prior or refers else [])
        if ids:
            from app.application.planning.heuristic_planner import _sql_over_ids

            return _sql_over_ids(
                ids,
                count_only=False,
                columns=[
                    "id",
                    "first_name",
                    "last_name",
                    "department",
                    "position",
                    "status",
                    "employment_status",
                ],
            )
        if name:
            # Fall through to profile so the formatter can surface status.
            return ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="e1",
                        kind="tool",
                        name="employee",
                        params={"action": "by_name", "name": name},
                    )
                ],
                response_strategy="template",
                active_cohort_node="e1",
            )
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "Which employees' statuses should I show? "
                "Name a person or refer to a previous list."
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

    if intent == "set_status":
        status_val = slots.get("status")
        if status_val is None:
            status_val = slots.get("status_value")
        # Explicit person name must not inherit prior-focus employee_ids.
        status_ids = [] if name else employee_ids
        return _set_status_plan(
            QueryState(
                intent="set_status",
                person_name=name,
                status_value=status_val if isinstance(status_val, bool) else None,
                status_email=slots.get("email"),
                status_employee_id=status_ids[0] if len(status_ids) == 1 else None,
                person_employee_ids=status_ids,
                confidence=0.9,
            ),
            filters={
                k: v
                for k, v in (("city", city), ("country", country))
                if v
            },
        )

    if intent == "greeting":
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="greet",
                    kind="tool",
                    name="greeting",
                    params={"message": question},
                )
            ],
            response_strategy="template",
        )

    return None


def _skill_plan(
    *,
    question: str,
    count_only: bool,
    intersect_with: list[str] | None,
    city: str | None = None,
    country: str | None = None,
    skill: str | None = None,
) -> ExecutionPlan:
    """Skill RAG (+ optional prior / place intersects). Reuse the battle-tested DAG."""
    from app.application.planning.heuristic_planner import _resume_search_plan

    plan = _resume_search_plan(
        question,
        count_only=count_only,
        intersect_with=intersect_with,
        city=city,
        country=country,
        skill=skill,
    )
    # Mark the skill retrieval node so purpose-aware tooling stays consistent.
    for node in plan.nodes:
        if node.name == "resume_search" and node.id == "r1":
            params = dict(node.params or {})
            params.setdefault("purpose", "skill")
            if skill:
                params.setdefault("skill", skill)
            node.params = params
            break
    return plan


def expand_selected_tools(
    selected: list[dict[str, Any]],
    *,
    question: str,
) -> ExecutionPlan | None:
    """Build a simple DAG from explicit tool selections (1–3 tools)."""
    if not selected:
        return None
    q_l = (question or "").lower()
    explicit_prior = any(
        phrase in q_l
        for phrase in ("of them", "among them", "among those", "of those", "from them")
    )
    nodes: list[PlanNode] = []
    prev: str | None = None
    for i, item in enumerate(selected):
        tool = str(item.get("tool") or item.get("name") or "").strip()
        if not tool:
            continue
        params = dict(item.get("params") or {})
        if tool == "sql":
            filters = dict(params.get("filters") or {})
            if "department" in filters:
                filters["department"] = _canon_department(filters.get("department"))
            # Drop inherited cohort ids on fresh department/location asks.
            if (
                not explicit_prior
                and (filters.get("department") or filters.get("city") or filters.get("country"))
            ):
                filters.pop("employee_ids", None)
            if filters:
                params["filters"] = filters
        if tool == "resume_search" and "question" not in params:
            params["question"] = question
        if tool == "greeting" and "message" not in params:
            params["message"] = question
        if tool == "clarify":
            # Prefer empty-node clarify plan for formatter compatibility.
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=str(
                    params.get("question") or "Could you clarify who or what you mean?"
                ),
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        nid = f"t{i+1}"
        node = PlanNode(
            id=nid,
            kind="tool",
            name=tool,
            params=params,
            depends_on=[prev] if prev else [],
        )
        nodes.append(node)
        prev = nid
    if not nodes:
        return None
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="template",
        active_cohort_node=nodes[-1].id,
    )
