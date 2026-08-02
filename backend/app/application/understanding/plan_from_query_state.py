from __future__ import annotations

from app.application.planning.location_plans import (
    LOCATION_FIELDS,
    location_cohort_nodes,
    location_cohort_plan,
    location_facet_plan,
    location_person_plan,
    profile_location_node,
    split_location_filters,
)
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.unsupported import UNSUPPORTED_ANSWER
from app.application.response.refusal import RefusalCode
from app.application.understanding.status_change import bind_status_subject_from_memory
from app.domain.query_state import QueryState
from app.domain.session import SessionMemory

_NAME_COLUMNS = ["id", "first_name", "last_name", "department", "position"]


def plan_from_query_state(
    state: QueryState,
    *,
    memory: SessionMemory | None = None,
    min_confidence: float = 0.7,
) -> ExecutionPlan | None:
    """Compile a constrained DAG from QueryState when confidence is high enough."""
    if state.confidence < min_confidence:
        return None
    if state.intent == "unsupported":
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=UNSUPPORTED_ANSWER,
            refusal_code=RefusalCode.OUT_OF_SCOPE.value,
        )
    if state.intent == "unknown":
        return None

    prior_ids = list(memory.last_employee_ids) if memory and memory.last_employee_ids else []
    if len(prior_ids) > 50:
        prior_ids = []
    focus = memory.last_focus if memory else None
    filters = state.filters_dict()

    # Short list follow-up without new filters → facet values or cohort names
    if state.intent == "list" and not filters and state.refers_to_prior:
        if focus and focus.kind == "facet" and focus.dimension:
            return _facet_list_plan(focus.dimension)
        if prior_ids:
            return _list_plan({"employee_ids": prior_ids})
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question=(
                "Which names should I list — countries, cities, departments, "
                "or employees from a previous search?"
            ),
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )

    if state.intent == "facet_count" and state.facet_dimension:
        return _facet_count_plan(state.facet_dimension)
    if state.intent == "facet_list" and state.facet_dimension:
        return _facet_list_plan(state.facet_dimension)

    if state.intent == "set_status":
        eid, display = bind_status_subject_from_memory(
            person_name=state.person_name,
            email=state.status_email,
            employee_id=state.status_employee_id,
            city=filters.get("city"),
            country=filters.get("country"),
            memory=memory,
        )
        if eid and not state.status_employee_id:
            state = state.model_copy(
                update={
                    "status_employee_id": eid,
                    "person_employee_ids": [eid],
                    "person_name": state.person_name or display,
                }
            )
        return _set_status_plan(state, filters=filters)

    if state.intent == "birthday":
        # Cohort birthday asks that refer to "them" / "from there" keep the prior set.
        if (
            state.refers_to_prior
            and prior_ids
            and (state.birthday_scope or "person") != "person"
            and not state.person_employee_ids
        ):
            state = state.model_copy(update={"person_employee_ids": list(prior_ids)})
        return _birthday_plan(state)

    if state.intent == "languages":
        if (
            state.refers_to_prior
            and prior_ids
            and not state.person_name
            and not state.person_employee_ids
        ):
            state = state.model_copy(update={"person_employee_ids": list(prior_ids)})
        return _languages_plan(state)

    if state.intent == "certifications":
        if (
            state.refers_to_prior
            and prior_ids
            and not state.person_name
            and not state.person_employee_ids
        ):
            state = state.model_copy(update={"person_employee_ids": list(prior_ids)})
        return _certifications_plan(state)

    if state.intent == "reports":
        from app.application.planning.heuristic_planner import (
            _reports_plan,
            _reports_skill_place_plan,
        )

        if state.skill and state.person_name:
            place, _ = split_location_filters(filters)
            return _reports_skill_place_plan(
                manager_name=state.person_name,
                skill=state.skill,
                city=place.get("city"),
                country=place.get("country"),
            )
        return _reports_plan(
            name=state.person_name,
            department=filters.get("department"),
        )

    if state.intent == "tenure_agg":
        from app.application.planning.heuristic_planner import _tenure_agg_plan

        return _tenure_agg_plan(department=filters.get("department"))

    if state.intent == "longest_tenured":
        from app.application.planning.heuristic_planner import _longest_tenured_plan

        return _longest_tenured_plan(department=filters.get("department"), limit=5)

    if state.intent == "manager":
        params: dict = {"action": "manager", "question": state.person_name or ""}
        if state.person_name:
            params["name"] = state.person_name
        if filters.get("department"):
            params["department"] = filters["department"]
        return ExecutionPlan(
            nodes=[PlanNode(id="e1", kind="tool", name="employee", params=params)],
            response_strategy="template",
            active_cohort_node="e1",
        )

    if state.intent == "profile" and state.person_name:
        params = {"action": "by_name", "name": state.person_name}
        if filters.get("department"):
            params["department"] = filters["department"]
        return ExecutionPlan(
            nodes=[
                PlanNode(id="e1", kind="tool", name="employee", params=params),
                # Location comes from the resume of the person this node resolved.
                profile_location_node("e1"),
            ],
            response_strategy="template",
            active_cohort_node="e1",
        )

    if state.intent == "skill_search" and state.skill:
        # Defer plain "who knows X" list UX to heuristic/LLM; handle scoped/count here.
        if not (state.want_count or state.refers_to_prior or state.filters_dict()):
            return None
        return _skill_plan(state, prior_ids=prior_ids if state.refers_to_prior else None)

    # Scope to prior cohort when anaphoric and IDs exist
    if state.refers_to_prior and prior_ids:
        filters = {**filters, "employee_ids": prior_ids}

    if state.intent == "count":
        # Need some signal: filters, prior ids, or explicit of-them count
        if not filters and not (state.refers_to_prior and prior_ids):
            # Org-wide headcount
            if not state.refers_to_prior:
                return _plain_count_plan({})
            return None
        return _filtered_count_plan(filters)

    if state.intent == "list":
        if not filters and not prior_ids:
            return None
        return _list_plan(filters)

    return None


def _location_split_plan(filters: dict, *, count_only: bool) -> ExecutionPlan | None:
    """Route a place filter through retrieval, keeping the rest in SQL.

    Returns None when no place is involved, so callers fall through to plain SQL.
    """
    place, sql_side = split_location_filters(filters)
    if not place:
        return None
    prior = sql_side.pop("employee_ids", None)
    return location_cohort_plan(
        city=place.get("city"),
        country=place.get("country"),
        count_only=count_only,
        extra_filters=sql_side,
        intersect_with=list(prior) if prior else None,
    )


def _certifications_plan(state: QueryState) -> ExecutionPlan:
    """Certifications live only in resume Certificates sections."""
    bound_ids = [str(x) for x in (state.person_employee_ids or []) if x]
    if state.person_name or (bound_ids and not state.certification):
        if not state.person_name and not bound_ids:
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question="Whose certifications would you like to know?",
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        name = state.person_name or "that employee"
        params: dict = {
            "question": name,
            "purpose": "certifications_person",
            "name": name,
        }
        if bound_ids:
            params["employee_ids"] = bound_ids
        return ExecutionPlan(
            nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
            response_strategy="template",
        )
    if not state.certification:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Which certification should I look for on resumes?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    params = {
        "question": state.certification,
        "purpose": "certifications_cohort",
        "certification": state.certification,
    }
    if bound_ids:
        params["employee_ids"] = bound_ids
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
        active_cohort_node="r1",
    )


def _languages_plan(state: QueryState) -> ExecutionPlan:
    """Languages live only in resume text."""
    bound_ids = [str(x) for x in (state.person_employee_ids or []) if x]
    if state.person_name or (bound_ids and not state.language):
        if not state.person_name and not bound_ids:
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question="Whose languages would you like to know?",
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        name = state.person_name or "that employee"
        params: dict = {
            "question": name,
            "purpose": "languages_person",
            "name": name,
        }
        if bound_ids:
            params["employee_ids"] = bound_ids
        return ExecutionPlan(
            nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
            response_strategy="template",
        )
    if not state.language:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Which language should I look for on resumes?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    params = {
        "question": f"speaks {state.language}",
        "purpose": "languages_cohort",
        "language": state.language,
    }
    if bound_ids:
        params["employee_ids"] = bound_ids
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
        active_cohort_node="r1",
    )


def _skills_person_plan(
    *,
    person_name: str | None,
    employee_ids: list[str] | None = None,
    skill: str | None = None,
) -> ExecutionPlan:
    """List (or check) skills from one person's resume Skills section."""
    bound_ids = [str(x) for x in (employee_ids or []) if x]
    if not person_name and not bound_ids:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Whose skills would you like to know?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    name = person_name or "that employee"
    params: dict = {
        "question": name,
        "purpose": "skills_person",
        "name": name,
    }
    if bound_ids:
        params["employee_ids"] = bound_ids
    if skill:
        params["skill"] = skill
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
    )


def _experience_person_plan(
    *,
    person_name: str | None,
    employee_ids: list[str] | None = None,
) -> ExecutionPlan:
    """Work history from one person's resume Experience section."""
    bound_ids = [str(x) for x in (employee_ids or []) if x]
    if not person_name and not bound_ids:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Whose work experience would you like to know?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    name = person_name or "that employee"
    params: dict = {
        "question": name,
        "purpose": "experience_person",
        "name": name,
    }
    if bound_ids:
        params["employee_ids"] = bound_ids
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
    )


def _projects_person_plan(
    *,
    person_name: str | None,
    employee_ids: list[str] | None = None,
) -> ExecutionPlan:
    """Projects from one person's resume Projects section."""
    bound_ids = [str(x) for x in (employee_ids or []) if x]
    if not person_name and not bound_ids:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Whose projects would you like to know?",
            refusal_code=RefusalCode.AMBIGUOUS.value,
        )
    name = person_name or "that employee"
    params: dict = {
        "question": name,
        "purpose": "projects_person",
        "name": name,
    }
    if bound_ids:
        params["employee_ids"] = bound_ids
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
    )


def _birthday_plan(state: QueryState) -> ExecutionPlan:
    """Birth dates exist only in resume text, so answer from retrieval alone."""
    scope = state.birthday_scope or "person"

    if scope == "person":
        bound_ids = [str(x) for x in (state.person_employee_ids or []) if x]
        if not state.person_name and not bound_ids:
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question="Whose birthday would you like to know?",
                refusal_code=RefusalCode.AMBIGUOUS.value,
            )
        name = state.person_name or "that employee"
        params: dict = {
            "question": name,
            "purpose": "birthday_person",
            "name": name,
            "wants_age": state.wants_age,
            "wants_wish": state.wants_wish,
        }
        if bound_ids:
            params["employee_ids"] = bound_ids
        return ExecutionPlan(
            nodes=[
                PlanNode(
                    id="r1",
                    kind="tool",
                    name="resume_search",
                    params=params,
                )
            ],
            response_strategy="template",
        )

    params: dict = {
        "question": "date of birth",
        "purpose": "birthday_cohort",
        "scope": scope,
    }
    if scope == "month" and state.birthday_month:
        params["month"] = state.birthday_month
    bound_ids = [str(x) for x in (state.person_employee_ids or []) if x]
    if bound_ids:
        params["employee_ids"] = bound_ids
        if scope == "closest":
            params["question"] = "closest birthday in prior set"
    return ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params=params)],
        response_strategy="template",
        active_cohort_node="r1" if bound_ids else None,
    )


def _set_status_plan(state: QueryState, *, filters: dict) -> ExecutionPlan:
    """Resolve the person/cohort via resumes, then write employees.status."""
    if state.status_value is None:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Should I set the employee's status to true or false?",
        )

    loc_filters = {
        k: filters[k] for k in ("city", "country") if filters.get(k) is not None
    }
    bound_ids = [str(x) for x in (state.person_employee_ids or []) if x]
    direct_id = state.status_employee_id or (bound_ids[0] if len(bound_ids) == 1 else None)
    has_person = bool(state.person_name or state.status_email or direct_id)
    if not has_person and not loc_filters:
        return ExecutionPlan(
            nodes=[],
            response_strategy="template",
            clarify_question="Which employee's status should I update?",
        )

    # Direct id/email — no RAG needed
    if direct_id or state.status_email:
        params: dict = {"action": "set_status", "status": state.status_value}
        if direct_id:
            params["employee_id"] = direct_id
        if state.status_email:
            params["email"] = state.status_email
        if state.person_name:
            params["name"] = state.person_name
        return ExecutionPlan(
            nodes=[PlanNode(id="e1", kind="tool", name="employee", params=params)],
            response_strategy="template",
            active_cohort_node="e1",
        )

    # Location cohort → resume retrieval → set_status. The place is read out of the
    # resumes, so this flow is RAG all the way to the write.
    if loc_filters and not state.person_name:
        nodes, id_source = location_cohort_nodes(
            city=loc_filters.get("city"), country=loc_filters.get("country")
        )
        nodes.append(
            PlanNode(
                id="e1",
                kind="tool",
                name="employee",
                depends_on=[id_source],
                params={
                    "action": "set_status",
                    "status": state.status_value,
                    "resolve_via": "location",
                },
                input_bindings={"employee_ids": f"nodes.{id_source}"},
            )
        )
        return ExecutionPlan(
            nodes=nodes,
            response_strategy="template",
            active_cohort_node="e1",
        )

    # Name → resume_search → extract ids → set_status
    name = state.person_name or ""
    emp_params = {
        "action": "set_status",
        "status": state.status_value,
        "name": name,
        "resolve_via": "rag",
    }
    if filters.get("department"):
        emp_params["department"] = filters["department"]
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="r1",
                kind="tool",
                name="resume_search",
                params={
                    "question": name,
                    "purpose": "status_resolve",
                },
            ),
            PlanNode(
                id="ids",
                kind="operator",
                name="extract_employee_ids",
                depends_on=["r1"],
                input_bindings={"data": "nodes.r1"},
            ),
            PlanNode(
                id="e1",
                kind="tool",
                name="employee",
                depends_on=["ids"],
                params=emp_params,
                input_bindings={"employee_ids": "nodes.ids"},
            ),
        ],
        response_strategy="template",
        active_cohort_node="e1",
    )


def _plain_count_plan(filters: dict) -> ExecutionPlan:
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={"mode": "constrained", "count_only": True, "filters": filters},
            )
        ],
        response_strategy="template",
    )


def _filtered_count_plan(filters: dict) -> ExecutionPlan:
    located = _location_split_plan(filters, count_only=True)
    if located is not None:
        return located
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="cohort",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": dict(filters),
                    "columns": ["id"],
                },
            ),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": True,
                    "filters": dict(filters),
                },
            ),
        ],
        response_strategy="template",
        active_cohort_node="cohort",
    )


def _list_plan(filters: dict) -> ExecutionPlan:
    located = _location_split_plan(filters, count_only=False)
    if located is not None:
        return located
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": dict(filters),
                    "columns": _NAME_COLUMNS,
                },
            )
        ],
        response_strategy="template",
        active_cohort_node="sql1",
    )


def _facet_count_plan(dimension: str) -> ExecutionPlan:
    if dimension in LOCATION_FIELDS:
        return location_facet_plan(dimension, count=True)
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="sql",
                params={"mode": "constrained", "distinct": True, "columns": [dimension]},
            ),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={"mode": "constrained", "count_distinct": dimension},
            ),
        ],
        response_strategy="template",
    )


def _facet_list_plan(dimension: str) -> ExecutionPlan:
    if dimension in LOCATION_FIELDS:
        return location_facet_plan(dimension)
    return ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="sql",
                params={"mode": "constrained", "distinct": True, "columns": [dimension]},
            )
        ],
        response_strategy="template",
    )


def _skill_plan(state: QueryState, *, prior_ids: list[str] | None) -> ExecutionPlan:
    # Skill RAG + optional place cohort: intersect ids so SQL never sees city/country.
    place, filters = split_location_filters(state.filters_dict())
    if state.hire_year_gt:
        filters["hire_date_gt"] = f"{state.hire_year_gt}-01-01"
    question = f"employees with {state.skill} experience"
    resume_params: dict = {
        "question": question,
        "purpose": "skill",
        "skill": state.skill,
    }
    if prior_ids:
        resume_params["employee_ids"] = list(prior_ids)
    nodes: list[PlanNode] = [
        PlanNode(id="r1", kind="tool", name="resume_search", params=resume_params),
        PlanNode(
            id="ids",
            kind="operator",
            name="extract_employee_ids",
            depends_on=["r1"],
            input_bindings={"data": "nodes.r1"},
        ),
    ]
    id_source = "ids"
    if prior_ids:
        nodes.append(
            PlanNode(
                id="ix",
                kind="operator",
                name="intersect_ids",
                depends_on=["ids"],
                input_bindings={"data": "nodes.ids"},
                params={"other": list(prior_ids)},
            )
        )
        id_source = "ix"
    if place:
        loc_nodes, loc_source = location_cohort_nodes(
            city=place.get("city"),
            country=place.get("country"),
        )
        nodes = loc_nodes + nodes
        nodes.append(
            PlanNode(
                id="skloc",
                kind="operator",
                name="intersect_ids",
                depends_on=[id_source, loc_source],
                input_bindings={
                    "data": f"nodes.{id_source}",
                    "other": f"nodes.{loc_source}",
                },
            )
        )
        id_source = "skloc"
    list_names = not (state.want_count or state.hire_year_gt)
    nodes.append(
        PlanNode(
            id="sql1",
            kind="tool",
            name="sql",
            depends_on=[id_source],
            params={
                "mode": "constrained",
                "count_only": bool(state.want_count or state.hire_year_gt),
                "filters": filters,
                **({"columns": _NAME_COLUMNS} if list_names else {}),
            },
            input_bindings={"employee_ids": f"nodes.{id_source}"},
        )
    )
    return ExecutionPlan(
        nodes=nodes,
        # Template + sql1 as active cohort so last_listed matches the shown order.
        response_strategy="template",
        active_cohort_node="sql1" if list_names else id_source,
    )
