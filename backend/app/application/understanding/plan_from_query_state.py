from __future__ import annotations

from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.unsupported import UNSUPPORTED_ANSWER
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
        )

    if state.intent == "facet_count" and state.facet_dimension:
        return _facet_count_plan(state.facet_dimension)
    if state.intent == "facet_list" and state.facet_dimension:
        return _facet_list_plan(state.facet_dimension)

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
            nodes=[PlanNode(id="e1", kind="tool", name="employee", params=params)],
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
    filters = state.filters_dict()
    if state.hire_year_gt:
        filters["hire_date_gt"] = f"{state.hire_year_gt}-01-01"
    question = f"employees with {state.skill} experience"
    resume_params: dict = {"question": question}
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
                **(
                    {}
                    if state.want_count or state.hire_year_gt
                    else {"columns": _NAME_COLUMNS}
                ),
            },
            input_bindings={"employee_ids": f"nodes.{id_source}"},
        )
    )
    return ExecutionPlan(
        nodes=nodes,
        response_strategy="template"
        if (state.want_count or prior_ids is not None)
        else "llm_format",
        active_cohort_node=id_source,
    )
