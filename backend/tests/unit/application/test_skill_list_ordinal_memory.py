from __future__ import annotations

from uuid import UUID

from app.application.execution.graph_state import GraphState
from app.application.memory.context_updates import extract_listed_employees
from app.application.planning.heuristic_planner import try_heuristic_plan
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.domain.tools.base import ToolResult
from app.application.understanding.list_referents import resolve_list_referent
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import EntityRef, SessionMemory


def test_plain_skill_plan_materializes_sql_names() -> None:
    plan = try_heuristic_plan("Show employees with AWS experience.")
    assert plan is not None
    assert plan.response_strategy == "template"
    assert plan.active_cohort_node == "sql1"
    assert any(n.name == "sql" for n in plan.nodes)
    assert any(n.name == "resume_search" for n in plan.nodes)


def test_extract_listed_from_resume_hits_when_no_sql_rows() -> None:
    e1 = "00000000-0000-0000-0000-000000000001"
    e2 = "00000000-0000-0000-0000-000000000002"
    plan = ExecutionPlan(
        nodes=[PlanNode(id="r1", kind="tool", name="resume_search", params={})],
        response_strategy="template",
        active_cohort_node="r1",
    )
    state = GraphState(
        question="who has AWS experience?",
        auth=AuthContext(
            user_id="u", tenant_id="t", role=Role.RECRUITER, department_id="Engineering"
        ),
        node_results={
            "r1": ToolResult(
                data={
                    "hits": [
                        {"employee_id": e1, "employee_name": "Jack Smith"},
                        {"employee_id": e2, "employee_name": "Grace Silva"},
                    ],
                    "employee_ids": [e1, e2],
                },
                confidence=0.9,
            )
        },
    )
    listed = extract_listed_employees(state, plan)
    assert [str(x.employee_id) for x in listed] == [e1, e2]
    assert listed[0].display_name == "Jack Smith"


def test_ordinal_first_one_binds_last_listed() -> None:
    e1 = UUID("00000000-0000-0000-0000-000000000001")
    e2 = UUID("00000000-0000-0000-0000-000000000002")
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_listed=[
            EntityRef(employee_id=e1, display_name="Jack Smith"),
            EntityRef(employee_id=e2, display_name="Grace Silva"),
        ],
    )
    ref = resolve_list_referent("give the birth date of first one", memory)
    assert ref.kind == "resolved"
    assert ref.employee_id == str(e1)
    assert ref.display_name == "Jack Smith"
