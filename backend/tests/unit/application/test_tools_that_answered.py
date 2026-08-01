"""The `tool` field the API returns must name where an answer came from."""
from __future__ import annotations

from app.application.chat.chat_service import tools_that_answered
from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.tools.base import ToolResult

AUTH = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)


def _state(**results: ToolResult) -> GraphState:
    return GraphState(question="q", auth=AUTH, node_results=dict(results))


def _tool(node_id: str, name: str) -> PlanNode:
    return PlanNode(id=node_id, kind="tool", name=name)


def test_single_tool_is_named() -> None:
    plan = ExecutionPlan(nodes=[_tool("n1", "sql")])
    assert tools_that_answered(plan, _state(n1=ToolResult(data=[]))) == "sql"


def test_hybrid_plan_joins_its_tools_in_plan_order() -> None:
    plan = ExecutionPlan(nodes=[_tool("n1", "sql"), _tool("n2", "resume_search")])
    state = _state(n1=ToolResult(data=[]), n2=ToolResult(data={}))
    assert tools_that_answered(plan, state) == "sql+resume_search"


def test_operators_are_not_reported_as_tools() -> None:
    """An operator reshapes another tool's output; it fetches nothing."""
    plan = ExecutionPlan(
        nodes=[
            _tool("n1", "sql"),
            PlanNode(id="n2", kind="operator", name="count", depends_on=["n1"]),
        ]
    )
    state = _state(n1=ToolResult(data=[]))
    state.node_results["n2"] = 17
    assert tools_that_answered(plan, state) == "sql"


def test_a_repeated_tool_is_named_once() -> None:
    plan = ExecutionPlan(nodes=[_tool("n1", "sql"), _tool("n2", "sql")])
    state = _state(n1=ToolResult(data=[]), n2=ToolResult(data=[]))
    assert tools_that_answered(plan, state) == "sql"


def test_failed_tools_are_excluded_and_nothing_is_invented() -> None:
    plan = ExecutionPlan(nodes=[_tool("n1", "sql"), _tool("n2", "resume_search")])
    state = _state(
        n1=ToolResult(data=None, error="timeout", degraded=True),
        n2=ToolResult(data={}),
    )
    assert tools_that_answered(plan, state) == "resume_search"

    all_failed = _state(n1=ToolResult(data=None, error="timeout", degraded=True))
    assert tools_that_answered(ExecutionPlan(nodes=[_tool("n1", "sql")]), all_failed) is None


def test_clarify_only_turn_reports_no_tool() -> None:
    plan = ExecutionPlan(nodes=[], clarify_question="Which department?")
    assert tools_that_answered(plan, _state()) is None
