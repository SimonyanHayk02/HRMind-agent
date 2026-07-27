import pytest

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.response.response_formatter import ResponseFormatter
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.tools.base import ToolResult


class _NoLLM:
    async def complete(self, **_kwargs):  # noqa: ANN003
        return "I do not have that information."


def _state(node_results: dict) -> GraphState:
    return GraphState(
        question="q",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results=node_results,
    )


@pytest.mark.asyncio
async def test_empty_count_is_zero_not_llm_hedge() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={"mode": "constrained", "count_only": True, "filters": {}},
            )
        ],
        response_strategy="template",
    )
    state = _state(
        {"sql1": ToolResult(data={"count": 0}, confidence=1.0, sources=[])}
    )
    answer, conf, _sources, clarify = await ResponseFormatter(_NoLLM()).format(
        "which of them know Kubernetes?", plan, state
    )
    assert answer == "The answer is 0."
    assert conf >= 0.5
    assert clarify is None


@pytest.mark.asyncio
async def test_empty_employees_not_raw_json() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="e1", kind="tool", name="employee", params={"action": "by_name"}
            )
        ],
        response_strategy="template",
    )
    state = _state(
        {"e1": ToolResult(data={"employees": []}, confidence=0.5, sources=[])}
    )
    answer, _conf, _sources, _clarify = await ResponseFormatter(_NoLLM()).format(
        "where does she live?", plan, state
    )
    assert not answer.strip().startswith("{")
    assert "employee" in answer.lower() or "which" in answer.lower()


@pytest.mark.asyncio
async def test_empty_prior_rows_grounded() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": False,
                    "filters": {"employee_ids": ["x"]},
                },
            )
        ],
        response_strategy="template",
    )
    state = _state(
        {"sql1": ToolResult(data={"rows": []}, confidence=1.0, sources=[])}
    )
    answer, _conf, _sources, _clarify = await ResponseFormatter(_NoLLM()).format(
        "which of them are in Dubai?", plan, state
    )
    assert "none of the previous set" in answer.lower()
