"""PlanValidator must reject any sql node that touches a resume-sourced field."""
from __future__ import annotations

import pytest

from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.plan_validator import PlanValidator
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import PlanInvalidError
from app.domain.operators.base import OperatorRegistry
from app.domain.operators.extract_ids import ExtractEmployeeIds
from app.domain.tools.base import ToolMeta, ToolResult
from app.domain.tools.registry import ToolRegistry


class _StubTool:
    def __init__(self, name: str) -> None:
        self.meta = ToolMeta(name=name, description=name, permissions=list(Role))

    async def run(self, params, *, auth) -> ToolResult:
        return ToolResult(data={})


def _validator() -> PlanValidator:
    tools = ToolRegistry()
    tools.register(_StubTool("sql"))
    tools.register(_StubTool("resume_search"))
    ops = OperatorRegistry()
    ops.register(ExtractEmployeeIds())
    return PlanValidator(tools, ops)


AUTH = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)


@pytest.mark.parametrize(
    "params",
    [
        {"mode": "constrained", "filters": {"city": "Berlin"}},
        {"mode": "constrained", "filters": {"country": "USA"}},
        {"mode": "constrained", "count_distinct": "city"},
        {"mode": "constrained", "distinct": True, "columns": ["country"]},
        {"mode": "constrained", "columns": ["id", "city"]},
    ],
)
def test_sql_node_with_resume_field_is_rejected(params: dict) -> None:
    plan = ExecutionPlan(
        nodes=[PlanNode(id="sql1", kind="tool", name="sql", params=params)]
    )
    with pytest.raises(PlanInvalidError, match="Resume-sourced"):
        _validator().validate(plan, AUTH)


def test_location_retrieval_plan_is_accepted() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="loc",
                kind="tool",
                name="resume_search",
                params={"purpose": "location_cohort", "city": "Berlin"},
            ),
            PlanNode(
                id="ids",
                kind="operator",
                name="extract_employee_ids",
                depends_on=["loc"],
                input_bindings={"data": "nodes.loc"},
            ),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                depends_on=["ids"],
                params={"mode": "constrained", "count_only": True, "filters": {}},
                input_bindings={"employee_ids": "nodes.ids"},
            ),
        ]
    )
    _validator().validate(plan, AUTH)
