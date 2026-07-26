import pytest

from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.plan_validator import PlanValidator
from app.application.routing.rule_router import RuleRouter
from app.domain.auth import AuthContext
from app.domain.enums import Role, RouterLabel
from app.domain.errors import PlanInvalidError
from app.domain.operators.base import OperatorRegistry
from app.domain.operators.count import Count
from app.domain.tools.registry import ToolRegistry
from app.tools.greeting.tool import GreetingTool


def test_rule_router_greeting() -> None:
    assert RuleRouter().route("hello there") == RouterLabel.GREETING
    assert RuleRouter().route("How many engineers?") is None
    assert RuleRouter().route("thanks, say their names") is None


def test_plan_validator_ok() -> None:
    tools = ToolRegistry()
    tools.register(GreetingTool())
    ops = OperatorRegistry()
    ops.register(Count())
    validator = PlanValidator(tools, ops)
    plan = ExecutionPlan(
        nodes=[PlanNode(id="a", kind="tool", name="greeting", params={"message": "hi"})]
    )
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    validator.validate(plan, auth)


def test_plan_validator_cycle() -> None:
    tools = ToolRegistry()
    tools.register(GreetingTool())
    ops = OperatorRegistry()
    validator = PlanValidator(tools, ops)
    plan = ExecutionPlan(
        nodes=[
            PlanNode(id="a", kind="tool", name="greeting", depends_on=["b"]),
            PlanNode(id="b", kind="tool", name="greeting", depends_on=["a"]),
        ]
    )
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    with pytest.raises(PlanInvalidError):
        validator.validate(plan, auth)
