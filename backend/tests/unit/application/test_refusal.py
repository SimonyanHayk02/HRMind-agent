"""Unit matrix for runtime refusal resolver + salary ACL + soft-chat whitelist."""
from __future__ import annotations

import pytest

from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.salary_acl import is_salary_question, salary_acl_plan
from app.application.planning.unsupported import UNSUPPORTED_ANSWER, is_unsupported_topic
from app.application.response.refusal import (
    RefusalCode,
    empty_unscoped_fallback,
    has_format_evidence,
    is_soft_unauthorized_error,
    resolve_refusal,
    unauthorized_plan,
)
from app.application.response.response_formatter import ResponseFormatter
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.tools.base import ToolResult


class _NoLLM:
    async def complete(self, **_kwargs):  # noqa: ANN003
        return "LLM should not be called"


def _auth(role: Role = Role.RECRUITER, department: str | None = "Engineering") -> AuthContext:
    return AuthContext(
        user_id="u",
        tenant_id="t",
        role=role,
        department_id=department,
    )


def _state(
    node_results: dict | None = None,
    *,
    degraded: bool = False,
    errors: list[str] | None = None,
    auth: AuthContext | None = None,
) -> GraphState:
    return GraphState(
        question="q",
        auth=auth or _auth(),
        node_results=node_results or {},
        degraded=degraded,
        errors=errors or [],
    )


def test_oos_hint_and_topic() -> None:
    plan = ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question=UNSUPPORTED_ANSWER,
        refusal_code=RefusalCode.OUT_OF_SCOPE.value,
    )
    out = resolve_refusal(plan, _state(), "how much PTO do we get?")
    assert out is not None
    assert out.code == RefusalCode.OUT_OF_SCOPE
    assert out.ui_clarify is None
    assert out.preserve_cohort


def test_travel_preferences_are_oos_not_profile() -> None:
    q = "give me information about her travelling preferences"
    assert is_unsupported_topic(q)
    # Even if a pronoun path wrongly ran employee lookup, refuse — don't dump profile.
    plan = ExecutionPlan(
        nodes=[PlanNode(id="e1", kind="tool", name="employee", params={})],
        response_strategy="template",
    )
    state = _state(
        {
            "e1": ToolResult(
                data={
                    "id": "1",
                    "full_name": "Alice Nguyen",
                    "department": "Sales",
                    "position": "Account Executive",
                    "education": "Bootcamp",
                },
                confidence=0.9,
            )
        }
    )
    out = resolve_refusal(plan, state, q, auth=_auth())
    assert out is not None
    assert out.code == RefusalCode.OUT_OF_SCOPE
    assert "don't have that information" in out.message.lower()


def test_unauthorized_hint() -> None:
    plan = unauthorized_plan()
    out = resolve_refusal(plan, _state(), "what's Alice's salary?", auth=_auth(Role.EMPLOYEE))
    assert out is not None
    assert out.code == RefusalCode.UNAUTHORIZED
    assert out.ui_clarify is None


def test_ambiguous_sets_ui_clarify() -> None:
    plan = ExecutionPlan(
        nodes=[],
        response_strategy="template",
        clarify_question="Which employee do you mean?",
        refusal_code=RefusalCode.AMBIGUOUS.value,
    )
    out = resolve_refusal(plan, _state(), "tell me about her")
    assert out is not None
    assert out.code == RefusalCode.AMBIGUOUS
    assert out.ui_clarify == "Which employee do you mean?"


def test_empty_cohort_refine_zero() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "count_only": True,
                    "filters": {"employee_ids": ["a", "b"]},
                },
            )
        ],
        response_strategy="template",
        active_cohort_node="sql1",
    )
    state = _state({"sql1": ToolResult(data={"count": 0}, confidence=1.0)})
    out = resolve_refusal(plan, state, "how many of them know Rust?")
    assert out is not None
    assert out.code == RefusalCode.EMPTY_COHORT
    assert out.message == "The answer is 0."
    assert not out.preserve_cohort


def test_org_wide_zero_is_ok_not_empty_cohort() -> None:
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
    state = _state({"sql1": ToolResult(data={"count": 0}, confidence=1.0)})
    out = resolve_refusal(plan, state, "how many employees in Atlantis?")
    assert out is None  # formatter emits typed 0 under ok


def test_missing_dob() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(id="r1", kind="tool", name="resume_search", params={}),
        ],
        response_strategy="template",
    )
    state = _state(
        {
            "r1": ToolResult(
                data={"answer": "I couldn't find a date of birth for Alice."},
                confidence=0.5,
            )
        }
    )
    out = resolve_refusal(plan, state, "when was Alice born?")
    assert out is not None
    assert out.code == RefusalCode.MISSING_DATA


def test_tool_error_degraded() -> None:
    plan = ExecutionPlan(
        nodes=[PlanNode(id="sql1", kind="tool", name="sql", params={})],
        response_strategy="template",
    )
    state = _state(
        {"sql1": ToolResult(data=None, error="connection refused", confidence=0.0)},
        degraded=True,
        errors=["connection refused"],
    )
    out = resolve_refusal(plan, state, "how many engineers?")
    assert out is not None
    assert out.code == RefusalCode.TOOL_ERROR


def test_salary_stripped_for_manager() -> None:
    plan = ExecutionPlan(
        nodes=[PlanNode(id="e1", kind="tool", name="employee", params={})],
        response_strategy="template",
    )
    state = _state(
        {
            "e1": ToolResult(
                data={
                    "id": "1",
                    "full_name": "Bob Peer",
                    "department": "Sales",
                },
                confidence=0.9,
            )
        },
        auth=_auth(Role.MANAGER, department="Engineering"),
    )
    out = resolve_refusal(
        plan,
        state,
        "what is Bob Peer's salary?",
        auth=_auth(Role.MANAGER, department="Engineering"),
    )
    assert out is not None
    assert out.code == RefusalCode.UNAUTHORIZED


def test_soft_unauthorized_whitelist() -> None:
    assert is_soft_unauthorized_error("Tool not permitted: sql")
    assert is_soft_unauthorized_error("Column not allowed for role: salary")
    assert not is_soft_unauthorized_error("Plan contains a cycle")
    assert not is_soft_unauthorized_error("Unknown tool foobar")


def test_salary_acl_employee_vs_recruiter() -> None:
    assert is_salary_question("what's Alice Nguyen's salary?")
    assert not is_salary_question("how much PTO do we get?")
    assert salary_acl_plan(
        "what's Alice's salary?", auth=_auth(Role.EMPLOYEE)
    ) is not None
    assert salary_acl_plan(
        "what's Alice's salary?", auth=_auth(Role.RECRUITER)
    ) is None
    assert salary_acl_plan(
        "what's Alice's salary?", auth=_auth(Role.MANAGER)
    ) is None


@pytest.mark.asyncio
async def test_formatter_refuses_without_llm() -> None:
    plan = unauthorized_plan()
    answer, conf, sources, clarify = await ResponseFormatter(_NoLLM()).format(
        "salary?",
        plan,
        _state(),
        auth=_auth(Role.EMPLOYEE),
    )
    assert "don't have access" in answer.lower() or "access" in answer.lower()
    assert clarify is None
    assert sources == []
    assert conf <= 0.5


@pytest.mark.asyncio
async def test_formatter_llm_gate_no_evidence() -> None:
    plan = ExecutionPlan(
        nodes=[PlanNode(id="sql1", kind="tool", name="sql", params={})],
        response_strategy="llm_format",
    )
    state = _state({"sql1": ToolResult(data=None, confidence=0.0)})
    assert not has_format_evidence(plan, state)
    answer, _c, _s, clarify = await ResponseFormatter(_NoLLM()).format(
        "list engineers in Mars", plan, state
    )
    assert answer == empty_unscoped_fallback()
    assert clarify is None
    assert "don't have that information" not in answer.lower() or "matching" in answer.lower()
