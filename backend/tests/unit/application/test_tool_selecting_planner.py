"""Tool-selecting planner + rich ToolMeta + intent expansion."""
from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from app.adapters.llm.fake_llm import FakeLLM
from app.application.planning.intent_templates import expand_intent
from app.application.planning.tool_schemas import (
    EMPLOYEE_DESCRIPTION,
    RESUME_SEARCH_DESCRIPTION,
    SQL_DESCRIPTION,
)
from app.application.planning.tool_selecting_planner import (
    ToolSelectingPlanner,
    ToolSelection,
)
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import SessionMemory
from app.domain.tools.registry import ToolRegistry
from app.tools.clarify.tool import ClarifyTool
from app.tools.employee.tool import EmployeeTool
from app.tools.greeting.tool import GreetingTool


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(GreetingTool())
    reg.register(ClarifyTool())
    reg.register(EmployeeTool(lambda _auth: _DummyUow()))
    return reg


class _DummyUow:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    employees = None


def test_tool_metas_include_schemas() -> None:
    g = GreetingTool()
    c = ClarifyTool()
    e = EmployeeTool(lambda _a: _DummyUow())
    assert g.meta.input_schema
    assert c.meta.input_schema
    assert e.meta.input_schema
    assert "DO NOT" in e.meta.description or "set_status" in e.meta.description.lower()
    assert "hello" in g.meta.description.lower() or "chitchat" in g.meta.description.lower()


def test_tool_schema_docs_cover_anti_patterns() -> None:
    assert "DO NOT" in SQL_DESCRIPTION
    assert "birthday" in SQL_DESCRIPTION.lower() or "date of birth" in SQL_DESCRIPTION.lower()
    assert "DO NOT" in RESUME_SEARCH_DESCRIPTION or "headcount" in RESUME_SEARCH_DESCRIPTION.lower()
    assert "set_status" in EMPLOYEE_DESCRIPTION


def test_discover_includes_registered_tools() -> None:
    reg = _registry()
    names = {d["name"] for d in reg.discover()}
    assert names == {"greeting", "clarify", "employee"}
    for meta in reg.discover():
        assert meta.get("description")
        assert meta.get("input_schema") is not None


def test_expand_intent_count_department() -> None:
    plan = expand_intent(
        "count",
        question="how many in Engineering?",
        slots={"department": "Engineering", "count_only": True},
        memory=None,
    )
    assert plan is not None
    assert any(n.name == "sql" for n in plan.nodes)
    assert any((n.params or {}).get("count_only") for n in plan.nodes if n.name == "sql")


def test_expand_intent_skill_plus_city_intersects() -> None:
    plan = expand_intent(
        "skill_count",
        question="how much of them knows python and lives in Dubai?",
        slots={"skill": "Python", "count_only": True, "refers_to_prior": True},
        memory=None,
    )
    assert plan is not None
    names = [n.name for n in plan.nodes]
    assert names.count("resume_search") >= 2
    assert "intersect_ids" in names
    sql = next(n for n in plan.nodes if n.name == "sql")
    assert "employee_ids" in (sql.input_bindings or {})


def test_expand_intent_fresh_department_drops_prior_ids() -> None:
    prior = [str(uuid4()), str(uuid4())]
    mem = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=prior,
    )
    plan = expand_intent(
        "count",
        question="count peeps in product pls",
        slots={
            "department": "product",
            "count_only": True,
            "employee_ids": prior,
            "refers_to_prior": False,
        },
        memory=mem,
    )
    assert plan is not None
    for node in plan.nodes:
        if node.name == "sql":
            filters = (node.params or {}).get("filters") or {}
            assert filters.get("department") == "Product"
            assert "employee_ids" not in filters


def test_expand_intent_skill_with_prior() -> None:
    mem = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=[str(uuid4()), str(uuid4())],
    )
    plan = expand_intent(
        "skill_count",
        question="python?",
        slots={
            "skill": "Python",
            "count_only": True,
            "refers_to_prior": True,
            "employee_ids": list(mem.last_employee_ids),
        },
        memory=mem,
    )
    assert plan is not None
    assert any(n.name == "resume_search" for n in plan.nodes)
    assert any(n.name == "sql" for n in plan.nodes)


def test_tool_selector_confidence_clarify() -> None:
    llm = FakeLLM()

    async def _low(*_a, **_k):
        return json.dumps(
            {
                "confidence": 0.2,
                "intent": "unknown",
                "slots": {},
                "selected": [],
                "clarify_question": "Which person?",
            }
        )

    llm.complete = _low  # type: ignore[method-assign]
    planner = ToolSelectingPlanner(llm, _registry(), hitl_status_writes=False)
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)

    async def _run() -> None:
        result = await planner.plan("something vague", auth=auth)
        assert result.plan.clarify_question
        assert not result.plan.nodes

    asyncio.run(_run())


def test_tool_selector_write_band_blocks_status() -> None:
    """0.55–0.75 allows reads but blocks set_status writes."""

    async def _mid(*_a, **_k):
        return json.dumps(
            {
                "confidence": 0.65,
                "intent": "set_status",
                "slots": {"name": "Carol Garcia", "status": True},
                "selected": [],
            }
        )

    llm = FakeLLM()
    llm.complete = _mid  # type: ignore[method-assign]
    planner = ToolSelectingPlanner(llm, _registry(), hitl_status_writes=False)
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)

    async def _run() -> None:
        result = await planner.plan("maybe set Carol status true?", auth=auth)
        assert result.plan.clarify_question
        assert not result.plan.nodes
        assert not result.needs_hitl

    asyncio.run(_run())


def test_tool_selector_hitl_status() -> None:
    async def _sel(*_a, **_k):
        return json.dumps(
            {
                "confidence": 0.95,
                "intent": "set_status",
                "slots": {
                    "name": "Carol Garcia",
                    "status": True,
                    "employee_ids": ["00000000-0000-0000-0000-000000000001"],
                },
                "selected": [],
            }
        )

    llm = FakeLLM()
    llm.complete = _sel  # type: ignore[method-assign]
    planner = ToolSelectingPlanner(llm, _registry(), hitl_status_writes=True)
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)

    async def _run() -> None:
        result = await planner.plan(
            "set Carol Garcia status to true", auth=auth
        )
        assert result.needs_hitl
        assert result.mode == "tool_select_hitl"
        assert result.plan.pending_tool_fact is not None
        pending = result.plan.pending_tool_fact["value"]
        assert pending["name"] == "Carol Garcia"
        # Named HITL must not stash prior/slot ids that can steal the confirm.
        assert pending["employee_ids"] == []
        assert "confirm status update" in (result.plan.clarify_question or "").lower()
        assert "carol" in (result.plan.clarify_question or "").lower()

    asyncio.run(_run())


def test_tool_selector_bounded_repair() -> None:
    calls = {"n": 0}

    async def _bad(*_a, **_k):
        calls["n"] += 1
        return "not-json"

    llm = FakeLLM()
    llm.complete = _bad  # type: ignore[method-assign]
    planner = ToolSelectingPlanner(
        llm, _registry(), max_repairs=2, hitl_status_writes=False
    )
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)

    async def _run() -> None:
        result = await planner.plan("whatever", auth=auth)
        assert result.mode == "tool_select_failed"
        assert result.repairs == 2
        # initial + 2 repairs = 3 attempts
        assert calls["n"] == 3
        assert result.plan.clarify_question

    asyncio.run(_run())


def test_tool_selector_validate_repair_then_ok() -> None:
    calls = {"n": 0}

    async def _sel(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "confidence": 0.9,
                    "intent": "count",
                    "slots": {"department": "Engineering", "count_only": True},
                    "selected": [],
                }
            )
        return json.dumps(
            {
                "confidence": 0.9,
                "intent": "clarify",
                "slots": {"clarify_question": "Which department?"},
                "selected": [],
                "clarify_question": "Which department?",
            }
        )

    def _validate(plan, _auth) -> None:
        if any(n.name == "sql" for n in plan.nodes):
            raise ValueError("simulated validator reject")

    llm = FakeLLM()
    llm.complete = _sel  # type: ignore[method-assign]
    planner = ToolSelectingPlanner(llm, _registry(), max_repairs=2)
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)

    async def _run() -> None:
        result = await planner.plan("how many?", auth=auth, validate=_validate)
        assert calls["n"] == 2
        assert result.mode == "tool_select_repair"
        assert result.plan.clarify_question

    asyncio.run(_run())


def test_tool_selection_model_parses() -> None:
    sel = ToolSelection.model_validate(
        {
            "confidence": 0.8,
            "intent": "count",
            "slots": {"department": "Sales"},
            "selected": [],
        }
    )
    assert sel.intent == "count"
