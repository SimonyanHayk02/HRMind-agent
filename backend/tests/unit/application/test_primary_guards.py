"""Primary tool-select mode must keep deterministic guards ahead of the LLM."""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from app.adapters.llm.fake_llm import FakeLLM
from app.application.planning.plan_compiler import PlanCompiler
from app.application.response.refusal import empty_refine_answer
from app.application.planning.heuristic_planner import try_heuristic_plan
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import EntityRef, LastFocus, SessionMemory
from app.domain.tools.registry import ToolRegistry

E1 = UUID("00000000-0000-0000-0000-000000000001")
E2 = UUID("00000000-0000-0000-0000-000000000002")
AUTH = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)


def _compiler() -> PlanCompiler:
    return PlanCompiler(llm=FakeLLM(), tools=ToolRegistry(), tool_selecting_mode="primary")


def _memory(**kwargs) -> SessionMemory:
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        **kwargs,
    )


@pytest.mark.parametrize(
    "question",
    [
        "in how different cities do we have employees?",
        "in how different countries do we have employees?",
        "how many different countries do we have employees in?",
    ],
)
def test_primary_facet_guard_beats_org_count(question: str) -> None:
    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile(question, auth=AUTH, memory=_memory())
        assert mode == "guard_facet", mode
        assert any(
            (n.params or {}).get("purpose") == "location_facet"
            or (n.params or {}).get("count_distinct")
            or (n.params or {}).get("facet")
            for n in plan.nodes
        )

    asyncio.run(_run())


def test_primary_facet_names_please_lists_dimension() -> None:
    mem = _memory(
        last_focus=LastFocus(kind="facet", dimension="country", values=["USA", "UK"]),
    )

    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile("names please", auth=AUTH, memory=mem)
        assert mode == "guard_facet_list", mode
        assert plan.nodes

    asyncio.run(_run())


def test_primary_tenure_and_senior_guards() -> None:
    async def _run() -> None:
        tenure, tmode, _meta = await _compiler().compile(
            "what is the average tenure in Engineering?", auth=AUTH, memory=_memory()
        )
        assert tmode == "guard_tenure", tmode
        assert tenure.nodes[0].params.get("template") in {
            "agg_tenure",
            "agg_tenure_by_dept",
        }

        senior, smode, _meta = await _compiler().compile(
            "who is the most senior in Engineering?", auth=AUTH, memory=_memory()
        )
        assert smode == "guard_tenure", smode
        assert senior.nodes[0].params.get("template") == "longest_tenured"

    asyncio.run(_run())


def test_primary_unknown_place_clarifies() -> None:
    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile(
            "List employees in Atlantis", auth=AUTH, memory=_memory()
        )
        assert mode == "guard_unknown_place", mode
        assert plan.clarify_question
        assert "don't recognize" in plan.clarify_question.lower()
        assert "Atlantis" in plan.clarify_question

    asyncio.run(_run())


def test_primary_empty_anaphora_clarifies() -> None:
    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile("of them?", auth=AUTH, memory=_memory())
        assert mode == "guard_empty_anaphora", mode
        assert plan.clarify_question
        text = plan.clarify_question.lower()
        assert "which" in text or "previous" in text

    asyncio.run(_run())


def test_primary_empty_cohort_list_followup() -> None:
    """After a zero refine, 'list their names' must not invent a department roster."""

    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile(
            "list their names please", auth=AUTH, memory=_memory()
        )
        assert mode == "guard_empty_cohort_list", mode
        assert not plan.nodes
        assert plan.refusal_code == "empty_cohort"
        assert plan.clarify_question
        low = plan.clarify_question.lower()
        assert "no names" in low or "previous" in low

    asyncio.run(_run())


def test_primary_list_followup_uses_prior_ids() -> None:
    mem = _memory(last_employee_ids=[str(E1), str(E2)])

    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile(
            "list their names please", auth=AUTH, memory=mem
        )
        assert mode == "guard_list_followup", mode
        assert any(n.name == "sql" for n in plan.nodes)

    asyncio.run(_run())


def test_primary_place_refine_intersects_prior_ids() -> None:
    mem = _memory(last_employee_ids=[str(E1), str(E2)])

    async def _run() -> None:
        plan, mode, _meta = await _compiler().compile(
            "how many of them are in Berlin?", auth=AUTH, memory=mem
        )
        assert mode == "guard_place_refine", mode
        assert any(n.name == "resume_search" for n in plan.nodes)
        assert any(n.name == "sql" for n in plan.nodes)
        sql = next(n for n in plan.nodes if n.name == "sql")
        assert sql.params.get("count_only") is True

    asyncio.run(_run())


def test_primary_place_refine_skips_single_person_focus() -> None:
    mem = _memory(last_employee_ids=[str(E1)])

    async def _run() -> None:
        _plan, mode, _meta = await _compiler().compile(
            "how much of them are from Dubai?", auth=AUTH, memory=mem
        )
        assert mode != "guard_place_refine", mode

    asyncio.run(_run())


def test_any_of_them_place_is_count_only() -> None:
    mem = _memory(
        last_employee_ids=[str(E1), str(E2)],
        last_listed=[
            EntityRef(employee_id=E1, display_name="A"),
            EntityRef(employee_id=E2, display_name="B"),
        ],
    )
    plan = try_heuristic_plan("any of them in Berlin?", memory=mem)
    assert plan is not None
    loc = next(n for n in plan.nodes if n.name == "resume_search")
    # Place refine may use location purpose; SQL count_only must be true.
    sql = next(n for n in plan.nodes if n.name == "sql")
    assert sql.params.get("count_only") is True


def test_primary_hire_window_guard() -> None:
    async def _run() -> None:
        for q in (
            "who joined in the last 90 days?",
            "who joined this quarter?",
        ):
            plan, mode, _meta = await _compiler().compile(q, auth=AUTH, memory=_memory())
            assert mode == "guard_hire_window", (q, mode)
            assert plan.nodes[0].name == "sql"
            filters = plan.nodes[0].params.get("filters") or {}
            assert "hire_date_gte" in filters

    asyncio.run(_run())


def test_skill_empty_refine_mentions_skill() -> None:
    msg = empty_refine_answer("which of them know Kubernetes?")
    assert "know" in msg.lower()
    assert "Kubernetes" in msg
    assert empty_refine_answer("of them?") == (
        "None of the previous set match that criteria."
    )
