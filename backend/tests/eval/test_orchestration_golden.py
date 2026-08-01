"""Golden checks for orchestration plan shapes (no live API)."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from app.application.memory.context_view import _ELLIPTICAL_ATTR_RE
from app.application.planning.heuristic_planner import (
    _PRONOUN_ONLY_RE,
    try_heuristic_plan,
)
from app.application.planning.plan_compiler import PlanCompiler
from app.application.routing.rule_router import RuleRouter
from app.domain.auth import AuthContext
from app.domain.enums import Role, RouterLabel
from app.domain.session import EntityRef, SessionMemory
from app.domain.tools.registry import ToolRegistry

GOLDEN = Path("data/golden/orchestration.jsonl")
E1 = UUID("00000000-0000-0000-0000-000000000001")
E2 = UUID("00000000-0000-0000-0000-000000000002")


def _two_entity_memory() -> SessionMemory:
    entities = [
        EntityRef(employee_id=E1, display_name="Kara Petrov"),
        EntityRef(employee_id=E2, display_name="Maya Khan"),
    ]
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        entity_memory=list(entities),
        last_listed=list(entities),
        last_employee_ids=[str(E1), str(E2)],
    )


def _load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text().splitlines()
        if line.strip()
    ]


def _check_case(case: dict) -> None:
    q = case["question"]
    check = case["check"]

    if check == "skill_location_intersect":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(n.id == "skloc" for n in plan.nodes), [
            (n.id, n.name) for n in plan.nodes
        ]
        assert any(
            (n.params or {}).get("purpose") == "location_cohort" for n in plan.nodes
        ), q
        return

    if check == "pronoun_two_entities_clarify":
        plan = try_heuristic_plan(q, memory=_two_entity_memory())
        assert plan is not None, q
        assert plan.nodes == [], q
        assert plan.clarify_question, q
        assert "which" in plan.clarify_question.lower(), plan.clarify_question
        return

    if check == "ordinal_birthday_resume":
        import asyncio

        compiler = PlanCompiler(llm=None, tools=ToolRegistry())

        async def _run() -> None:
            plan, mode = await compiler.compile(
                q,
                auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
                memory=_two_entity_memory(),
            )
            assert mode == "heuristic_list_referent", mode
            assert plan.nodes[0].name == "resume_search"
            assert plan.nodes[0].params["purpose"] == "birthday_person"

        asyncio.run(_run())
        return

    if check == "elliptical_attr_pattern":
        assert _ELLIPTICAL_ATTR_RE.search(q) or _PRONOUN_ONLY_RE.search(q), q
        return

    if check == "no_nl2sql":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert not any(
            n.name == "sql" and (n.params or {}).get("mode") == "nl2sql"
            for n in plan.nodes
        ), q
        return

    if check == "location_cohort":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(
            n.name == "resume_search"
            and (n.params or {}).get("purpose") == "location_cohort"
            for n in plan.nodes
        ), [(n.name, n.params) for n in plan.nodes]
        return

    if check == "skill_rag":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(n.name == "resume_search" for n in plan.nodes), q
        return

    if check == "languages_cohort":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(
            (n.params or {}).get("purpose") == "languages_cohort" for n in plan.nodes
        ), [(n.name, n.params) for n in plan.nodes]
        return

    if check == "certifications_cohort":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(
            (n.params or {}).get("purpose") == "certifications_cohort"
            for n in plan.nodes
        ), [(n.name, n.params) for n in plan.nodes]
        return

    if check == "hire_window":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(
            n.name == "sql"
            and "hire_date_gte" in ((n.params or {}).get("filters") or {})
            for n in plan.nodes
        ), [(n.name, n.params) for n in plan.nodes]
        return

    if check == "tenure_template":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(
            (n.params or {}).get("template") in {"agg_tenure", "agg_tenure_by_dept"}
            for n in plan.nodes
        ), [(n.name, n.params) for n in plan.nodes]
        return

    if check == "reports_action":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any(
            (n.params or {}).get("action") == "reports" for n in plan.nodes
        ), [(n.name, n.params) for n in plan.nodes]
        return

    if check == "reports_skill_place":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert any((n.params or {}).get("action") == "reports" for n in plan.nodes), q
        assert any(n.name == "resume_search" for n in plan.nodes), q
        assert any(n.name == "intersect_ids" for n in plan.nodes), q
        return

    if check == "unsupported_hris":
        plan = try_heuristic_plan(q)
        assert plan is not None, q
        assert plan.nodes == [], q
        assert plan.clarify_question, q
        assert "pto" in plan.clarify_question.lower() or "leave" in plan.clarify_question.lower()
        return

    if check == "greeting_route":
        assert RuleRouter().route(q) == RouterLabel.GREETING, q
        return

    raise AssertionError(f"unknown check {check!r} for {case.get('id')}")


@pytest.mark.parametrize(
    "case",
    _load_cases(),
    ids=[c["id"] for c in _load_cases()],
)
def test_orchestration_golden(case: dict) -> None:
    _check_case(case)
