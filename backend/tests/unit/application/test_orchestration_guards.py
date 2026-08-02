"""Hermetic checks for orchestration / hallucination-control contracts.

These pin the Wave A–D success criteria without a live API:
  (a) invalid employee_ids → SQL AND 1=0 (never org dump)
  (b) elliptical follow-ups stay on tools when person focus exists
  (c) weak / score-gated RAG does not create a cohort
  (d) pronoun with 2+ entities clarifies
plus skill∩location composition, SELECT *, and llm_format claim checks.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from app.adapters.sql.query_builder import QueryBuilder
from app.adapters.sql.validator import validate_sql
from app.application.memory.context_view import _ELLIPTICAL_ATTR_RE
from app.application.planning.heuristic_planner import (
    _PRONOUN_ONLY_RE,
    try_heuristic_plan,
)
from app.application.planning.plan_compiler import PlanCompiler
from app.application.response.response_formatter import _has_ungrounded_claims
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.operators.extract_ids import ExtractEmployeeIds
from app.domain.operators.intersect_ids import IntersectIds
from app.domain.query_state import FilterSlot, QueryState
from app.domain.session import EntityRef, SessionMemory
from app.domain.tools.registry import ToolRegistry


E1 = UUID("00000000-0000-0000-0000-000000000001")
E2 = UUID("00000000-0000-0000-0000-000000000002")


def _mem(*people: tuple[UUID, str]) -> SessionMemory:
    entities = [
        EntityRef(employee_id=eid, display_name=name) for eid, name in people
    ]
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        entity_memory=list(entities),
        last_listed=list(entities),
        last_employee_ids=[str(eid) for eid, _ in people],
    )


def test_invalid_employee_ids_force_empty_sql() -> None:
    sql, _ = QueryBuilder().build(
        columns=["id"],
        filters={"employee_ids": ["not-a-uuid", "also-bad"]},
    )
    assert "1=0" in sql
    assert "eid_" not in sql


def test_empty_employee_ids_force_empty_sql() -> None:
    sql, _ = QueryBuilder().build(
        columns=["id"],
        filters={"employee_ids": []},
    )
    assert "1=0" in sql


def test_select_star_rejected_by_validator() -> None:
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    with pytest.raises(Exception) as exc:
        validate_sql("SELECT * FROM employees LIMIT 10", auth)
    assert "SELECT *" in str(exc.value) or "explicit" in str(exc.value).lower()


def test_score_gated_empty_rag_yields_no_cohort_ids() -> None:
    ids = ExtractEmployeeIds().run(
        {"score_gated": True, "employee_ids": [], "hits": []}
    )
    assert ids == []


def test_weak_hit_scores_filtered_from_cohort() -> None:
    ids = ExtractEmployeeIds().run(
        {
            "hits": [
                {
                    "employee_id": str(E1),
                    "score": 0.10,
                    "snippets": [{"score": 0.10}],
                },
                {
                    "employee_id": str(E2),
                    "score": 0.55,
                    "snippets": [{"score": 0.55}],
                },
            ]
        }
    )
    assert ids == [str(E2)]


def test_empty_location_intersect_does_not_keep_skill_ids() -> None:
    left = [str(E1), str(E2)]
    assert IntersectIds().run(left, {"other": []}) == []
    assert IntersectIds().run(left, {}) == left  # missing other → unchanged


def test_skill_and_location_plan_intersects_cohorts() -> None:
    state = QueryState(
        intent="skill_search",
        skill="Python",
        confidence=0.9,
        filters=[FilterSlot(field="city", op="eq", value="Berlin")],
    )
    plan = plan_from_query_state(state)
    assert plan is not None
    names = [n.name for n in plan.nodes]
    ids = [n.id for n in plan.nodes]
    assert "resume_search" in names
    assert "intersect_ids" in names
    assert "sql" in names
    assert "skloc" in ids
    assert "loc" in ids
    # Place never leaks into SQL filters.
    sql = next(n for n in plan.nodes if n.name == "sql")
    assert "city" not in (sql.params or {}).get("filters", {})


def test_heuristic_skill_in_city_builds_intersect() -> None:
    plan = try_heuristic_plan("Who knows Python in Berlin?")
    assert plan is not None
    assert any(n.id == "skloc" for n in plan.nodes)
    assert any(
        n.name == "resume_search" and (n.params or {}).get("purpose") == "location_cohort"
        for n in plan.nodes
    )


def test_pronoun_with_two_entities_clarifies() -> None:
    mem = _mem((E1, "Kara Petrov"), (E2, "Maya Khan"))
    plan = try_heuristic_plan("where does she live?", memory=mem)
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question
    assert "which" in plan.clarify_question.lower()


def test_pronoun_with_unique_binding_resolves() -> None:
    mem = _mem((E1, "Kara Petrov"), (E2, "Maya Khan"))
    mem.person_bindings = {"she": str(E1), "her": str(E1)}
    plan = try_heuristic_plan("where does she live?", memory=mem)
    assert plan is not None
    assert plan.nodes
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[0].params["employee_ids"] == [str(E1)]


def test_elliptical_attr_regex_matches_followups() -> None:
    for q in ("her email?", "his manager", "their department?", "job title?"):
        assert _ELLIPTICAL_ATTR_RE.search(q) or _PRONOUN_ONLY_RE.search(q), q


def test_ordinal_birthday_after_list_uses_resume_search() -> None:
    mem = _mem((E1, "Kara Petrov"), (E2, "Maya Khan"))
    compiler = PlanCompiler(llm=None, tools=ToolRegistry())

    async def _run() -> None:
        plan, mode, _meta = await compiler.compile(
            "give me the first persons date of birth",
            auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
            memory=mem,
        )
        assert mode == "heuristic_list_referent"
        assert plan.nodes[0].name == "resume_search"
        assert plan.nodes[0].params["purpose"] == "birthday_person"
        assert plan.nodes[0].params["employee_ids"] == [str(E1)]

    import asyncio

    asyncio.run(_run())


def test_llm_format_rejects_ungrounded_proper_name() -> None:
    payloads = [{"count": 2, "rows": [{"first_name": "Alice", "last_name": "Nguyen"}]}]
    assert _has_ungrounded_claims("Bob Martinez works in Sales.", payloads) is True
    assert _has_ungrounded_claims("Alice Nguyen is listed.", payloads) is False


def test_llm_format_rejects_ungrounded_large_number() -> None:
    payloads = [{"count": 3, "rows": []}]
    assert _has_ungrounded_claims("There are 47 engineers.", payloads) is True
    assert _has_ungrounded_claims("There are 3 engineers.", payloads) is False


def test_nl2sql_heuristic_tail_clarifies_instead() -> None:
    """Residual analytics must not emit an nl2sql node."""
    plan = try_heuristic_plan("how many employees were hired on Tuesdays?")
    assert plan is not None
    assert not any(
        n.name == "sql" and (n.params or {}).get("mode") == "nl2sql" for n in plan.nodes
    )
    if not plan.nodes:
        assert plan.clarify_question


def test_unknown_place_clarifies_not_generic_plan_fail() -> None:
    plan = try_heuristic_plan("List employees in Atlantis")
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question
    assert "don't recognize" in plan.clarify_question.lower()
    assert "atlantis" in plan.clarify_question.lower()


def test_employee_profile_dict_binds_id_for_location_node() -> None:
    from app.application.execution.node_runner import _as_id_list

    eid = str(E1)
    assert _as_id_list({"id": eid, "first_name": "Alice", "last_name": "Nguyen"}) == [
        eid
    ]
    assert _as_id_list({"employee_ids": [eid, str(E2)]}) == [eid, str(E2)]


def test_pronoun_manager_uses_bound_id() -> None:
    mem = _mem((E1, "Alice Nguyen"))
    mem.person_bindings = {
        "she": str(E1),
        "her": str(E1),
        "he": str(E1),
        "him": str(E1),
        "his": str(E1),
        "hers": str(E1),
    }
    plan = try_heuristic_plan("her manager?", memory=mem)
    assert plan is not None
    assert plan.nodes
    assert plan.nodes[0].name == "employee"
    assert plan.nodes[0].params.get("action") == "manager"
    assert plan.nodes[0].params.get("employee_id") == str(E1)
