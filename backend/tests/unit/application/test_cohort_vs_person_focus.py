"""Cohort asks must not bind to a focused person."""
from __future__ import annotations

from app.application.planning.heuristic_planner import (
    elliptical_bound_person_plan,
    is_cohort_question,
)
from app.application.planning.intent_templates import expand_intent
from app.domain.enums import Role
from app.domain.session import ActiveReferent, SessionMemory


def _focused_memory() -> SessionMemory:
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        active_referent=ActiveReferent(
            ids=["00000000-0000-0000-0000-000000000099"],
            label="Kara Petrov",
        ),
    )


def test_cohort_detector() -> None:
    assert is_cohort_question("how many employees have status active ?")
    assert is_cohort_question("give all employees names who lives in dubai")
    assert is_cohort_question("how many employees knows python ?")
    assert not is_cohort_question("what is her status")
    assert not is_cohort_question("where does she live")


def test_elliptical_skips_cohort_when_person_focused() -> None:
    mem = _focused_memory()
    assert elliptical_bound_person_plan(
        "how many employees have status active ?", memory=mem
    ) is None
    assert elliptical_bound_person_plan(
        "give all employees names who lives in dubai", memory=mem
    ) is None


def test_skill_count_uses_skill_rag_query() -> None:
    plan = expand_intent(
        "skill_count",
        question="how many employees knows python ?",
        slots={"skill": "Python", "count_only": True},
        memory=None,
    )
    assert plan is not None
    r1 = next(n for n in plan.nodes if n.id == "r1")
    q = (r1.params or {}).get("question", "").lower()
    assert "python" in q
    assert "who knows" in q
    assert "how many" not in q


def test_count_status_active_filters_employment_status() -> None:
    plan = expand_intent(
        "count",
        question="how many employees have status active ?",
        slots={"count_only": True, "want_count": True},
        memory=None,
    )
    assert plan is not None
    sql = next(n for n in plan.nodes if n.name == "sql" and (n.params or {}).get("count_only"))
    assert (sql.params or {}).get("filters", {}).get("employment_status") == "active"
