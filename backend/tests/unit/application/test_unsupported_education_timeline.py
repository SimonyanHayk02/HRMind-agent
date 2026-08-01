from __future__ import annotations

from uuid import UUID

from app.application.planning.heuristic_planner import try_heuristic_plan
from app.application.planning.unsupported import (
    is_unsupported_education_timeline,
    is_unsupported_topic,
    unsupported_answer_for,
)
from app.domain.enums import Role
from app.domain.session import SessionMemory


def test_education_timeline_is_unsupported() -> None:
    for q in (
        "how long have he been learning there?",
        "how long has she been studying?",
        "when did he graduate?",
        "what year did she start school?",
        "education start date",
    ):
        assert is_unsupported_education_timeline(q), q
        assert is_unsupported_topic(q), q
        ans = unsupported_answer_for(q)
        assert "don't have enough information" in ans.lower(), ans
        assert "education level" in ans.lower() or "bootcamp" in ans.lower(), ans


def test_company_tenure_not_treated_as_education_timeline() -> None:
    q = "how long has he been here?"
    assert not is_unsupported_education_timeline(q)
    q2 = "what is the average tenure in Engineering?"
    assert not is_unsupported_education_timeline(q2)


def test_heuristic_refuses_education_timeline_without_pronoun_clarify() -> None:
    eid = UUID("00000000-0000-0000-0000-000000000099")
    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        person_bindings={"he": str(eid), "him": str(eid), "his": str(eid)},
    )
    plan = try_heuristic_plan(
        "how long have he been learning there?", memory=memory
    )
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question
    assert "don't have enough information" in plan.clarify_question.lower()
    assert "which" not in plan.clarify_question.lower()
