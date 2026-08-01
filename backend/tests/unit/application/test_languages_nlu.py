from __future__ import annotations

from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.languages import extract_language
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.domain.schema_catalog import default_employee_catalog
from app.tools.resume_search.languages import parse_languages


def test_parse_languages_labelled() -> None:
    assert parse_languages("Languages\nEnglish, German") == ["English", "German"]
    assert parse_languages("Speaks french and arabic") == ["French", "Arabic"]


def test_extract_who_speaks() -> None:
    req = extract_language("who speaks German?")
    assert req.matched and req.scope == "cohort" and req.language == "German"


def test_languages_plan_cohort() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("who speaks French?", catalog=catalog)
    assert state.intent == "languages"
    assert state.language == "French"
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[0].params.get("purpose") == "languages_cohort"
