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


def test_extract_pronoun_known_languages() -> None:
    for q in (
        "give me information about her known languages",
        "what languages does she speak?",
        "her languages",
    ):
        req = extract_language(q)
        assert req.matched and req.scope == "person", q
        assert req.person_name is None, q


def test_extract_named_person_languages() -> None:
    req = extract_language("Tell me about Maya Khan's languages")
    assert req.matched and req.scope == "person"
    assert req.person_name == "Maya Khan"


def test_bound_pronoun_languages_uses_resume_search() -> None:
    from app.application.planning.bound_person import bound_person_attribute_plan

    plan = bound_person_attribute_plan(
        "give me information about her known languages",
        employee_id="e00e3f63-5890-5627-8ca7-b2adabbe51b4",
        display_name="Maya Khan",
    )
    assert len(plan.nodes) == 1
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[0].params.get("purpose") == "languages_person"
    assert plan.nodes[0].params.get("employee_ids") == [
        "e00e3f63-5890-5627-8ca7-b2adabbe51b4"
    ]


def test_languages_plan_cohort() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("who speaks French?", catalog=catalog)
    assert state.intent == "languages"
    assert state.language == "French"
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[0].params.get("purpose") == "languages_cohort"
