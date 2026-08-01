from __future__ import annotations

from datetime import date

from app.application.understanding.birthday import extract_birthday
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.domain.schema_catalog import default_employee_catalog
from app.tools.resume_search.birthday import (
    BirthdayFact,
    build_cohort_answer,
    build_person_answer,
    facts_from_hits,
    format_birth_date,
    parse_birth_date,
)

TODAY = date(2026, 7, 31)


def test_parse_birth_date_formats() -> None:
    assert parse_birth_date("Date of Birth: 12 March 1991") == date(1991, 3, 12)
    assert parse_birth_date("DOB: March 12, 1991") == date(1991, 3, 12)
    assert parse_birth_date("Date of Birth: 1991-03-12") == date(1991, 3, 12)
    assert parse_birth_date("Date of Birth: 12/03/1991") == date(1991, 3, 12)
    assert parse_birth_date("born 3 April 1980") == date(1980, 4, 3)
    assert parse_birth_date("Skills\nPython, Communication") is None
    assert parse_birth_date("") is None


def test_parse_birth_date_prefers_labelled_line() -> None:
    text = "Experience\n- Engineer since 2015-06-01\n\nPersonal\nDate of Birth: 9 May 1988"
    assert parse_birth_date(text) == date(1988, 5, 9)


def test_format_birth_date_roundtrip() -> None:
    born = date(1994, 3, 1)
    assert format_birth_date(born) == "1 March 1994"
    assert parse_birth_date(f"Date of Birth: {format_birth_date(born)}") == born


def test_facts_from_hits_one_per_employee() -> None:
    hits = [
        {
            "id": "c1",
            "employee_id": "e1",
            "section": "Personal",
            "content": "Date of Birth: 12 March 1991",
            "metadata": {"employee_name": "Eva Kim"},
        },
        {
            "id": "c2",
            "employee_id": "e1",
            "section": "Skills",
            "content": "Python",
            "metadata": {"employee_name": "Eva Kim"},
        },
        {
            "id": "c3",
            "employee_id": "e2",
            "section": "Personal",
            "content": "Date of Birth: 31 July 1985",
            "metadata": {"employee_name": "Bob Smith"},
        },
    ]
    facts = facts_from_hits(hits)
    assert {f.employee_id for f in facts} == {"e1", "e2"}
    assert {f.name for f in facts} == {"Eva Kim", "Bob Smith"}


def test_person_answer_wishes_on_the_day() -> None:
    fact = BirthdayFact("e1", "Bob Smith", date(1985, 7, 31))
    answer = build_person_answer([fact], name_asked="Bob Smith", today=TODAY)
    assert "Happy birthday, Bob Smith!" in answer
    assert "41" in answer


def test_person_answer_future_date_and_age() -> None:
    fact = BirthdayFact("e1", "Eva Kim", date(1991, 3, 12))
    plain = build_person_answer([fact], name_asked="Eva Kim", today=TODAY)
    assert "12 March 1991" in plain

    wish = build_person_answer(
        [fact], name_asked="Eva Kim", wants_wish=True, today=TODAY
    )
    assert "happy birthday" in wish.lower()

    age = build_person_answer([fact], name_asked="Eva Kim", wants_age=True, today=TODAY)
    assert age == "Eva Kim is 35 (born 12 March 1991)."


def test_person_answer_lists_namesakes_and_handles_missing() -> None:
    facts = [
        BirthdayFact("e1", "Eva Kim", date(1991, 3, 12)),
        BirthdayFact("e2", "Eva Kim", date(1977, 9, 30)),
    ]
    ambiguous = build_person_answer(facts, name_asked="Eva Kim", today=TODAY)
    assert "2 employees match" in ambiguous
    assert "30 September 1977" in ambiguous

    missing = build_person_answer([], name_asked="Zzz Nobody", today=TODAY)
    assert "couldn't find a date of birth" in missing


def test_cohort_answers() -> None:
    today_fact = BirthdayFact("e1", "Carol Garcia", date(1976, 7, 31))
    later_fact = BirthdayFact("e2", "Eva Kim", date(1991, 3, 12))

    assert "Happy birthday" in build_cohort_answer(
        [today_fact, later_fact], scope="today", today=TODAY
    )
    assert "No employee has a birthday today" in build_cohort_answer(
        [later_fact], scope="today", today=TODAY
    )

    march = build_cohort_answer([today_fact, later_fact], scope="month", month=3, today=TODAY)
    assert "1 employee has a birthday in March" in march
    assert "Eva Kim" in march

    upcoming = build_cohort_answer([today_fact, later_fact], scope="upcoming", today=TODAY)
    assert "Carol Garcia" in upcoming and "Eva Kim" not in upcoming

    closest = build_cohort_answer(
        [later_fact, BirthdayFact("e3", "Ada Lee", date(1990, 8, 2))],
        scope="closest",
        today=TODAY,
    )
    assert "Ada Lee" in closest and "closest upcoming birthday" in closest
    assert "Eva Kim" not in closest


def test_extract_birthday_person_phrasings() -> None:
    req = extract_birthday("when is Eva Kim's birthday?")
    assert req.matched and req.scope == "person" and req.person_name == "Eva Kim"

    req = extract_birthday("wish Eva Kim a happy birthday")
    assert req.matched and req.person_name == "Eva Kim" and req.wants_wish

    req = extract_birthday("say happy birthday to Bob Mueller")
    assert req.matched and req.person_name == "Bob Mueller" and req.wants_wish

    req = extract_birthday("how old is Ivy Chen?")
    assert req.matched and req.person_name == "Ivy Chen" and req.wants_age

    req = extract_birthday("when was Bob Smith born?")
    assert req.matched and req.person_name == "Bob Smith"

    req = extract_birthday("what is the date of birth of Carol Garcia")
    assert req.matched and req.person_name == "Carol Garcia"


def test_extract_birthday_cohort_phrasings() -> None:
    assert extract_birthday("whose birthday is today?").scope == "today"
    assert extract_birthday("any birthdays today").scope == "today"

    july = extract_birthday("who has a birthday in July?")
    assert july.scope == "month" and july.month == 7

    assert extract_birthday("upcoming birthdays").scope == "upcoming"
    assert extract_birthday("any birthdays coming up?").scope == "upcoming"

    assert extract_birthday(
        "which persons birthday is the closest one ?"
    ).scope == "closest"
    assert extract_birthday(
        "i want to know the name of a person which birthday is the closest with current date"
    ).scope == "closest"
    assert extract_birthday("closest birthday").scope == "closest"
    assert extract_birthday("closest birthday").person_name is None


def test_extract_birthday_ignores_unrelated_and_status_questions() -> None:
    assert not extract_birthday("how many employees do we have?").matched
    assert not extract_birthday("change the status of Eva Kim to true").matched
    assert not extract_birthday("who knows Python?").matched


def test_birthday_never_captures_keywords_as_a_name() -> None:
    req = extract_birthday("whose birthday is it?")
    assert req.person_name is None

    req = extract_birthday("list employee birthdays")
    assert req.person_name is None


def test_person_birthday_plan_uses_resume_search_only() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("when is Eva Kim's birthday?", catalog=catalog)
    assert state.intent == "birthday"
    assert state.birthday_scope == "person"
    assert state.person_name == "Eva Kim"

    plan = plan_from_query_state(state)
    assert plan is not None
    names = [n.name for n in plan.nodes]
    assert names == ["resume_search"]
    assert "sql" not in names
    assert "employee" not in names
    assert plan.nodes[0].params.get("purpose") == "birthday_person"
    assert plan.nodes[0].params.get("name") == "Eva Kim"
    assert plan.response_strategy == "template"


def test_cohort_birthday_plan_uses_resume_search_only() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("whose birthday is today?", catalog=catalog)
    assert state.intent == "birthday"
    assert state.birthday_scope == "today"

    plan = plan_from_query_state(state)
    assert plan is not None
    assert [n.name for n in plan.nodes] == ["resume_search"]
    assert plan.nodes[0].params.get("purpose") == "birthday_cohort"
    assert plan.nodes[0].params.get("scope") == "today"

    state = extract_query_state("who has a birthday in July?", catalog=catalog)
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[0].params.get("month") == 7


def test_birthday_without_subject_clarifies() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("when is the birthday?", catalog=catalog)
    assert state.intent == "birthday"

    plan = plan_from_query_state(state, min_confidence=0.5)
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question is not None


def test_no_birthday_phrasing_ever_reaches_sql() -> None:
    """Birth dates exist only in resume text, so the SQL and employee tools are
    never part of a birthday plan — whatever the phrasing."""
    catalog = default_employee_catalog()
    questions = [
        "when is Carol Garcia's birthday?",
        "how old is Alice Nguyen?",
        "say happy birthday to Carol Garcia",
        "wish Alice Bauer a happy birthday",
        "when was Hugo Marino born?",
        "what is the date of birth of Carol Garcia",
        "whose birthday is today?",
        "any birthdays today",
        "who has a birthday in July?",
        "upcoming birthdays",
        "list employee birthdays",
    ]
    for question in questions:
        state = extract_query_state(question, catalog=catalog)
        assert state.intent == "birthday", question
        plan = plan_from_query_state(state, min_confidence=0.5)
        assert plan is not None, question
        names = {n.name for n in plan.nodes}
        assert names <= {"resume_search"}, f"{question} -> {sorted(names)}"


def test_age_question_is_not_treated_as_unsupported_topic() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("how old is Ivy Chen?", catalog=catalog)
    assert state.intent == "birthday"
    assert state.wants_age is True
