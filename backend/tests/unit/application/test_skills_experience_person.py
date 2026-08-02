from __future__ import annotations

from app.application.understanding.experience import extract_experience
from app.application.understanding.person_skills import extract_person_skills
from app.application.understanding.plan_from_query_state import (
    _experience_person_plan,
    _projects_person_plan,
    _skills_person_plan,
)
from app.application.understanding.projects import extract_projects
from app.tools.resume_search.experience import parse_experience
from app.tools.resume_search.person_skills import parse_skills_list
from app.tools.resume_search.projects import parse_projects


def test_parse_skills_and_experience() -> None:
    assert parse_skills_list("Python, Communication, Collaboration") == [
        "Python",
        "Communication",
        "Collaboration",
    ]
    assert parse_experience(
        "- Product Designer at Acme Corp\n- Delivered projects using Python"
    ) == [
        "Product Designer at Acme Corp",
        "Delivered projects using Python",
    ]


def test_parse_strips_enrichment_header() -> None:
    skills = (
        "Maya Khan — Product Designer, Product (Dubai, UAE)\n"
        "Skills\n"
        "Python, Communication, Collaboration"
    )
    assert parse_skills_list(skills) == [
        "Python",
        "Communication",
        "Collaboration",
    ]
    experience = (
        "Maya Khan — Product Designer, Product (Dubai, UAE)\n"
        "Experience\n"
        "- Product Designer at Acme Corp\n"
        "- Delivered projects using Python"
    )
    assert parse_experience(experience) == [
        "Product Designer at Acme Corp",
        "Delivered projects using Python",
    ]
    # Summary bleed must not become "experience".
    summary = (
        "Maya Khan — Product Designer, Product (Dubai, UAE)\n"
        "Summary\n"
        "Experienced Product Designer in Product based in Dubai, UAE."
    )
    assert parse_experience(summary) is None


def test_parse_projects() -> None:
    text = (
        "Quinn Martin — Head of Product, Product (Paris, France)\n"
        "Projects\n"
        "- Internal platform initiative involving Kubernetes"
    )
    assert parse_projects(text) == [
        "Internal platform initiative involving Kubernetes"
    ]


def test_extract_person_skills() -> None:
    req = extract_person_skills("What skills does Maya Khan have?")
    assert req.matched and req.person_name == "Maya Khan" and req.skill is None
    req = extract_person_skills("Does Maya Khan know Python?")
    assert req.matched and req.person_name == "Maya Khan" and req.skill == "Python"
    assert not extract_person_skills("Who knows Python?").matched


def test_extract_experience_not_location() -> None:
    req = extract_experience("Where has Maya Khan worked?")
    assert req.matched and req.person_name == "Maya Khan"
    assert not extract_experience("Where does Maya Khan live?").matched


def test_experience_lists_all_bullets() -> None:
    from app.tools.resume_search.experience import build_person_answer, ExperienceFact

    fact = ExperienceFact(
        employee_id="e1",
        name="Maya Khan",
        items=[
            "Product Designer at Acme Corp",
            "Delivered projects using Python",
        ],
    )
    ans = build_person_answer([fact], name_asked="Maya Khan")
    assert "Product Designer at Acme Corp" in ans
    assert "Delivered projects using Python" in ans
    assert "experience at:" not in ans

    only = build_person_answer(
        [fact], name_asked="Maya Khan", completeness_ask=True
    )
    assert only.startswith("No —")
    assert "2 experience entries" in only
    assert "no further experience" in only.lower()

    single = ExperienceFact(
        employee_id="e1", name="Maya Khan", items=["Product Designer at Acme Corp"]
    )
    only_one = build_person_answer(
        [single], name_asked="Maya Khan", completeness_ask=True
    )
    assert only_one.startswith("Yes —")
    assert "only one experience" in only_one.lower()


def test_extract_experience_completeness_followup() -> None:
    for q in (
        "is that the only experience?",
        "is it the only one experience?",
        "are there more experiences?",
        "what else has she done?",
    ):
        req = extract_experience(q)
        assert req.matched and req.completeness_ask, q


def test_extract_projects_pronoun() -> None:
    assert extract_projects("what is her projects?").matched
    assert extract_projects("give information about her project").matched
    req = extract_projects("What projects does Quinn Martin have?")
    assert req.matched and req.person_name == "Quinn Martin"


def test_plans_use_resume_purposes() -> None:
    skills = _skills_person_plan(person_name="Maya Khan", skill="Python")
    assert skills.nodes[0].params["purpose"] == "skills_person"
    assert skills.nodes[0].params["skill"] == "Python"
    exp = _experience_person_plan(person_name="Maya Khan")
    assert exp.nodes[0].params["purpose"] == "experience_person"
    proj = _projects_person_plan(person_name="Quinn Martin")
    assert proj.nodes[0].params["purpose"] == "projects_person"
