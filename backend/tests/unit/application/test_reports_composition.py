from __future__ import annotations

from app.application.planning.heuristic_planner import try_heuristic_plan
from app.application.understanding.extract_query_state import extract_query_state
from app.domain.schema_catalog import default_employee_catalog
from app.tools.employee.tool import extract_reports_subject


def test_extract_reports_subject() -> None:
    assert extract_reports_subject("who reports to Alice Nguyen?") == "Alice Nguyen"
    assert extract_reports_subject("list Bob Mueller's team") == "Bob Mueller"
    assert extract_reports_subject("who on Ivy Chen's team knows Python?") == "Ivy Chen"


def test_reports_plan() -> None:
    plan = try_heuristic_plan("who reports to Alice Nguyen?")
    assert plan is not None
    assert plan.nodes[0].params.get("action") == "reports"


def test_reports_skill_place_composition() -> None:
    plan = try_heuristic_plan(
        "who on Alice Nguyen's team knows Kubernetes and is in Dubai?"
    )
    assert plan is not None
    names = [n.name for n in plan.nodes]
    assert "employee" in names
    assert "resume_search" in names
    assert "intersect_ids" in names
    assert any((n.params or {}).get("action") == "reports" for n in plan.nodes)


def test_query_state_reports_intent() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("who reports to Eva Kim?", catalog=catalog)
    assert state.intent == "reports"
    assert state.person_name and "eva" in state.person_name.lower()
