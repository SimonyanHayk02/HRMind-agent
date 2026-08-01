from __future__ import annotations

from app.application.understanding.certifications import extract_certification
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.domain.schema_catalog import default_employee_catalog
from app.tools.resume_search.certifications import parse_certifications


def test_parse_certifications() -> None:
    text = "Certificates\n- AWS Certified\n- Professional certificate in Python"
    assert "AWS Certified" in (parse_certifications(text) or [])


def test_extract_cert_vs_skill() -> None:
    req = extract_certification("who has AWS Certified?")
    assert req.matched and req.certification == "AWS Certified"

    req = extract_certification("who has AWS experience?")
    assert not req.matched


def test_certifications_plan() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("who holds a CKA certification?", catalog=catalog)
    assert state.intent == "certifications"
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[0].params.get("purpose") == "certifications_cohort"
