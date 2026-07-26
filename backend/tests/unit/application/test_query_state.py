from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.domain.enums import Role
from app.domain.schema_catalog import default_employee_catalog
from app.domain.session import SessionMemory


def test_extract_of_them_from_usa() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("how much of them are from USA?", catalog=catalog)
    assert state.intent == "count"
    assert state.refers_to_prior is True
    assert any(f.field == "country" and f.value == "USA" for f in state.filters)
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[-1].params["filters"]["country"] == "USA"


def test_extract_education_bootcamp() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "how many employees have Bootcamp education?", catalog=catalog
    )
    assert state.intent == "count"
    assert any(f.field == "education" and f.value == "Bootcamp" for f in state.filters)
    plan = plan_from_query_state(state)
    assert plan is not None
    assert plan.nodes[-1].params["filters"]["education"] == "Bootcamp"


def test_extract_united_states_alias() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "how many of them are from the United States?", catalog=catalog
    )
    assert any(f.field == "country" and f.value == "USA" for f in state.filters)


def test_org_headcount_then_of_them_uses_universe() -> None:
    catalog = default_employee_catalog()
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        constraint_memory=[],
    )
    head = extract_query_state("How many employees do we have?", catalog=catalog)
    assert head.intent == "count"
    assert head.confidence >= 0.7
    plan = plan_from_query_state(head, memory=memory)
    assert plan is not None
    assert plan.nodes[-1].params.get("count_only") is True
