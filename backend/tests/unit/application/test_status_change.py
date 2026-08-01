from __future__ import annotations

from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.application.understanding.status_change import extract_status_change
from app.domain.schema_catalog import default_employee_catalog


def test_extract_status_change_patterns() -> None:
    req = extract_status_change("set Alice Nguyen's status to true")
    assert req.matched and req.person_name == "Alice Nguyen" and req.status_value is True

    req = extract_status_change("change the status of bob mueller to true")
    assert req.matched and req.person_name and req.person_name.lower() == "bob mueller"
    assert req.status_value is True

    req = extract_status_change("change the status of the bob mueller to true")
    assert req.matched and req.person_name and req.person_name.lower() == "bob mueller"

    req = extract_status_change(
        "change the status of 419c49ad-c06e-439c-944d-df9640e44ac5 to true"
    )
    assert req.matched and req.employee_id == "419c49ad-c06e-439c-944d-df9640e44ac5"


def test_extract_status_active_inactive_and_pronoun() -> None:
    req = extract_status_change("change her status to active")
    assert req.matched and req.status_value is True and req.person_name is None

    req = extract_status_change("set her status to inactive")
    assert req.matched and req.status_value is False

    req = extract_status_change("change Alice Nguyen's status to active")
    assert req.matched and req.person_name == "Alice Nguyen" and req.status_value is True

    req = extract_status_change("mark him as inactive")
    assert req.matched and req.status_value is False and req.person_name is None


def test_location_status_change_not_person_name() -> None:
    req = extract_status_change(
        "update status of employees which are living in Dubai to true"
    )
    assert req.matched
    assert req.status_value is True
    assert req.city == "Dubai"
    assert req.person_name is None


def test_update_status_does_not_capture_update_as_name() -> None:
    req = extract_status_change("update status of employees in Dubai to true")
    assert req.matched
    assert req.person_name is None
    assert req.city == "Dubai"


def test_extract_query_state_set_status_intent() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "change the status of Eva Kim to true", catalog=catalog
    )
    assert state.intent == "set_status"
    assert state.person_name and state.person_name.lower() == "eva kim"
    assert state.status_value is True

    plan = plan_from_query_state(state)
    assert plan is not None
    names = [n.name for n in plan.nodes]
    assert names == ["resume_search", "extract_employee_ids", "employee"]
    assert plan.nodes[0].params.get("purpose") == "status_resolve"


def test_location_status_plan_uses_resume_location() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state(
        "update status of employees which are living in Dubai to true",
        catalog=catalog,
    )
    assert state.intent == "set_status"
    assert state.status_value is True
    assert any(f.field == "city" and f.value == "Dubai" for f in state.filters)

    plan = plan_from_query_state(state)
    assert plan is not None
    assert [n.name for n in plan.nodes] == [
        "resume_search",
        "extract_employee_ids",
        "employee",
    ]
    # The Dubai cohort is resolved from resume text, so the write is RAG all the
    # way through and no SQL node ever sees the place.
    assert plan.nodes[0].params.get("purpose") == "location_cohort"
    assert plan.nodes[0].params.get("city") == "Dubai"
    assert "sql" not in [n.name for n in plan.nodes]


def test_set_status_does_not_collide_with_employment_status_facet() -> None:
    catalog = default_employee_catalog()
    state = extract_query_state("how many different statuses do we have?", catalog=catalog)
    assert state.intent in {"facet_count", "facet_list", "count"}
    assert state.intent != "set_status"


def test_current_employee_is_deictic_not_a_name() -> None:
    for q in (
        "change status of the current employee to true",
        "set the status of this employee to active",
        "update status of that one to false",
    ):
        req = extract_status_change(q)
        assert req.matched, q
        assert req.person_name is None, (q, req.person_name)
        assert req.status_value is not None, q


def test_current_employee_status_writes_active_referent() -> None:
    from uuid import uuid4

    from app.application.planning.heuristic_planner import elliptical_bound_person_plan
    from app.domain.enums import Role
    from app.domain.session import ActiveReferent, EntityRef, SessionMemory

    priya = uuid4()
    jack = uuid4()
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        active_referent=ActiveReferent(ids=[str(priya)], label="Priya Moreau"),
        last_listed=[
            EntityRef(employee_id=priya, display_name="Priya Moreau", aliases=["Priya"])
        ],
        entity_memory=[
            EntityRef(employee_id=priya, display_name="Priya Moreau", aliases=["Priya"]),
            EntityRef(employee_id=jack, display_name="Jack Smith", aliases=["Jack"]),
        ],
        # Stale pronoun binding must not win over the current focus.
        person_bindings={"he": str(jack), "him": str(jack), "his": str(jack)},
    )
    plan = elliptical_bound_person_plan(
        "change status of the current employee to true", memory=memory
    )
    assert plan is not None
    assert plan.clarify_question is None
    assert [n.name for n in plan.nodes] == ["employee"]
    assert plan.nodes[0].params.get("action") == "set_status"
    assert plan.nodes[0].params.get("employee_id") == str(priya)
    assert plan.nodes[0].params.get("status") is True


def test_set_status_without_name_binds_active_referent() -> None:
    from uuid import uuid4

    from app.domain.enums import Role
    from app.domain.session import ActiveReferent, EntityRef, SessionMemory

    eid = uuid4()
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        active_referent=ActiveReferent(ids=[str(eid)], label="Katya Okafor"),
        entity_memory=[
            EntityRef(employee_id=eid, display_name="Katya Okafor", aliases=["Katya"])
        ],
        person_bindings={"she": str(eid), "her": str(eid)},
    )
    catalog = default_employee_catalog()
    state = extract_query_state(
        "can you set the status to inactive please", catalog=catalog, memory=memory
    )
    assert state.intent == "set_status"
    assert state.status_value is False
    plan = plan_from_query_state(state, memory=memory)
    assert plan is not None
    assert plan.clarify_question is None
    assert [n.name for n in plan.nodes] == ["employee"]
    assert plan.nodes[0].params.get("employee_id") == str(eid)
    assert plan.nodes[0].params.get("status") is False


def test_looks_like_bare_person_name() -> None:
    from app.application.understanding.status_change import looks_like_bare_person_name

    assert looks_like_bare_person_name("Katya Andersen")
    assert looks_like_bare_person_name("Alice Nguyen")
    assert not looks_like_bare_person_name("set status to inactive")
    assert not looks_like_bare_person_name("how many employees")
    assert not looks_like_bare_person_name("Engineering")
