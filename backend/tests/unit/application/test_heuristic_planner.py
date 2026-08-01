from app.application.planning.heuristic_planner import refers_to_prior_set, try_heuristic_plan
from app.domain.enums import Role
from app.domain.session import SessionMemory


def test_engineering_count_plan() -> None:
    plan = try_heuristic_plan("How many employees work in Engineering?")
    assert plan is not None
    assert [n.name for n in plan.nodes] == ["sql", "sql"]
    assert plan.active_cohort_node == "cohort"
    assert plan.nodes[0].params["filters"]["department"] == "Engineering"
    assert plan.nodes[0].params["count_only"] is False
    assert plan.nodes[1].params["count_only"] is True
    assert plan.response_strategy == "template"


def test_who_knows_python_uses_resume_search() -> None:
    plan = try_heuristic_plan("Who knows Python?")
    assert plan is not None
    assert [n.name for n in plan.nodes] == [
        "resume_search",
        "extract_employee_ids",
        "sql",
    ]
    # SQL name materialization keeps last_listed aligned with the answer order.
    assert plan.active_cohort_node == "sql1"
    assert plan.response_strategy == "template"


def test_how_many_know_python_counts_via_resume_ids() -> None:
    plan = try_heuristic_plan("and from that 100 employees how much knows python")
    assert plan is not None
    # Without prior last_employee_ids this is a fresh skill count (anaphora detected,
    # but no cohort to reuse yet).
    assert refers_to_prior_set("and from that 100 employees how much knows python")
    assert [n.name for n in plan.nodes] == ["resume_search", "extract_employee_ids", "sql"]
    assert plan.nodes[-1].params["count_only"] is True
    assert plan.response_strategy == "template"


def test_from_that_n_employees_reuses_prior_ids() -> None:
    prior = "00000000-0000-0000-0000-000000000001"
    memory = _memory_with_ids(prior)
    plan = try_heuristic_plan(
        "and from that 100 employees how much knows python", memory=memory
    )
    assert plan is not None
    # With a prior cohort, follow-up scopes resume + intersect to those IDs.
    assert any(n.name == "intersect_ids" for n in plan.nodes)
    resume = next(n for n in plan.nodes if n.name == "resume_search")
    assert prior in resume.params.get("employee_ids", [])
    assert plan.nodes[-1].params.get("count_only") is True


def test_aws_experience_uses_resume_search() -> None:
    plan = try_heuristic_plan("Show employees with AWS experience.")
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
    # SQL name materialization keeps last_listed aligned with the answer order.
    assert any(n.name == "sql" for n in plan.nodes)
    assert plan.active_cohort_node == "sql1"
    assert plan.response_strategy == "template"


def _memory_with_ids(*ids: str) -> SessionMemory:
    return SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=list(ids),
    )


def test_followup_names_uses_last_employee_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("please say there names", memory=memory)
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["filters"]["employee_ids"] == [
        "00000000-0000-0000-0000-000000000001"
    ]
    assert plan.nodes[0].params["count_only"] is False


def test_followup_names_without_memory_is_none() -> None:
    plan = try_heuristic_plan("please say there names")
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question is not None


def test_of_them_count_uses_prior_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002")
    plan = try_heuristic_plan("how many of them?", memory=memory)
    assert plan is not None
    assert plan.active_cohort_node == "cohort"
    assert plan.nodes[-1].params["count_only"] is True
    assert len(plan.nodes[-1].params["filters"]["employee_ids"]) == 2


def test_of_them_in_berlin_retrieves_the_place_and_counts_in_sql() -> None:
    """Berlin comes from the resumes; SQL only counts the ids retrieval resolved."""
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("how many of them in Berlin", memory=memory)
    assert plan is not None
    assert [n.name for n in plan.nodes] == [
        "resume_search",
        "extract_employee_ids",
        "intersect_ids",
        "sql",
    ]
    assert plan.nodes[0].params["purpose"] == "location_cohort"
    assert plan.nodes[0].params["city"] == "Berlin"
    assert plan.nodes[2].params["other"] == ["00000000-0000-0000-0000-000000000001"]
    assert plan.nodes[-1].params["count_only"] is True
    assert "city" not in plan.nodes[-1].params["filters"]
    assert plan.active_cohort_node == "locix"


def test_skill_of_them_intersects_prior_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("how many of them know python", memory=memory)
    assert plan is not None
    assert [n.name for n in plan.nodes] == [
        "resume_search",
        "extract_employee_ids",
        "intersect_ids",
        "sql",
    ]
    assert plan.nodes[0].params["employee_ids"] == ["00000000-0000-0000-0000-000000000001"]
    assert plan.nodes[2].params["other"] == ["00000000-0000-0000-0000-000000000001"]
    assert plan.active_cohort_node == "ix"


def test_skill_of_them_uses_constraint_memory_without_ids() -> None:
    from app.domain.session import ConstraintRef

    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        constraint_memory=[ConstraintRef(field="department", op="eq", value="Engineering")],
    )
    plan = try_heuristic_plan("how many of them know python", memory=memory)
    assert plan is not None
    assert [n.name for n in plan.nodes] == ["resume_search", "extract_employee_ids", "sql"]
    assert plan.nodes[-1].params["filters"]["department"] == "Engineering"
    assert plan.nodes[-1].params["count_only"] is True
    assert "employee_ids" not in plan.nodes[0].params


from app.application.planning.heuristic_planner import refers_to_prior_set, try_heuristic_plan
from app.domain.enums import Role
from app.domain.session import EntityRef, SessionMemory
from uuid import UUID


def test_where_ivy_chen_lives_retrieves_from_her_resume() -> None:
    plan = try_heuristic_plan("where the ivy chen lives?")
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[0].params["purpose"] == "location_person"
    assert "ivy chen" in plan.nodes[0].params["name"].lower()


def test_where_lives_uses_entity_memory_name() -> None:
    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        entity_memory=[
            EntityRef(
                employee_id=UUID("00000000-0000-0000-0000-000000000001"),
                display_name="Ivy Chen",
            )
        ],
    )
    plan = try_heuristic_plan("where does ivy live?", memory=memory)
    assert plan is not None
    assert plan.nodes[0].params["name"] == "Ivy Chen"

    plan = try_heuristic_plan("got it, so how much vacation are taking the developers?")
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question is not None
    assert "don't have that information" in plan.clarify_question.lower()

    plan = try_heuristic_plan("how much of them are developing in python ?")
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[-1].name == "sql"
    assert plan.nodes[-1].params["count_only"] is True


def test_countries_count_aggregates_the_corpus() -> None:
    """A place facet is counted from the resumes, with no SQL node at all."""
    plan = try_heuristic_plan("in how different countries do we have employees?")
    assert plan is not None
    assert [n.name for n in plan.nodes] == ["resume_search"]
    assert plan.nodes[0].params["purpose"] == "location_facet"
    assert plan.nodes[0].params["facet"] == "country"
    assert plan.nodes[0].params["facet_count"] is True


def test_department_count_still_uses_sql_facets() -> None:
    """Only resume-sourced dimensions move; employees columns are unaffected."""
    plan = try_heuristic_plan("how many different departments do we have?")
    assert plan is not None
    assert [n.id for n in plan.nodes] == ["facet", "sql1"]
    assert plan.nodes[1].params["count_distinct"] == "department"


def test_names_please_after_country_focus_lists_countries() -> None:
    from app.domain.session import LastFocus

    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_focus=LastFocus(
            kind="facet",
            dimension="country",
            values=["Germany", "USA", "UK", "UAE", "France"],
        ),
    )
    plan = try_heuristic_plan("names please", memory=memory)
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[0].params["facet"] == "country"
    assert plan.nodes[0].params["facet_count"] is False


def test_names_please_with_employee_ids_lists_people() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("names please", memory=memory)
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["filters"]["employee_ids"] == [
        "00000000-0000-0000-0000-000000000001"
    ]


def test_names_please_without_context_clarifies() -> None:
    plan = try_heuristic_plan("names please")
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question is not None
    assert "which names" in plan.clarify_question.lower()


def test_of_them_from_usa_without_ids_counts_country() -> None:
    """After org headcount we may have no last_employee_ids — still answer location."""
    plan = try_heuristic_plan("how much of them are from USA?")
    assert plan is not None
    assert plan.nodes[0].params["purpose"] == "location_cohort"
    assert plan.nodes[0].params["country"] == "USA"
    assert plan.nodes[-1].params["count_only"] is True
    assert plan.active_cohort_node == "locids"


def test_of_them_from_usa_with_prior_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("how many of them are from the United States?", memory=memory)
    assert plan is not None
    assert plan.nodes[0].params["country"] == "USA"
    assert plan.nodes[2].params["other"] == ["00000000-0000-0000-0000-000000000001"]

    plan = try_heuristic_plan("List employees in Engineering")
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["filters"]["department"] == "Engineering"
    assert plan.nodes[0].params["count_only"] is False
    assert plan.active_cohort_node == "sql1"


def test_manager_of_alice_passes_name() -> None:
    plan = try_heuristic_plan("Who is the manager of Alice Nguyen?")
    assert plan is not None
    assert plan.nodes[0].name == "employee"
    assert plan.nodes[0].params["action"] == "manager"
    assert plan.nodes[0].params["name"] == "Alice Nguyen"
    assert plan.response_strategy == "template"


def test_extract_manager_subject() -> None:
    from app.tools.employee.tool import extract_manager_subject

    assert extract_manager_subject("Who is the manager of Alice Nguyen?") == "Alice Nguyen"
    assert extract_manager_subject("Who manages Grace Mueller?") == "Grace Mueller"
    assert extract_manager_subject("Alice Nguyen's manager") == "Alice Nguyen"


def test_meta_count_uses_tool_fact_cache() -> None:
    from datetime import UTC, datetime

    from app.domain.session import ToolFact

    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        tool_fact_cache={
            "last_count": ToolFact(
                key="last_count",
                value=7,
                created_at=datetime.now(UTC),
            )
        },
    )
    plan = try_heuristic_plan("how many was that again?", memory=memory)
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question == "The answer is 7."


def test_pronoun_location_uses_person_binding() -> None:
    eid = UUID("00000000-0000-0000-0000-000000000099")
    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        person_bindings={"she": str(eid), "her": str(eid)},
    )
    plan = try_heuristic_plan("where does she live?", memory=memory)
    assert plan is not None
    # The pronoun already fixes who, so the place is read for that exact id
    # rather than by looking the name up again.
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[0].params["purpose"] == "location_person"
    assert plan.nodes[0].params["employee_ids"] == [str(eid)]


def test_pronoun_location_without_binding_clarifies() -> None:
    plan = try_heuristic_plan("where does she live?")
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question is not None
    assert "which employee" in plan.clarify_question.lower()


def test_pronoun_education_and_title_use_binding() -> None:
    eid = UUID("00000000-0000-0000-0000-000000000099")
    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        person_bindings={"she": str(eid), "her": str(eid)},
    )
    for q in (
        "what education does she have",
        "whats her job title?",
        "what is her email",
        "what department is she in",
    ):
        plan = try_heuristic_plan(q, memory=memory)
        assert plan is not None, q
        assert plan.nodes[0].name == "employee", q
        assert plan.nodes[0].params.get("action") == "by_id", q
        assert str(plan.nodes[0].params.get("employee_id")) == str(eid), q


def test_elliptical_education_uses_bound_person_not_resume_purpose() -> None:
    """Bare 'what is the education?' must hit employees.education, not education_person."""
    eid = UUID("00000000-0000-0000-0000-000000000099")
    memory = SessionMemory(
        session_id="s1",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        person_bindings={"he": str(eid), "she": str(eid), "him": str(eid), "her": str(eid)},
    )
    plan = try_heuristic_plan("what is the education?", memory=memory)
    assert plan is not None
    assert plan.nodes[0].name == "employee"
    assert plan.nodes[0].params.get("action") == "by_id"
    assert str(plan.nodes[0].params.get("employee_id")) == str(eid)
    assert not any(
        (n.params or {}).get("purpose") == "education_person" for n in plan.nodes
    )


def test_which_of_them_know_is_count() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("which of them know Kubernetes?", memory=memory)
    assert plan is not None
    assert any(n.name == "resume_search" for n in plan.nodes)
    assert plan.nodes[-1].params.get("count_only") is True
    assert plan.response_strategy == "template"


def test_employees_from_previous_search_lists_prior_ids() -> None:
    eid = "00000000-0000-0000-0000-000000000001"
    memory = _memory_with_ids(eid)
    plan = try_heuristic_plan("employees from a previous search", memory=memory)
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["filters"]["employee_ids"] == [eid]
    assert plan.nodes[0].params.get("count_only") is False


def test_employees_from_previous_search_without_ids_clarifies() -> None:
    plan = try_heuristic_plan("employees from a previous search")
    assert plan is not None
    assert plan.nodes == []
    assert plan.clarify_question
    assert "previous employee list" in plan.clarify_question.lower()


def test_who_joined_this_year_lists_hire_window() -> None:
    from datetime import date

    from app.application.planning.heuristic_planner import _hire_date_filters

    today = date(2026, 8, 1)
    filters = _hire_date_filters("Who joined this year?", today=today)
    assert filters == {
        "hire_date_gte": "2026-01-01",
        "hire_date_lt": "2027-01-01",
    }
    plan = try_heuristic_plan("Who joined this year?")
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["count_only"] is False
    assert "hire_date_gte" in plan.nodes[0].params["filters"]
    assert "hire_date_lt" in plan.nodes[0].params["filters"]


def test_closest_birthday_from_there_scopes_prior_ids() -> None:
    from app.application.planning.heuristic_planner import refers_to_prior_set
    from app.application.understanding.birthday import extract_birthday

    q = "give from there which ones birthday is the closest one"
    assert extract_birthday(q).scope == "closest"
    assert refers_to_prior_set(q)
    ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    plan = try_heuristic_plan(q, memory=_memory_with_ids(*ids))
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[0].params["purpose"] == "birthday_cohort"
    assert plan.nodes[0].params["scope"] == "closest"
    assert plan.nodes[0].params["employee_ids"] == ids


def test_closest_from_previous_list_without_birthday_word() -> None:
    from app.application.understanding.birthday import extract_birthday

    q = "which ones from the previous list is the closest one"
    assert extract_birthday(q).matched
    assert extract_birthday(q).scope == "closest"
    ids = ["00000000-0000-0000-0000-000000000001"]
    plan = try_heuristic_plan(q, memory=_memory_with_ids(*ids))
    assert plan is not None
    assert plan.nodes[0].params.get("employee_ids") == ids


def test_how_many_hired_last_year_counts() -> None:
    from datetime import date

    from app.application.planning.heuristic_planner import _hire_date_filters

    filters = _hire_date_filters("How many were hired last year?", today=date(2026, 8, 1))
    assert filters == {
        "hire_date_gte": "2025-01-01",
        "hire_date_lt": "2026-01-01",
    }
    plan = try_heuristic_plan("How many were hired last year?")
    assert plan is not None
    assert any(n.params.get("count_only") for n in plan.nodes)

