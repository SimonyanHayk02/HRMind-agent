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
    assert [n.name for n in plan.nodes] == ["resume_search", "extract_employee_ids"]
    assert plan.active_cohort_node == "ids"


def test_how_many_know_python_counts_via_resume_ids() -> None:
    plan = try_heuristic_plan("and from that 100 employees how much knows python")
    assert plan is not None
    # Without prior last_employee_ids this is a fresh skill count (not anaphora).
    assert [n.name for n in plan.nodes] == ["resume_search", "extract_employee_ids", "sql"]
    assert plan.nodes[-1].params["count_only"] is True
    assert plan.response_strategy == "template"


def test_aws_experience_uses_resume_search() -> None:
    plan = try_heuristic_plan("Show employees with AWS experience.")
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"


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


def test_of_them_in_berlin_filters_prior_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("how many of them in Berlin", memory=memory)
    assert plan is not None
    assert plan.active_cohort_node == "cohort"
    filters = plan.nodes[-1].params["filters"]
    assert filters["city"] == "Berlin"
    assert filters["employee_ids"] == ["00000000-0000-0000-0000-000000000001"]
    assert plan.nodes[-1].params["count_only"] is True


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


def test_where_ivy_chen_lives_uses_employee_lookup() -> None:
    plan = try_heuristic_plan("where the ivy chen lives?")
    assert plan is not None
    assert plan.nodes[0].name == "employee"
    assert plan.nodes[0].params["action"] == "by_name"
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


def test_countries_count_uses_facet_plan() -> None:
    plan = try_heuristic_plan("in how different countries do we have employees?")
    assert plan is not None
    assert [n.id for n in plan.nodes] == ["facet", "sql1"]
    assert plan.nodes[0].params["distinct"] is True
    assert plan.nodes[0].params["columns"] == ["country"]
    assert plan.nodes[1].params["count_distinct"] == "country"


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
    assert plan.nodes[0].params["distinct"] is True
    assert plan.nodes[0].params["columns"] == ["country"]


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


def test_list_engineering_uses_department_list_plan() -> None:
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

