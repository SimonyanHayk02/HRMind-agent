from app.application.planning.heuristic_planner import refers_to_prior_set, try_heuristic_plan
from app.domain.enums import Role
from app.domain.session import SessionMemory


def test_engineering_count_plan() -> None:
    plan = try_heuristic_plan("How many employees work in Engineering?")
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["filters"]["department"] == "Engineering"
    assert plan.nodes[0].params["count_only"] is True
    assert plan.response_strategy == "template"


def test_who_knows_python_uses_resume_search() -> None:
    plan = try_heuristic_plan("Who knows Python?")
    assert plan is not None
    assert [n.name for n in plan.nodes] == ["resume_search"]


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
    assert try_heuristic_plan("please say there names") is None


def test_of_them_count_uses_prior_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002")
    plan = try_heuristic_plan("how many of them?", memory=memory)
    assert plan is not None
    assert plan.nodes[0].params["count_only"] is True
    assert len(plan.nodes[0].params["filters"]["employee_ids"]) == 2


def test_of_them_in_berlin_filters_prior_ids() -> None:
    memory = _memory_with_ids("00000000-0000-0000-0000-000000000001")
    plan = try_heuristic_plan("how many of them in Berlin", memory=memory)
    assert plan is not None
    filters = plan.nodes[0].params["filters"]
    assert filters["city"] == "Berlin"
    assert filters["employee_ids"] == ["00000000-0000-0000-0000-000000000001"]
    assert plan.nodes[0].params["count_only"] is True


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
    assert plan.nodes[2].params["other"] == ["00000000-0000-0000-0000-000000000001"]


def test_developing_in_python_uses_resume_count() -> None:
    plan = try_heuristic_plan("how much of them are developing in python ?")
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
    assert plan.nodes[-1].name == "sql"
    assert plan.nodes[-1].params["count_only"] is True


def test_refers_to_prior_set() -> None:
    assert refers_to_prior_set("how many of them?")
    assert refers_to_prior_set("please say their names")
    assert not refers_to_prior_set("how are you")

