from app.application.planning.heuristic_planner import try_heuristic_plan


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
    assert [n.name for n in plan.nodes] == ["resume_search", "extract_employee_ids", "sql"]
    assert plan.nodes[-1].params["count_only"] is True
    assert plan.response_strategy == "template"


def test_aws_experience_uses_resume_search() -> None:
    plan = try_heuristic_plan("Show employees with AWS experience.")
    assert plan is not None
    assert plan.nodes[0].name == "resume_search"
