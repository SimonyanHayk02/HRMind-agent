from app.application.planning.heuristic_planner import try_heuristic_plan


def test_engineering_count_plan() -> None:
    plan = try_heuristic_plan("How many employees work in Engineering?")
    assert plan is not None
    assert plan.nodes[0].name == "sql"
    assert plan.nodes[0].params["filters"]["department"] == "Engineering"
    assert plan.nodes[0].params["count_only"] is True
    assert plan.response_strategy == "template"
