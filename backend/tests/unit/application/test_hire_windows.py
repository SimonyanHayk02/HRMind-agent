from __future__ import annotations

from datetime import date

from app.adapters.sql.query_builder import QueryBuilder
from app.application.planning.heuristic_planner import (
    _hire_date_filters,
    try_heuristic_plan,
)


def test_hire_quarter_and_90_days() -> None:
    today = date(2026, 8, 1)
    q = _hire_date_filters("who joined this quarter?", today=today)
    assert q == {"hire_date_gte": "2026-07-01", "hire_date_lt": "2026-10-01"}

    q = _hire_date_filters("who joined in the last 90 days?", today=today)
    assert q is not None
    assert q["hire_date_gte"] == "2026-05-03"
    assert q["hire_date_lt"] == "2026-08-02"

    q = _hire_date_filters("who joined since March 2025?", today=today)
    assert q == {"hire_date_gte": "2025-03-01"}


def test_tenure_templates() -> None:
    sql, _ = QueryBuilder().build(
        columns=[],
        filters={"department": "Engineering"},
        template="agg_tenure",
    )
    assert "AVG(CURRENT_DATE - e.hire_date)" in sql
    assert "Engineering" in sql or ":department" in sql

    sql, _ = QueryBuilder().build(
        columns=[],
        filters={},
        template="longest_tenured",
        limit=3,
    )
    assert "ORDER BY e.hire_date ASC" in sql
    assert "LIMIT 3" in sql


def test_heuristic_tenure_plan() -> None:
    plan = try_heuristic_plan("what is the average tenure in Engineering?")
    assert plan is not None
    assert plan.nodes[0].params.get("template") == "agg_tenure"
    plan = try_heuristic_plan("who is the most senior in Engineering?")
    assert plan is not None
    assert plan.nodes[0].params.get("template") == "longest_tenured"
