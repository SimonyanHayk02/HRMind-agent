"""Cohort status reads use SQL over prior ids — not resume status_resolve."""
from __future__ import annotations

from uuid import UUID, uuid4

from app.application.planning.heuristic_planner import try_cohort_status_read_plan
from app.domain.enums import Role
from app.domain.session import EntityRef, SessionMemory


def _mem(ids: list[UUID]) -> SessionMemory:
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=[str(i) for i in ids],
        last_listed=[
            EntityRef(employee_id=i, display_name=f"Person {n}")
            for n, i in enumerate(ids, start=1)
        ],
    )


def test_statuses_of_them_compiles_sql() -> None:
    ids = [uuid4(), uuid4()]
    plan = try_cohort_status_read_plan("what are the statuses of them?", memory=_mem(ids))
    assert plan is not None
    assert any(n.name == "sql" for n in plan.nodes)
    sql = next(n for n in plan.nodes if n.name == "sql")
    cols = (sql.params or {}).get("columns") or []
    assert "status" in cols
    assert "employment_status" in cols
    assert (sql.params or {}).get("filters", {}).get("employee_ids") == [
        str(i) for i in ids
    ]


def test_give_there_statuses_typo() -> None:
    ids = [uuid4()]
    plan = try_cohort_status_read_plan("give there statuses", memory=_mem(ids))
    assert plan is not None
    assert any(n.name == "sql" for n in plan.nodes)


def test_her_status_not_cohort() -> None:
    ids = [uuid4(), uuid4()]
    assert try_cohort_status_read_plan("what is her status?", memory=_mem(ids)) is None
