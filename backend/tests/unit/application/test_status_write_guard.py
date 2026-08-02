"""Status-write guard stays deterministic; novel phrasing uses tool-select HITL."""
from __future__ import annotations

from app.application.planning.heuristic_planner import try_status_write_plan


def test_seeded_status_phrase_compiles() -> None:
    plan = try_status_write_plan("change the status of Carol Garcia to true")
    assert plan is not None
    assert any(n.name == "employee" for n in plan.nodes) or any(
        n.name == "resume_search" for n in plan.nodes
    )


def test_novel_activate_misses_guard() -> None:
    assert try_status_write_plan("activate Carol Garcia") is None
