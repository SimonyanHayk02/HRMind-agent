#!/usr/bin/env python3
"""Run golden eval cases against rule/plan heuristics (no live API)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import UUID

from app.application.memory.context_view import _ELLIPTICAL_ATTR_RE
from app.application.planning.heuristic_planner import (
    _PRONOUN_ONLY_RE,
    try_heuristic_plan,
)
from app.application.planning.plan_compiler import PlanCompiler
from app.application.routing.rule_router import RuleRouter
from app.domain.auth import AuthContext
from app.domain.enums import Role, RouterLabel
from app.domain.session import EntityRef, SessionMemory
from app.domain.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "golden" / "cases.jsonl"
ORCH = ROOT / "data" / "golden" / "orchestration.jsonl"
E1 = UUID("00000000-0000-0000-0000-000000000001")
E2 = UUID("00000000-0000-0000-0000-000000000002")


def _two_entity_memory() -> SessionMemory:
    entities = [
        EntityRef(employee_id=E1, display_name="Kara Petrov"),
        EntityRef(employee_id=E2, display_name="Maya Khan"),
    ]
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        entity_memory=list(entities),
        last_listed=list(entities),
        last_employee_ids=[str(E1), str(E2)],
    )


def eval_router_cases() -> tuple[int, int]:
    router = RuleRouter()
    hits = total = 0
    for line in CASES.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        total += 1
        expected = case.get("expected_route")
        got = router.route(case["question"])
        if expected == "greeting":
            ok = got == RouterLabel.GREETING
        else:
            ok = got is None
        hits += int(ok)
        print(("PASS" if ok else "FAIL"), "route", case["id"], case["question"][:60], "->", got)
    return hits, total


def _check_orch(case: dict) -> bool:
    q = case["question"]
    check = case["check"]
    if check == "skill_location_intersect":
        plan = try_heuristic_plan(q)
        return bool(plan and any(n.id == "skloc" for n in plan.nodes))
    if check == "pronoun_two_entities_clarify":
        plan = try_heuristic_plan(q, memory=_two_entity_memory())
        return bool(
            plan
            and not plan.nodes
            and plan.clarify_question
            and "which" in plan.clarify_question.lower()
        )
    if check == "ordinal_birthday_resume":
        import asyncio

        compiler = PlanCompiler(llm=None, tools=ToolRegistry())

        async def _run() -> bool:
            plan, mode, _meta = await compiler.compile(
                q,
                auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
                memory=_two_entity_memory(),
            )
            return (
                mode == "heuristic_list_referent"
                and plan.nodes
                and plan.nodes[0].name == "resume_search"
                and plan.nodes[0].params.get("purpose") == "birthday_person"
            )

        return bool(asyncio.run(_run()))
    if check == "elliptical_attr_pattern":
        return bool(_ELLIPTICAL_ATTR_RE.search(q) or _PRONOUN_ONLY_RE.search(q))
    if check == "no_nl2sql":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and not any(
                n.name == "sql" and (n.params or {}).get("mode") == "nl2sql"
                for n in plan.nodes
            )
        )
    if check == "location_cohort":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any(
                n.name == "resume_search"
                and (n.params or {}).get("purpose") == "location_cohort"
                for n in plan.nodes
            )
        )
    if check == "skill_rag":
        plan = try_heuristic_plan(q)
        return bool(plan and any(n.name == "resume_search" for n in plan.nodes))
    if check == "languages_cohort":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any(
                (n.params or {}).get("purpose") == "languages_cohort" for n in plan.nodes
            )
        )
    if check == "certifications_cohort":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any(
                (n.params or {}).get("purpose") == "certifications_cohort"
                for n in plan.nodes
            )
        )
    if check == "hire_window":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any(
                n.name == "sql"
                and "hire_date_gte" in ((n.params or {}).get("filters") or {})
                for n in plan.nodes
            )
        )
    if check == "tenure_template":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any(
                (n.params or {}).get("template") in {"agg_tenure", "agg_tenure_by_dept"}
                for n in plan.nodes
            )
        )
    if check == "reports_action":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any((n.params or {}).get("action") == "reports" for n in plan.nodes)
        )
    if check == "reports_skill_place":
        plan = try_heuristic_plan(q)
        return bool(
            plan
            and any((n.params or {}).get("action") == "reports" for n in plan.nodes)
            and any(n.name == "resume_search" for n in plan.nodes)
            and any(n.name == "intersect_ids" for n in plan.nodes)
        )
    if check == "unsupported_hris":
        plan = try_heuristic_plan(q)
        ans = (plan.clarify_question or "").lower() if plan else ""
        return bool(
            plan
            and not plan.nodes
            and plan.clarify_question
            and ("pto" in ans or "leave" in ans)
        )
    if check == "greeting_route":
        return RuleRouter().route(q) == RouterLabel.GREETING
    return False


def eval_orchestration_cases() -> tuple[int, int]:
    hits = total = 0
    for line in ORCH.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        total += 1
        ok = _check_orch(case)
        hits += int(ok)
        print(
            ("PASS" if ok else "FAIL"),
            "orch",
            case["id"],
            case["check"],
            case["question"][:50],
        )
    return hits, total


def main() -> int:
    r_hits, r_total = eval_router_cases()
    o_hits, o_total = eval_orchestration_cases()
    print(f"router_accuracy={r_hits}/{r_total}")
    print(f"orchestration_accuracy={o_hits}/{o_total}")
    ok = r_hits == r_total and o_hits == o_total
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
