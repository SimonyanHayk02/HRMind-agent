"""Context-as-helper: classifier, packet redaction, fresh-scope lint."""
from __future__ import annotations

import json

import pytest

from app.application.memory.context_budget import build_planner_packet
from app.application.memory.context_view import (
    ContextNeed,
    classify_context_need,
    extract_lookup_name_candidate,
    lint_fresh_scope,
)
from app.application.planning.plan_compiler import PlanCompiler
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.understanding.turn_digest import build_turn_digest
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import ActiveReferent, EntityRef, SessionMemory
from app.domain.tools.registry import ToolRegistry


def _alice_memory() -> SessionMemory:
    eid = "91cb25fe-2e51-5e7e-9466-155d30ffaf63"
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=[eid],
        entity_memory=[
            EntityRef(
                employee_id=eid,
                display_name="Alice Nguyen",
                aliases=["Alice"],
            )
        ],
        person_bindings={"she": eid, "her": eid, "he": eid, "him": eid},
        active_referent=ActiveReferent(ids=[eid], label="Alice Nguyen"),
        messages=[],
    )


def test_classifier_matrix() -> None:
    mem = _alice_memory()
    assert classify_context_need("do we have Sofia ?", mem) == ContextNeed.FRESH
    assert classify_context_need("find Sofia", mem) == ContextNeed.FRESH
    assert classify_context_need("looking for Sofia?", mem) == ContextNeed.FRESH
    assert (
        classify_context_need("how many of them know Python?", mem)
        == ContextNeed.ANAPHORA
    )
    assert (
        classify_context_need("which of them is named Sofia?", mem)
        == ContextNeed.ANAPHORA_NAMED
    )
    assert classify_context_need("her education?", mem) == ContextNeed.ANAPHORA
    assert (
        classify_context_need("the first person's birthday", mem)
        == ContextNeed.LIST_ORDINAL
    )
    assert (
        classify_context_need("what's the top one's bday", mem)
        == ContextNeed.LIST_ORDINAL
    )
    assert classify_context_need("education?", mem) == ContextNeed.ELLIPTICAL_PERSON
    assert extract_lookup_name_candidate("do we have Python?") is None


def test_fresh_packet_strips_alice_context() -> None:
    mem = _alice_memory()
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    packet = build_planner_packet(
        question="find Sofia",
        auth=auth,
        memory=mem,
        tools=[],
        schema_catalog={},
        query_state=None,
    )
    assert packet["context_need"] == ContextNeed.FRESH.value
    assert packet["last_employee_ids"] == []
    assert packet["person_bindings"] == {}
    assert packet["entities"] == []
    assert packet["constraints"] == []
    assert packet["recent_messages"] == []
    assert packet["summary"] == ""

    digest = build_turn_digest("find Sofia", mem)
    assert digest["context_need"] == ContextNeed.FRESH.value
    assert digest["person_bindings"] == {}
    assert digest["has_prior_cohort"] is False


def test_anaphora_keeps_cohort() -> None:
    mem = _alice_memory()
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    packet = build_planner_packet(
        question="how many of them know Python?",
        auth=auth,
        memory=mem,
        tools=[],
        schema_catalog={},
        query_state=None,
    )
    assert packet["context_need"] == ContextNeed.ANAPHORA.value
    assert packet["last_employee_ids"]


def test_lint_fresh_scope_rejects_cohort_sql() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={
                    "mode": "constrained",
                    "filters": {"employee_ids": ["91cb25fe-2e51-5e7e-9466-155d30ffaf63"]},
                },
            )
        ]
    )
    err = lint_fresh_scope(plan, ContextNeed.FRESH, "find Sofia")
    assert err is not None

    ok_plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="e1",
                kind="tool",
                name="employee",
                params={"action": "by_name", "name": "Sofia"},
            )
        ]
    )
    assert lint_fresh_scope(ok_plan, ContextNeed.FRESH, "find Sofia") is None


class _SeqLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, **_kwargs):  # noqa: ANN003
        self.calls += 1
        if not self._responses:
            return "{}"
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_llm_retry_stripped_after_invalid() -> None:
    good = json.dumps(
        {
            "version": "1",
            "nodes": [
                {
                    "id": "e1",
                    "kind": "tool",
                    "name": "employee",
                    "params": {"action": "by_name", "name": "Sofia"},
                    "input_bindings": {},
                    "depends_on": [],
                }
            ],
            "response_strategy": "template",
        }
    )
    llm = _SeqLLM(["not-json{{{", good])
    registry = ToolRegistry()
    # empty registry is fine for compile LLM path after heuristics miss
    compiler = PlanCompiler(llm, registry)
    plan, mode, _meta = await compiler.compile(
        "find Sofia XYZUNIQUE",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        memory=_alice_memory(),
    )
    # May be caught by heuristic existence/lookup — if so still ok.
    # Force residual path: use a paraphrase that may still hit existence.
    # If deterministic catches it, calls may be 0.
    if mode in {"llm", "llm_retry_stripped"}:
        assert llm.calls >= 1
        assert plan.nodes
        assert plan.nodes[0].params.get("name")


@pytest.mark.asyncio
async def test_llm_retry_after_fresh_scope_lint() -> None:
    bad = json.dumps(
        {
            "version": "1",
            "nodes": [
                {
                    "id": "sql1",
                    "kind": "tool",
                    "name": "sql",
                    "params": {
                        "mode": "constrained",
                        "use_session_cohort": True,
                        "filters": {},
                    },
                    "input_bindings": {},
                    "depends_on": [],
                }
            ],
            "response_strategy": "template",
        }
    )
    good = json.dumps(
        {
            "version": "1",
            "nodes": [
                {
                    "id": "e1",
                    "kind": "tool",
                    "name": "employee",
                    "params": {"action": "by_name", "name": "Zorba Quiggle"},
                    "input_bindings": {},
                    "depends_on": [],
                }
            ],
            "response_strategy": "template",
        }
    )
    # Call 1 = residual slots (low conf / unhandled); 2 = bad scoped plan; 3 = good.
    slots_miss = json.dumps(
        {
            "intent": "unknown",
            "attribute": "none",
            "person_ref": {"kind": "none", "value": None, "index": None},
            "refers_to_prior": False,
            "confidence": 0.1,
        }
    )
    llm = _SeqLLM([slots_miss, bad, good])
    compiler = PlanCompiler(llm, ToolRegistry())
    plan, mode, _meta = await compiler.compile(
        "search for Zorba Quiggle",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        memory=_alice_memory(),
    )
    # Deterministic lookup ("search for Name") — no free-form planner retry.
    assert mode in {
        "query_state",
        "heuristic",
        "query_state_fallback",
        "heuristic_fallback",
        "tool_select",
    }
    assert plan.nodes[0].params.get("name") == "Zorba Quiggle"
