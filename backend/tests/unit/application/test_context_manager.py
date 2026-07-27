"""Unit tests for ContextManager — referent lifecycle, budget packing, clear rules."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.adapters.persistence.memory_session_store import MemorySessionStore
from app.application.execution.graph_state import GraphState
from app.application.memory.context_budget import (
    build_planner_packet,
    compact_tool_payloads,
)
from app.application.memory.context_manager import ContextManager
from app.application.memory.memory_service import MemoryService
from app.application.planning.heuristic_planner import refers_to_prior_set
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import (
    ActiveReferent,
    EntityRef,
    LastFocus,
    SessionMemory,
    ToolFact,
)
from app.domain.tools.base import ToolResult


def _auth() -> AuthContext:
    return AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)


def _mgr() -> ContextManager:
    return ContextManager(MemoryService(MemorySessionStore()))


def test_anaphora_from_that_n_employees() -> None:
    assert refers_to_prior_set("and from that 100 employees how much knows python")
    assert refers_to_prior_set("of that group who knows Docker?")
    assert refers_to_prior_set("which of them knows Kubernetes?")
    assert not refers_to_prior_set("Who knows Python?")


def test_resolve_and_clear_on_greeting() -> None:
    async def _run() -> None:
        mgr = _mgr()
        session = await mgr.load(None, _auth())
        session = await mgr.memory_service.set_last_employee_ids(
            session, ["00000000-0000-0000-0000-000000000001"]
        )
        session = await mgr.memory_service.set_active_referent(
            session,
            ActiveReferent(
                ids=["00000000-0000-0000-0000-000000000001"],
                label="python developers",
            ),
        )
        session = await mgr.memory_service.set_last_focus(
            session, LastFocus(kind="cohort", dimension="employees")
        )

        session, working = await mgr.prepare_turn("hello", session)
        assert working.resolved.clear_referents is True
        assert session.last_employee_ids == []
        assert session.active_referent is None
        assert session.last_focus is None

    asyncio.run(_run())


def test_resolve_them_keeps_ids() -> None:
    async def _run() -> None:
        mgr = _mgr()
        session = await mgr.load(None, _auth())
        eid = "00000000-0000-0000-0000-000000000001"
        session = await mgr.memory_service.set_last_employee_ids(session, [eid])
        session = await mgr.memory_service.set_active_referent(
            session, ActiveReferent(ids=[eid], label="python developers")
        )
        session, working = await mgr.prepare_turn(
            "Which of them knows Docker?", session
        )
        assert working.resolved.refers_to_prior is True
        assert working.resolved.employee_ids == [eid]
        assert working.resume_retrieval_allowed is False

    asyncio.run(_run())


def test_commit_sets_named_set_and_person_bindings() -> None:
    async def _run() -> None:
        mgr = _mgr()
        session = await mgr.load(None, _auth())
        eid = "00000000-0000-0000-0000-000000000001"
        plan = ExecutionPlan(
            nodes=[
                PlanNode(id="r1", kind="tool", name="resume_search", params={}),
                PlanNode(id="ids", kind="operator", name="extract_employee_ids"),
            ],
            active_cohort_node="ids",
        )
        state = GraphState(
            question="Find Python developers",
            auth=_auth(),
            node_results={
                "r1": ToolResult(
                    data={
                        "employee_ids": [eid],
                        "hits": [
                            {
                                "employee_id": eid,
                                "employee_name": "Ada Lovelace",
                                "snippets": ["x" * 800],
                            }
                        ],
                    }
                ),
                "ids": [eid],
            },
        )
        session = await mgr.commit(
            session,
            question="Find Python developers",
            plan=plan,
            state=state,
        )
        assert session.last_employee_ids == [eid]
        assert session.active_referent is not None
        assert "python" in (session.active_referent.label or "")
        assert any(k.startswith("python") for k in session.named_sets)
        assert session.entity_memory
        assert "Ada" in session.entity_memory[0].aliases
        assert session.person_bindings.get("he") == eid
        assert session.tool_fact_cache.get("last_count") is None  # no count in payload

    asyncio.run(_run())


def test_planner_packet_caps_ids_and_summary() -> None:
    ids = [str(uuid4()) for _ in range(50)]
    memory = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=ids,
        summary="S" * 5000,
        active_referent=ActiveReferent(ids=ids, label="big set"),
    )
    packet = build_planner_packet(
        question="q",
        auth=_auth(),
        memory=memory,
        tools=[],
        schema_catalog={},
        query_state=None,
        max_ids=20,
        max_summary_chars=100,
    )
    assert len(packet["last_employee_ids"]) == 20
    assert packet["last_employee_ids_truncated"] is True
    assert len(packet["summary"]) <= 100
    assert packet["approx_tokens"] > 0


def test_compact_tool_payloads_truncates_snippets() -> None:
    payloads = [
        {
            "employee_ids": [str(uuid4()) for _ in range(40)],
            "hits": [
                {"employee_id": "1", "snippets": ["y" * 1000]} for _ in range(10)
            ],
        }
    ]
    packed = compact_tool_payloads(payloads, auth=_auth())
    assert packed[0]["employee_ids_truncated"] is True
    assert len(packed[0]["hits"]) <= 5
    assert len(packed[0]["hits"][0]["snippets"][0]) <= 400


def test_tool_fact_expiration() -> None:
    async def _run() -> None:
        mgr = _mgr()
        session = await mgr.load(None, _auth())
        session.tool_fact_cache["old"] = ToolFact(
            key="old",
            value=1,
            created_at=datetime.now(UTC) - timedelta(seconds=9999),
            ttl_seconds=10,
        )
        session.tool_fact_cache["fresh"] = ToolFact(
            key="fresh",
            value=2,
            created_at=datetime.now(UTC),
            ttl_seconds=180,
        )
        session, _ = await mgr.prepare_turn("Find Java engineers", session)
        assert "old" not in session.tool_fact_cache
        assert "fresh" in session.tool_fact_cache

    asyncio.run(_run())


def test_entity_resolver_from_refs() -> None:
    from app.domain.services.entity_resolver import EntityResolver

    eid = uuid4()
    refs = [
        EntityRef(
            employee_id=eid,
            display_name="Ada Lovelace",
            aliases=["Ada"],
        )
    ]
    resolver = EntityResolver()
    found, conf = resolver.resolve_from_refs("Ada", refs)
    assert found == eid
    assert conf > 0
