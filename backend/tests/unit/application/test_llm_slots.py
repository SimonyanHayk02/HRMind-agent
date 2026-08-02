from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from app.adapters.llm.fake_llm import FakeLLM
from app.application.planning.plan_compiler import PlanCompiler
from app.application.understanding.plan_from_slots import plan_from_slots
from app.application.understanding.slot_bundle import PersonRef, SlotBundle
from app.application.understanding.turn_digest import build_turn_digest
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.session import EntityRef, SessionMemory
from app.domain.tools.registry import ToolRegistry

E1 = UUID("00000000-0000-0000-0000-000000000001")
E2 = UUID("00000000-0000-0000-0000-000000000002")
_GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "golden"
    / "nlu_slots_paraphrase.jsonl"
)


def _auth() -> AuthContext:
    return AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)


def _memory_listed() -> SessionMemory:
    return SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_listed=[
            EntityRef(employee_id=E1, display_name="Kara Petrov"),
            EntityRef(employee_id=E2, display_name="Maya Khan"),
        ],
        last_employee_ids=[str(E1), str(E2)],
        messages=[],
    )


def test_turn_digest_includes_listed_and_recent() -> None:
    mem = _memory_listed()
    mem.messages = []  # type: ignore[assignment]
    from app.domain.session import ChatMessage
    from datetime import UTC, datetime

    mem.messages = [
        ChatMessage(role="user", content="names please", created_at=datetime.now(UTC)),
        ChatMessage(
            role="assistant",
            content="- Kara Petrov\n- Maya Khan",
            created_at=datetime.now(UTC),
        ),
    ]
    digest = build_turn_digest("the top one's dob", mem)
    assert digest["question"] == "the top one's dob"
    assert digest["last_listed"][0]["display_name"] == "Kara Petrov"
    assert len(digest["recent_messages"]) == 2


def test_ordinal_slots_never_emit_sql_city_filter() -> None:
    bundle = SlotBundle(
        intent="location_cohort",
        attribute="location",
        city="Berlin",
        want_count=True,
        confidence=0.9,
    )
    result = plan_from_slots(bundle, memory=None, question="how many based in Berlin")
    assert result.handled and result.plan is not None
    for node in result.plan.nodes:
        if node.name == "sql":
            filters = (node.params or {}).get("filters") or {}
            assert "city" not in filters
            assert "country" not in filters
        if node.name == "resume_search":
            assert (node.params or {}).get("purpose") == "location_cohort"
            assert (node.params or {}).get("city") == "Berlin"


def test_top_one_dob_binds_last_listed() -> None:
    bundle = SlotBundle(
        intent="birthday",
        attribute="dob",
        person_ref=PersonRef(kind="ordinal", value="top", index=1),
        confidence=0.9,
    )
    result = plan_from_slots(
        bundle, memory=_memory_listed(), question="what's the top one's bday"
    )
    assert result.handled and result.plan is not None
    node = result.plan.nodes[0]
    assert node.name == "resume_search"
    assert node.params["purpose"] == "birthday_person"
    assert node.params["employee_ids"] == [str(E1)]


def test_their_dob_with_two_listed_clarifies() -> None:
    bundle = SlotBundle(
        intent="birthday",
        attribute="dob",
        person_ref=PersonRef(kind="pronoun", value="their"),
        confidence=0.9,
    )
    result = plan_from_slots(bundle, memory=_memory_listed(), question="their dob")
    assert result.handled
    assert result.plan is not None
    assert result.plan.nodes == []
    assert "which person" in (result.plan.clarify_question or "").lower()


def test_unsupported_pto() -> None:
    bundle = SlotBundle(intent="unsupported", confidence=0.9)
    result = plan_from_slots(bundle, memory=None, question="how much PTO")
    assert result.handled and result.plan is not None
    assert "PTO" in (result.plan.clarify_question or "") or "don't have" in (
        result.plan.clarify_question or ""
    ).lower()


def test_low_confidence_not_handled() -> None:
    bundle = SlotBundle(intent="count", want_count=True, confidence=0.2)
    result = plan_from_slots(bundle, memory=None)
    assert result.handled is False


def test_plan_compiler_heuristic_path_skips_slot_llm() -> None:
    """Happy-path regex must not invoke the slot extractor."""
    llm = FakeLLM()
    compiler = PlanCompiler(llm=llm, tools=ToolRegistry())

    async def _run() -> None:
        plan, mode, _meta = await compiler.compile(
            "How many employees work in Engineering?",
            auth=_auth(),
            memory=None,
        )
        assert mode in {
            "query_state",
            "guard_query_state",
            "heuristic",
            "query_state_fallback",
            "heuristic_fallback",
            "tool_select",
        }
        assert plan.nodes
        assert any(
            (n.params or {}).get("filters", {}).get("department") == "Engineering"
            or (n.params or {}).get("department") == "Engineering"
            for n in plan.nodes
        )
        # Residual slot extractor not used on this happy path.
        assert not any(
            "dialogue slots" in (c.get("system") or "").lower() for c in llm.calls
        )

    asyncio.run(_run())


def test_plan_compiler_top_one_binds_list_referent() -> None:
    llm = FakeLLM()
    compiler = PlanCompiler(llm=llm, tools=ToolRegistry())
    mem = _memory_listed()

    async def _run() -> None:
        # "top one" is deterministic list deixis — no residual LLM needed.
        plan, mode, _meta = await compiler.compile(
            "what's the top one's bday",
            auth=_auth(),
            memory=mem,
        )
        assert mode == "heuristic_list_referent"
        assert plan.nodes[0].params.get("purpose") == "birthday_person"
        assert plan.nodes[0].params.get("employee_ids") == [str(E1)]
        assert llm.calls == []

    asyncio.run(_run())


def test_fake_llm_slot_fixture_roundtrip() -> None:
    llm = FakeLLM()

    async def _run() -> None:
        raw = await llm.complete(
            system="Extract structured dialogue slots. person_ref required.",
            user=json.dumps(
                {
                    "question": "people based in Berlin",
                    "last_listed": [],
                    "recent_messages": [],
                }
            ),
            response_json=True,
        )
        data = json.loads(raw)
        assert data["city"] == "Berlin"
        assert data["intent"] == "location_cohort"

    asyncio.run(_run())


def _assert_no_sql_place_filters(plan) -> None:
    for node in plan.nodes:
        if node.name != "sql":
            continue
        filters = (node.params or {}).get("filters") or {}
        assert "city" not in filters
        assert "country" not in filters


def test_nlu_slots_paraphrase_goldens() -> None:
    """Hermetic paraphrase suite — FakeLLM + SlotBundle mapping, no live OpenAI."""
    assert _GOLDEN.is_file(), f"missing {_GOLDEN}"
    cases = [
        json.loads(line)
        for line in _GOLDEN.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(cases) >= 5

    for case in cases:
        mem = _memory_listed() if case.get("needs_last_listed") else None
        if "slots" in case:
            bundle = SlotBundle.model_validate(case["slots"])
            result = plan_from_slots(
                bundle, memory=mem, question=case.get("question") or ""
            )
            if case.get("expect_unhandled"):
                assert result.handled is False, case["id"]
                continue
            assert result.handled, case["id"]
            assert result.plan is not None, case["id"]
            if case.get("expect_clarify"):
                assert result.plan.nodes == []
                assert result.plan.clarify_question
                continue
            if case.get("expect_unsupported"):
                assert result.plan.nodes == []
                continue
            purpose = None
            for node in result.plan.nodes:
                purpose = (node.params or {}).get("purpose") or purpose
            if case.get("expected_purpose"):
                assert purpose == case["expected_purpose"], (
                    case["id"],
                    purpose,
                )
            if case.get("forbid_sql_fields"):
                _assert_no_sql_place_filters(result.plan)
            continue

        # End-to-end residual compile path (FakeLLM fixture).
        llm = FakeLLM()
        compiler = PlanCompiler(llm=llm, tools=ToolRegistry())

        async def _run(q: str = case["question"], m=mem):
            return await compiler.compile(q, auth=_auth(), memory=m)

        plan, mode, _meta = asyncio.run(_run())
        assert mode == case["expected_mode"], (case["id"], mode)
        if case.get("expected_purpose"):
            assert any(
                (n.params or {}).get("purpose") == case["expected_purpose"]
                for n in plan.nodes
            ), case["id"]
        if case.get("forbid_sql_fields"):
            _assert_no_sql_place_filters(plan)
        # Latency/cost guard: residual path is one LLM call; deterministic is zero.
        expected_calls = case.get("expected_llm_calls", 1)
        assert len(llm.calls) == expected_calls, (case["id"], len(llm.calls))
