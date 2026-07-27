"""Central owner of session working memory for HRMind turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.application.execution.graph_state import GraphState
from app.application.memory.cohort import should_update_last_employee_ids
from app.application.memory.context_budget import (
    DEFAULT_MAX_ENTITY_MEMORY,
    DEFAULT_MAX_IDS_IN_PACKET,
    build_planner_packet,
    compact_tool_payloads,
)
from app.application.memory.context_updates import (
    extract_entities_from_state,
    extract_last_focus,
    infer_constraints_from_question,
)
from app.application.memory.memory_service import MemoryService
from app.application.memory.result_ids import extract_cohort_ids
from app.application.planning.heuristic_planner import refers_to_prior_set
from app.application.planning.plan_schema import ExecutionPlan
from app.application.understanding.merge_query_state import apply_universe_marker
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.query_state import QueryState
from app.domain.session import (
    ActiveReferent,
    EntityRef,
    LastFocus,
    SessionMemory,
    ToolFact,
)

logger = get_logger(__name__)

_TOPIC_SHIFT_RE = re.compile(
    r"\b("
    r"new (?:topic|search|question)|"
    r"start over|never ?mind|forget (?:that|them|those)|"
    r"different (?:people|employees|set)|"
    r"instead find|now find|switch to"
    r")\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good (?:morning|afternoon|evening)|thanks|thank you)\b",
    re.IGNORECASE,
)
_PRONOUN_PERSON_RE = re.compile(r"\b(he|she|him|her|his|hers)\b", re.IGNORECASE)


@dataclass
class ResolvedRefs:
    refers_to_prior: bool
    employee_ids: list[str] = field(default_factory=list)
    person_id: str | None = None
    person_name: str | None = None
    clear_referents: bool = False
    topic_shift: bool = False


@dataclass
class WorkingContext:
    session: SessionMemory
    resolved: ResolvedRefs
    resume_retrieval_allowed: bool = True


@dataclass
class TurnTrace:
    trace_id: str
    session_id: str
    router_label: str | None = None
    planner_mode: str | None = None
    refers_to_prior: bool = False
    referent_before: int = 0
    referent_after: int = 0
    tools_called: list[str] = field(default_factory=list)
    prompt_approx_tokens: int = 0
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "router_label": self.router_label,
            "planner_mode": self.planner_mode,
            "refers_to_prior": self.refers_to_prior,
            "referent_before": self.referent_before,
            "referent_after": self.referent_after,
            "tools_called": self.tools_called,
            "prompt_approx_tokens": self.prompt_approx_tokens,
            "degraded": self.degraded,
            "notes": self.notes,
        }


class ContextManager:
    """Single owner for load → resolve → pack → commit of session context."""

    def __init__(
        self,
        memory: MemoryService,
        *,
        max_ids_in_packet: int = DEFAULT_MAX_IDS_IN_PACKET,
        max_entity_memory: int = DEFAULT_MAX_ENTITY_MEMORY,
        tool_fact_ttl_seconds: int = 180,
        max_context: int = 8,
        max_summary_chars: int = 2000,
    ) -> None:
        self._memory = memory
        self._max_ids = max_ids_in_packet
        self._max_entities = max_entity_memory
        self._tool_fact_ttl = tool_fact_ttl_seconds
        self._max_context = max_context
        self._max_summary_chars = max_summary_chars

    @property
    def memory_service(self) -> MemoryService:
        return self._memory

    async def load(self, session_id: str | None, auth: AuthContext) -> SessionMemory:
        return await self._memory.get_or_create(session_id, auth)

    async def append_user(self, session: SessionMemory, content: str) -> SessionMemory:
        return await self._memory.append_user(session, content)

    async def append_assistant(
        self, session: SessionMemory, content: str
    ) -> SessionMemory:
        return await self._memory.append_assistant(session, content)

    def resolve_references(self, question: str, session: SessionMemory) -> ResolvedRefs:
        refers = refers_to_prior_set(question)
        topic_shift = bool(_TOPIC_SHIFT_RE.search(question))
        greeting = bool(_GREETING_RE.search(question.strip()))
        clear = greeting or topic_shift

        ids = list(
            (session.active_referent.ids if session.active_referent else None)
            or session.last_employee_ids
        )
        person_id = None
        person_name = None

        if _PRONOUN_PERSON_RE.search(question) and session.person_bindings:
            for key in ("he", "she", "him", "her", "his", "hers"):
                if key in session.person_bindings and re.search(
                    rf"\b{key}\b", question, re.IGNORECASE
                ):
                    person_id = session.person_bindings[key]
                    break

        if session.entity_memory:
            q_lower = question.lower()
            for ent in sorted(
                session.entity_memory, key=lambda e: len(e.display_name), reverse=True
            ):
                names = [ent.display_name.lower(), *[a.lower() for a in ent.aliases]]
                if any(n and n in q_lower for n in names):
                    person_id = str(ent.employee_id)
                    person_name = ent.display_name
                    break

        return ResolvedRefs(
            refers_to_prior=refers,
            employee_ids=ids if refers else [],
            person_id=person_id,
            person_name=person_name,
            clear_referents=clear,
            topic_shift=topic_shift,
        )

    async def prepare_turn(
        self, question: str, session: SessionMemory
    ) -> tuple[SessionMemory, WorkingContext]:
        resolved = self.resolve_references(question, session)
        if resolved.clear_referents:
            session = await self.clear_referents(session, reason="topic_shift_or_greeting")
            resolved.employee_ids = []
            resolved.refers_to_prior = False

        session = self._expire_tool_facts(session)
        working = WorkingContext(
            session=session,
            resolved=resolved,
            resume_retrieval_allowed=not (
                resolved.refers_to_prior and bool(resolved.employee_ids)
            ),
        )
        return session, working

    def build_for_planner(
        self,
        *,
        question: str,
        auth: AuthContext,
        memory: SessionMemory,
        tools: list[dict],
        schema_catalog: dict,
        query_state: QueryState | None = None,
    ) -> dict[str, Any]:
        return build_planner_packet(
            question=question,
            auth=auth,
            memory=memory,
            tools=tools,
            schema_catalog=schema_catalog,
            query_state=query_state.model_dump(mode="json") if query_state else None,
            max_context=self._max_context,
            max_summary_chars=self._max_summary_chars,
            max_ids=self._max_ids,
        )

    def build_for_responder(
        self, payloads: list[Any], *, auth: AuthContext
    ) -> list[Any]:
        return compact_tool_payloads(payloads, auth=auth)

    async def merge_query_constraints(
        self, session: SessionMemory, query_state: QueryState
    ) -> SessionMemory:
        constraints = apply_universe_marker(session, query_state)
        if constraints is not None:
            # apply_universe_marker may mutate constraint_memory when dropping universe
            session = await self._memory.merge_constraints(session, constraints)
        return session

    async def commit(
        self,
        session: SessionMemory,
        *,
        question: str,
        plan: ExecutionPlan,
        state: GraphState,
        set_label: str | None = None,
    ) -> SessionMemory:
        ids = extract_cohort_ids(state, plan)
        saved_ids = (
            ids if ids and should_update_last_employee_ids(plan, question, ids) else []
        )

        if saved_ids:
            session = await self._memory.set_last_employee_ids(session, saved_ids)
            label = set_label or _infer_set_label(question)
            session = await self._memory.set_active_referent(
                session,
                ActiveReferent(
                    ids=saved_ids,
                    label=label,
                    source_turn=len(session.messages),
                    confidence=0.9,
                ),
            )
            label_key = _slug_label(label)
            if label_key:
                session = await self._memory.set_named_set(session, label_key, saved_ids)
        elif plan.nodes and all(n.name == "greeting" for n in plan.nodes):
            session = await self.clear_referents(session, reason="greeting_plan")

        entities = extract_entities_from_state(state)
        if entities:
            enriched: list[EntityRef] = []
            for e in entities:
                aliases = list(e.aliases)
                first = e.display_name.split()[0] if e.display_name else ""
                if first and first.lower() not in {a.lower() for a in aliases}:
                    aliases.append(first)
                enriched.append(
                    EntityRef(
                        employee_id=e.employee_id,
                        display_name=e.display_name,
                        confidence=e.confidence,
                        aliases=aliases,
                    )
                )
            session = await self._memory.upsert_entities(
                session, enriched, max_entities=self._max_entities
            )
            if len(enriched) == 1:
                eid = str(enriched[0].employee_id)
                session = await self._memory.set_person_bindings(
                    session,
                    {
                        "he": eid,
                        "she": eid,
                        "him": eid,
                        "her": eid,
                        "his": eid,
                        "hers": eid,
                    },
                )

        constraints = infer_constraints_from_question(question)
        if constraints:
            session = await self._memory.merge_constraints(session, constraints)

        focus = extract_last_focus(state, plan, employee_ids=saved_ids or None)
        if focus:
            session = await self._memory.set_last_focus(session, focus)
        elif saved_ids:
            session = await self._memory.set_last_focus(
                session, LastFocus(kind="cohort", dimension="employees", values=[])
            )

        for result in state.node_results.values():
            data = getattr(result, "data", result)
            if isinstance(data, dict) and data.get("count") is not None:
                session = await self._memory.put_tool_fact(
                    session,
                    ToolFact(
                        key="last_count",
                        value=int(data["count"]),
                        created_at=datetime.now(UTC),
                        ttl_seconds=self._tool_fact_ttl,
                    ),
                )
                break

        return session

    async def clear_referents(
        self, session: SessionMemory, *, reason: str
    ) -> SessionMemory:
        logger.info(
            "context_clear_referents",
            session_id=session.session_id,
            reason=reason,
        )
        return await self._memory.clear_referents(session)

    def _expire_tool_facts(self, session: SessionMemory) -> SessionMemory:
        now = datetime.now(UTC)
        keep: dict[str, ToolFact] = {}
        for key, fact in session.tool_fact_cache.items():
            age = (now - fact.created_at).total_seconds()
            if age <= fact.ttl_seconds:
                keep[key] = fact
        session.tool_fact_cache = keep
        return session


def _infer_set_label(question: str) -> str | None:
    m = re.search(
        r"\b(python|java|go|react|kubernetes|aws|docker)\b.{0,40}\b"
        r"(developers?|engineers?|employees?)",
        question,
        re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)} {m.group(2)}".lower()
    m2 = re.search(r"\b([\w+#.+]+)\s+(developers?|engineers?)\b", question, re.IGNORECASE)
    if m2:
        return f"{m2.group(1)} {m2.group(2)}".lower()
    return None


def _slug_label(label: str | None) -> str | None:
    if not label:
        return None
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:64]
