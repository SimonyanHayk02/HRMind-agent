from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.application.memory.context_budget import DEFAULT_MAX_ENTITY_MEMORY
from app.application.memory.summarizer import Summarizer
from app.domain.auth import AuthContext
from app.domain.errors import ForbiddenError
from app.domain.session import (
    ActiveReferent,
    ChatMessage,
    ConstraintRef,
    EntityRef,
    LastFocus,
    SessionMemory,
    ToolFact,
)
from app.ports.session_store import SessionStore


class MemoryService:
    def __init__(
        self,
        store: SessionStore,
        summarizer: Summarizer | None = None,
        *,
        max_history: int = 20,
        summary_trigger: int = 12,
        max_entity_memory: int = DEFAULT_MAX_ENTITY_MEMORY,
        max_named_sets: int = 8,
    ) -> None:
        self._store = store
        self._summarizer = summarizer
        self._max_history = max_history
        self._summary_trigger = summary_trigger
        self._max_entity_memory = max_entity_memory
        self._max_named_sets = max_named_sets

    async def get_or_create(self, session_id: str | None, auth: AuthContext) -> SessionMemory:
        sid = session_id or str(uuid4())
        existing = await self._store.get(sid)
        if existing is not None:
            if existing.role != auth.role or existing.user_id != auth.user_id:
                raise ForbiddenError("Session belongs to a different principal/role")
            if existing.tenant_id != auth.tenant_id:
                raise ForbiddenError("Session belongs to a different tenant")
            return existing
        session = SessionMemory(
            session_id=sid,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            role=auth.role,
            updated_at=datetime.now(UTC),
        )
        await self._store.save(session)
        return session

    async def save(self, session: SessionMemory) -> SessionMemory:
        session.updated_at = datetime.now(UTC)
        await self._store.save(session)
        return session

    async def append_user(self, session: SessionMemory, content: str) -> SessionMemory:
        session.messages.append(
            ChatMessage(role="user", content=content, created_at=datetime.now(UTC))
        )
        return await self._maybe_summarize(session)

    async def append_assistant(self, session: SessionMemory, content: str) -> SessionMemory:
        session.messages.append(
            ChatMessage(role="assistant", content=content, created_at=datetime.now(UTC))
        )
        return await self.save(session)

    async def upsert_entities(
        self,
        session: SessionMemory,
        entities: list[EntityRef],
        *,
        max_entities: int | None = None,
    ) -> SessionMemory:
        by_id = {e.employee_id: e for e in session.entity_memory}
        for e in entities:
            existing = by_id.get(e.employee_id)
            if existing:
                aliases = list({*existing.aliases, *e.aliases})
                by_id[e.employee_id] = EntityRef(
                    employee_id=e.employee_id,
                    display_name=e.display_name or existing.display_name,
                    confidence=max(existing.confidence, e.confidence),
                    aliases=aliases,
                )
            else:
                by_id[e.employee_id] = e
        cap = max_entities or self._max_entity_memory
        # Keep most recently upserted entities (dict preserves insertion; re-insert moves)
        session.entity_memory = list(by_id.values())[-cap:]
        return await self.save(session)

    async def merge_constraints(
        self, session: SessionMemory, constraints: list[ConstraintRef]
    ) -> SessionMemory:
        """Last write wins per field (active filter stack)."""
        by_field = {c.field: c for c in session.constraint_memory}
        for c in constraints:
            by_field[c.field] = c
        session.constraint_memory = list(by_field.values())
        return await self.save(session)

    async def upsert_constraints(
        self, session: SessionMemory, constraints: list[ConstraintRef]
    ) -> None:
        await self.merge_constraints(session, constraints)

    async def drop_constraint_fields(
        self, session: SessionMemory, fields: set[str]
    ) -> SessionMemory:
        session.constraint_memory = [
            c for c in session.constraint_memory if c.field not in fields
        ]
        return await self.save(session)

    async def set_last_employee_ids(
        self, session: SessionMemory, employee_ids: list[str]
    ) -> SessionMemory:
        seen: set[str] = set()
        ordered: list[str] = []
        for eid in employee_ids:
            s = str(eid)
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
        session.last_employee_ids = ordered
        return await self.save(session)

    async def set_last_focus(self, session: SessionMemory, focus: LastFocus) -> SessionMemory:
        session.last_focus = focus
        return await self.save(session)

    async def set_active_referent(
        self, session: SessionMemory, referent: ActiveReferent | None
    ) -> SessionMemory:
        session.active_referent = referent
        return await self.save(session)

    async def set_named_set(
        self, session: SessionMemory, key: str, ids: list[str]
    ) -> SessionMemory:
        named = dict(session.named_sets)
        named[key] = list(ids)
        # Bound number of named sets (drop oldest keys)
        if len(named) > self._max_named_sets:
            overflow = list(named.keys())[: -self._max_named_sets]
            for k in overflow:
                named.pop(k, None)
        session.named_sets = named
        return await self.save(session)

    async def set_person_bindings(
        self, session: SessionMemory, bindings: dict[str, str]
    ) -> SessionMemory:
        session.person_bindings = {**session.person_bindings, **bindings}
        return await self.save(session)

    async def put_tool_fact(self, session: SessionMemory, fact: ToolFact) -> SessionMemory:
        cache = dict(session.tool_fact_cache)
        cache[fact.key] = fact
        # Cap cache size
        if len(cache) > 20:
            keys = list(cache.keys())[:-20]
            for k in keys:
                cache.pop(k, None)
        session.tool_fact_cache = cache
        return await self.save(session)

    async def clear_referents(self, session: SessionMemory) -> SessionMemory:
        session.last_employee_ids = []
        session.active_referent = None
        session.last_focus = None
        session.person_bindings = {}
        # Keep named_sets and entity_memory (long-lived within session) but drop universe
        session.constraint_memory = [
            c for c in session.constraint_memory if c.field != "_universe"
        ]
        return await self.save(session)

    async def _maybe_summarize(self, session: SessionMemory) -> SessionMemory:
        if len(session.messages) > self._summary_trigger and self._summarizer is not None:
            older = session.messages[: -self._max_history // 2]
            summary = await self._summarizer.summarize(session.summary, older)
            session.summary = summary
            session.messages = session.messages[-self._max_history // 2 :]
        return await self.save(session)
