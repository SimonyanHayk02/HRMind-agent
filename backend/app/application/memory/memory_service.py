from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.application.memory.summarizer import Summarizer
from app.domain.auth import AuthContext
from app.domain.errors import ForbiddenError
from app.domain.session import ChatMessage, ConstraintRef, EntityRef, SessionMemory
from app.ports.session_store import SessionStore


class MemoryService:
    def __init__(
        self,
        store: SessionStore,
        summarizer: Summarizer | None = None,
        *,
        max_history: int = 20,
        summary_trigger: int = 12,
    ) -> None:
        self._store = store
        self._summarizer = summarizer
        self._max_history = max_history
        self._summary_trigger = summary_trigger

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

    async def append_user(self, session: SessionMemory, content: str) -> SessionMemory:
        session.messages.append(ChatMessage(role="user", content=content, created_at=datetime.now(UTC)))
        return await self._maybe_summarize(session)

    async def append_assistant(self, session: SessionMemory, content: str) -> SessionMemory:
        session.messages.append(ChatMessage(role="assistant", content=content, created_at=datetime.now(UTC)))
        session.updated_at = datetime.now(UTC)
        await self._store.save(session)
        return session

    async def upsert_entities(self, session: SessionMemory, entities: list[EntityRef]) -> SessionMemory:
        by_id = {e.employee_id: e for e in session.entity_memory}
        for e in entities:
            by_id[e.employee_id] = e
        session.entity_memory = list(by_id.values())
        session.updated_at = datetime.now(UTC)
        await self._store.save(session)
        return session

    async def merge_constraints(self, session: SessionMemory, constraints: list[ConstraintRef]) -> SessionMemory:
        """Last write wins per field (active filter stack)."""
        by_field = {c.field: c for c in session.constraint_memory}
        for c in constraints:
            by_field[c.field] = c
        session.constraint_memory = list(by_field.values())
        session.updated_at = datetime.now(UTC)
        await self._store.save(session)
        return session

    async def upsert_constraints(self, session: SessionMemory, constraints: list[ConstraintRef]) -> None:
        await self.merge_constraints(session, constraints)

    async def set_last_employee_ids(self, session: SessionMemory, employee_ids: list[str]) -> SessionMemory:
        seen: set[str] = set()
        ordered: list[str] = []
        for eid in employee_ids:
            s = str(eid)
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
        session.last_employee_ids = ordered
        session.updated_at = datetime.now(UTC)
        await self._store.save(session)
        return session

    async def _maybe_summarize(self, session: SessionMemory) -> SessionMemory:
        if len(session.messages) > self._summary_trigger and self._summarizer is not None:
            older = session.messages[: -self._max_history // 2]
            summary = await self._summarizer.summarize(session.summary, older)
            session.summary = summary
            session.messages = session.messages[-self._max_history // 2 :]
        session.updated_at = datetime.now(UTC)
        await self._store.save(session)
        return session
