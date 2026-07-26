from __future__ import annotations

import asyncio

from app.domain.session import SessionMemory


class MemorySessionStore:
    def __init__(self) -> None:
        self._store: dict[str, SessionMemory] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> SessionMemory | None:
        async with self._lock:
            return self._store.get(session_id)

    async def save(self, session: SessionMemory) -> None:
        async with self._lock:
            self._store[session.session_id] = session.model_copy(deep=True)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)
