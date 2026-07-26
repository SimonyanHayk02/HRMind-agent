from __future__ import annotations

from typing import Protocol

from app.domain.session import SessionMemory


class SessionStore(Protocol):
    async def get(self, session_id: str) -> SessionMemory | None: ...

    async def save(self, session: SessionMemory) -> None: ...

    async def delete(self, session_id: str) -> None: ...
