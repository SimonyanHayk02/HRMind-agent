from __future__ import annotations

from redis.asyncio import Redis

from app.domain.session import SessionMemory


class RedisSessionStore:
    def __init__(self, redis: Redis, *, prefix: str = "hrmind:session:") -> None:
        self._redis = redis
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def get(self, session_id: str) -> SessionMemory | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return SessionMemory.model_validate_json(raw)

    async def save(self, session: SessionMemory) -> None:
        await self._redis.set(self._key(session.session_id), session.model_dump_json(), ex=86400)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
