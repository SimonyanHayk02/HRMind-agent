from __future__ import annotations

from pydantic import ValidationError
from redis.asyncio import Redis

from app.config.logging import get_logger
from app.domain.session import SessionMemory

logger = get_logger(__name__)


class RedisSessionStore:
    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "hrmind:session:",
        ttl_seconds: int = 86400,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def get(self, session_id: str) -> SessionMemory | None:
        key = self._key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        # Refresh TTL on active reads so long conversations stay alive.
        await self._redis.expire(key, self._ttl)
        try:
            return SessionMemory.model_validate_json(raw)
        except ValidationError as exc:
            # Corrupt / schema-drifted sessions must not 500 cold starts.
            logger.warning(
                "session_deserialize_failed",
                session_id=session_id,
                error=str(exc)[:300],
            )
            await self._redis.delete(key)
            return None

    async def save(self, session: SessionMemory) -> None:
        await self._redis.set(
            self._key(session.session_id),
            session.model_dump_json(),
            ex=self._ttl,
        )

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
