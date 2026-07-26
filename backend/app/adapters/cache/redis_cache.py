from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis


class RedisCache:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def get(self, key: str) -> Any | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, default=str)
        if ttl_seconds is None:
            await self._redis.set(key, payload)
        else:
            await self._redis.set(key, payload, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def delete_prefix(self, prefix: str) -> None:
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor=cursor, match=f"{prefix}*", count=100)
            if keys:
                await self._redis.delete(*keys)
            if cursor == 0:
                break
