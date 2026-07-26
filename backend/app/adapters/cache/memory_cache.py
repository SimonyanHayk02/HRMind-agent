from __future__ import annotations

import asyncio
from typing import Any


class MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            return self._store.get(key)

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        async with self._lock:
            self._store[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def delete_prefix(self, prefix: str) -> None:
        async with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for key in keys:
                del self._store[key]
