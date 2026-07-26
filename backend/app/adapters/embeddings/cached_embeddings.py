from __future__ import annotations

from typing import Sequence

from app.adapters.cache.keys import embedding_key
from app.ports.cache import CachePort
from app.ports.embeddings import EmbeddingClient


class CachedEmbeddings:
    def __init__(self, inner: EmbeddingClient, cache: CachePort) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dims(self) -> int:
        return self._inner.dims

    async def embed(self, text: str) -> list[float]:
        key = embedding_key(model=self.model_name, text=text)
        cached = await self._cache.get(key)
        if cached is not None:
            return cached
        vec = await self._inner.embed(text)
        await self._cache.set(key, vec)
        return vec

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            key = embedding_key(model=self.model_name, text=text)
            cached = await self._cache.get(key)
            if cached is not None:
                out[i] = cached
            else:
                missing.append((i, text))
        if missing:
            vectors = await self._inner.embed_many([t for _, t in missing])
            for (i, text), vec in zip(missing, vectors):
                key = embedding_key(model=self.model_name, text=text)
                await self._cache.set(key, vec)
                out[i] = vec
        return [v if v is not None else [] for v in out]
