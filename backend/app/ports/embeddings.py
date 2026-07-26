from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingClient(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dims(self) -> int: ...

    async def embed(self, text: str) -> list[float]: ...

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...
