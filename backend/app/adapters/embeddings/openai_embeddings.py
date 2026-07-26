from __future__ import annotations

from typing import Sequence

from openai import AsyncOpenAI


class OpenAIEmbeddings:
    def __init__(self, *, api_key: str, model: str, dims: int = 1536) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dims = dims

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in resp.data]
