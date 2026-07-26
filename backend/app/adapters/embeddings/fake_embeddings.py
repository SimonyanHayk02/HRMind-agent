from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class FakeEmbeddings:
    def __init__(self, *, model_name: str = "fake-embed", dims: int = 64) -> None:
        self._model_name = model_name
        self._dims = dims

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return self._dims

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dims)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.astype(float).tolist()

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
