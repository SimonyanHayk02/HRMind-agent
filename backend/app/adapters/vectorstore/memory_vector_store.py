from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import numpy as np


class MemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []

    async def upsert(self, *, chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks:
            item = dict(chunk)
            item.setdefault("id", str(uuid4()))
            self._chunks.append(item)

    async def similarity_search(
        self,
        *,
        embedding: list[float],
        top_k: int,
        tenant_id: UUID | str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._chunks:
            return []
        q = np.asarray(embedding, dtype=float)
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._chunks:
            emb = np.asarray(chunk.get("embedding") or [], dtype=float)
            if emb.size == 0 or emb.size != q.size:
                continue
            score = float(np.dot(q, emb) / ((np.linalg.norm(q) * np.linalg.norm(emb)) + 1e-9))
            hit = {
                "id": chunk.get("id"),
                "employee_id": chunk.get("employee_id"),
                "employee_name": (chunk.get("metadata") or {}).get("employee_name"),
                "section": chunk.get("section"),
                "content": chunk.get("content"),
                "score": score,
                "metadata": chunk.get("metadata") or {},
            }
            scored.append((score, hit))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:top_k]]

    async def delete_by_employee(self, employee_id: UUID | str) -> None:
        eid = str(employee_id)
        self._chunks = [c for c in self._chunks if str(c.get("employee_id")) != eid]
