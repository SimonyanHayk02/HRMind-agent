from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any
from uuid import UUID, uuid4

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall((value or "").lower()))


class MemoryVectorStore:
    """In-process mirror of PgVectorStore, used by tests and the no-database path.

    Every retrieval method has the same contract as the Postgres implementation so
    the hermetic tests exercise the real pipeline rather than a simplified one.
    """

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []

    async def upsert(self, *, chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks:
            item = dict(chunk)
            item.setdefault("id", str(uuid4()))
            self._chunks.append(item)

    @staticmethod
    def _hit(chunk: dict[str, Any], score: float) -> dict[str, Any]:
        meta = chunk.get("metadata") or {}
        return {
            "id": chunk.get("id"),
            "employee_id": str(chunk.get("employee_id")),
            "employee_name": meta.get("employee_name"),
            "section": chunk.get("section"),
            "content": chunk.get("content"),
            "score": score,
            "metadata": meta,
        }

    def _matches_filters(self, chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        section = str(chunk.get("section") or "")
        sections = [s for s in (filters.get("sections") or []) if s]
        if sections and section not in sections:
            return False
        excluded = [s for s in (filters.get("exclude_sections") or []) if s]
        if excluded and section in excluded:
            return False
        employee_ids = {str(e) for e in (filters.get("employee_ids") or []) if e}
        return not (employee_ids and str(chunk.get("employee_id")) not in employee_ids)

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
            if not self._matches_filters(chunk, filters):
                continue
            emb = np.asarray(chunk.get("embedding") or [], dtype=float)
            if emb.size == 0 or emb.size != q.size:
                continue
            score = float(np.dot(q, emb) / ((np.linalg.norm(q) * np.linalg.norm(emb)) + 1e-9))
            scored.append((score, self._hit(chunk, score)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:top_k]]

    async def lexical_search(
        self,
        query: str,
        *,
        top_k: int = 40,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Token-overlap stand-in for Postgres full-text search."""
        terms = _tokens(query)
        if not terms:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._chunks:
            if not self._matches_filters(chunk, filters):
                continue
            content = _tokens(str(chunk.get("content") or ""))
            overlap = len(terms & content)
            if not overlap:
                continue
            score = overlap / len(terms)
            scored.append((score, self._hit(chunk, score)))
        scored.sort(key=lambda x: (-x[0], str(x[1].get("employee_id"))))
        return [h for _, h in scored[:top_k]]

    async def resolve_names(
        self,
        name: str,
        *,
        limit: int = 40,
        min_similarity: float = 0.35,
    ) -> list[dict[str, Any]]:
        q = (name or "").strip().lower()
        if not q:
            return []
        best: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            meta = chunk.get("metadata") or {}
            emp_name = str(meta.get("employee_name") or "")
            if not emp_name:
                continue
            lowered = emp_name.lower()
            if q in lowered:
                score = 1.0
            else:
                score = SequenceMatcher(None, q, lowered).ratio()
                if score < min_similarity:
                    continue
            eid = str(chunk.get("employee_id"))
            current = best.get(eid)
            if current is None or score > current["score"]:
                best[eid] = {
                    "employee_id": eid,
                    "employee_name": emp_name,
                    "score": score,
                }
        rows = sorted(
            best.values(), key=lambda r: (-r["score"], str(r["employee_name"]))
        )
        return rows[:limit]

    async def fetch_section_chunks(
        self,
        sections: Sequence[str],
        *,
        employee_ids: Sequence[str] | None = None,
        content_hints: Sequence[str] = (),
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        section_list = {s for s in sections if s}
        hints = [h.lower() for h in content_hints if h]
        if not section_list and not hints:
            return []
        scope = {str(e) for e in (employee_ids or []) if e}
        out: list[dict[str, Any]] = []
        for chunk in self._chunks:
            section = str(chunk.get("section") or "")
            content = str(chunk.get("content") or "")
            if section not in section_list and not any(h in content.lower() for h in hints):
                continue
            if scope and str(chunk.get("employee_id")) not in scope:
                continue
            out.append(self._hit(chunk, 1.0))
        out.sort(key=lambda h: (str(h.get("employee_id")), str(h.get("section"))))
        return out[:limit]

    async def count_indexed_employees(self) -> int:
        return len({str(c.get("employee_id")) for c in self._chunks})

    async def search_by_employee_name(
        self,
        name: str,
        *,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        q = (name or "").strip().lower()
        if not q:
            return []
        out = [
            self._hit(chunk, 1.0)
            for chunk in self._chunks
            if q in str((chunk.get("metadata") or {}).get("employee_name") or "").lower()
        ]
        return out[:limit]

    async def delete_by_employee(self, employee_id: UUID | str) -> None:
        eid = str(employee_id)
        self._chunks = [c for c in self._chunks if str(c.get("employee_id")) != eid]
