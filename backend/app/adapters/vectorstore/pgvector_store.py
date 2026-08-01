from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# A name match that contains the query verbatim; anything lower came from trigram
# similarity and is therefore a fuzzy (typo-tolerant) match.
EXACT_NAME_SCORE = 1.0


def _vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _uuid_array(values: Iterable[Any]) -> list[UUID]:
    """asyncpg binds ``uuid[]`` parameters from UUID objects, not strings."""
    out: list[UUID] = []
    for value in values:
        out.append(value if isinstance(value, UUID) else UUID(str(value)))
    return out


@dataclass(frozen=True)
class _Capabilities:
    trigram: bool


class PgVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._caps: _Capabilities | None = None

    async def _capabilities(self) -> _Capabilities:
        """Probe optional extensions once so a DB without them degrades instead of failing."""
        if self._caps is None:
            async with self._session_factory() as session:
                row = (
                    await session.execute(
                        text(
                            "SELECT EXISTS "
                            "(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') AS trigram"
                        )
                    )
                ).mappings().one()
            self._caps = _Capabilities(trigram=bool(row["trigram"]))
        return self._caps

    async def upsert(self, *, chunks: list[dict[str, Any]]) -> None:
        async with self._session_factory() as session:
            for chunk in chunks:
                await session.execute(
                    text(
                        """
                        INSERT INTO resume_chunks
                        (id, employee_id, resume_id, section, chunk_index, content, embedding, metadata)
                        VALUES
                        (
                          CAST(:id AS uuid),
                          CAST(:employee_id AS uuid),
                          CAST(:resume_id AS uuid),
                          :section,
                          :chunk_index,
                          :content,
                          CAST(:embedding AS vector),
                          CAST(:metadata AS jsonb)
                        )
                        """
                    ),
                    {
                        "id": chunk.get("id") or str(uuid.uuid4()),
                        "employee_id": str(chunk["employee_id"]),
                        "resume_id": str(chunk["resume_id"]),
                        "section": chunk["section"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "embedding": _vec_literal(chunk["embedding"]),
                        "metadata": json.dumps(chunk.get("metadata") or {}),
                    },
                )
            await session.commit()

    @staticmethod
    def _filter_clauses(filters: dict[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
        """Translate the retrieval filter dict into indexable SQL predicates."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if not filters:
            return clauses, params

        sections = [str(s) for s in (filters.get("sections") or []) if s]
        if sections:
            clauses.append("section = ANY(CAST(:f_sections AS text[]))")
            params["f_sections"] = sections

        excluded = [str(s) for s in (filters.get("exclude_sections") or []) if s]
        if excluded:
            clauses.append("NOT (section = ANY(CAST(:f_excluded AS text[])))")
            params["f_excluded"] = excluded

        employee_ids = [e for e in (filters.get("employee_ids") or []) if e]
        if employee_ids:
            clauses.append("employee_id = ANY(CAST(:f_eids AS uuid[]))")
            params["f_eids"] = _uuid_array(employee_ids)

        return clauses, params

    @staticmethod
    def _to_hits(result: Any) -> list[dict[str, Any]]:
        rows = []
        for r in result.mappings().all():
            meta = r["metadata"] or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            rows.append(
                {
                    "id": str(r["id"]),
                    "employee_id": str(r["employee_id"]),
                    "section": r["section"],
                    "content": r["content"],
                    "score": float(r["score"] or 0),
                    "metadata": meta,
                    "employee_name": meta.get("employee_name"),
                }
            )
        return rows

    async def similarity_search(
        self,
        *,
        embedding: list[float],
        top_k: int,
        tenant_id: UUID | str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense kNN, restricted to the requested slice of the corpus.

        Filtered kNN under an HNSW index can under-fill its result set, which is
        why migration 003 turns on ``hnsw.iterative_scan`` for this database.
        """
        clauses, params = self._filter_clauses(filters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.update({"embedding": _vec_literal(embedding), "top_k": int(top_k)})
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id, employee_id, section, content, metadata,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM resume_chunks
                    {where}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                    """
                ),
                params,
            )
            return self._to_hits(result)

    async def lexical_search(
        self,
        query: str,
        *,
        top_k: int = 40,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Postgres full-text search over chunk content (the sparse retrieval arm).

        Chunks are enriched with the employee's name and role at ingest, so exact
        token matches on people, positions and cities land here rather than
        relying on embedding similarity alone.
        """
        q = (query or "").strip()
        if not q:
            return []
        clauses, params = self._filter_clauses(filters)
        clauses.append("to_tsvector('simple', content) @@ plainto_tsquery('simple', :q)")
        params.update({"q": q, "top_k": int(top_k)})
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id, employee_id, section, content, metadata,
                           ts_rank(
                             to_tsvector('simple', content),
                             plainto_tsquery('simple', :q)
                           ) AS score
                    FROM resume_chunks
                    WHERE {' AND '.join(clauses)}
                    ORDER BY score DESC, employee_id
                    LIMIT :top_k
                    """
                ),
                params,
            )
            return self._to_hits(result)

    async def resolve_names(
        self,
        name: str,
        *,
        limit: int = 40,
        min_similarity: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Candidate employees for a person name, tolerant of typos.

        Returns one row per employee (``employee_id``, ``employee_name``,
        ``score``), where a score of 1.0 means the name contains the query
        verbatim and anything lower is a trigram near-match.
        """
        q = (name or "").strip()
        if not q:
            return []
        caps = await self._capabilities()
        name_expr = "lower(COALESCE(metadata->>'employee_name', ''))"
        exact = f"CASE WHEN {name_expr} LIKE lower(:pattern) THEN 1.0 ELSE 0.0 END"
        params: dict[str, Any] = {
            "pattern": f"%{q}%",
            "limit": int(limit),
        }
        if caps.trigram:
            score_expr = f"GREATEST({exact}, similarity({name_expr}, lower(:name)))"
            where = f"{name_expr} LIKE lower(:pattern) OR similarity({name_expr}, lower(:name)) >= :min_sim"
            params.update({"name": q, "min_sim": float(min_similarity)})
        else:
            score_expr = exact
            where = f"{name_expr} LIKE lower(:pattern)"

        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT employee_id,
                           max(COALESCE(metadata->>'employee_name', '')) AS employee_name,
                           max({score_expr}) AS score
                    FROM resume_chunks
                    WHERE {where}
                    GROUP BY employee_id
                    ORDER BY score DESC, employee_name
                    LIMIT :limit
                    """
                ),
                params,
            )
            return [
                {
                    "employee_id": str(r["employee_id"]),
                    "employee_name": r["employee_name"] or None,
                    "score": float(r["score"] or 0),
                }
                for r in result.mappings().all()
            ]

    async def fetch_section_chunks(
        self,
        sections: Sequence[str],
        *,
        employee_ids: Sequence[str] | None = None,
        content_hints: Sequence[str] = (),
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """Every chunk in the given sections, optionally for specific employees.

        Cohort questions ("whose birthday is today") need completeness rather than
        ranking, so this retrieves the whole section slice instead of a top-k.
        """
        section_list = [str(s) for s in sections if s]
        hints = [f"%{h}%" for h in content_hints if h]
        if not section_list and not hints:
            return []

        params: dict[str, Any] = {"limit": int(limit)}
        matchers: list[str] = []
        if section_list:
            matchers.append("section = ANY(CAST(:sections AS text[]))")
            params["sections"] = section_list
        if hints:
            matchers.append("content ILIKE ANY(CAST(:hints AS text[]))")
            params["hints"] = hints
        clauses = [f"({' OR '.join(matchers)})"]

        ids = [e for e in (employee_ids or []) if e]
        if ids:
            clauses.append("employee_id = ANY(CAST(:eids AS uuid[]))")
            params["eids"] = _uuid_array(ids)

        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id, employee_id, section, content, metadata, 1.0 AS score
                    FROM resume_chunks
                    WHERE {' AND '.join(clauses)}
                    ORDER BY employee_id, chunk_index
                    LIMIT :limit
                    """
                ),
                params,
            )
            return self._to_hits(result)

    async def count_indexed_employees(self) -> int:
        """Employees with at least one ingested chunk — the denominator for coverage."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT count(DISTINCT employee_id) AS n FROM resume_chunks")
            )
            return int(result.scalar() or 0)

    async def search_by_employee_name(
        self,
        name: str,
        *,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Find resume chunks whose metadata employee_name matches (resume-side resolve)."""
        q = (name or "").strip()
        if not q:
            return []
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT id, employee_id, section, content, metadata, 1.0 AS score
                    FROM resume_chunks
                    WHERE lower(COALESCE(metadata->>'employee_name', ''))
                          LIKE lower(:pattern)
                    ORDER BY employee_id, chunk_index
                    LIMIT :limit
                    """
                ),
                {"pattern": f"%{q}%", "limit": int(limit)},
            )
            return self._to_hits(result)

    async def delete_by_employee(self, employee_id: UUID | str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM resume_chunks WHERE employee_id = CAST(:eid AS uuid)"),
                {"eid": str(employee_id)},
            )
            await session.commit()
