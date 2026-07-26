from __future__ import annotations

import json
import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


class PgVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

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

    async def similarity_search(
        self,
        *,
        embedding: list[float],
        top_k: int,
        tenant_id: UUID | str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT id, employee_id, section, content, metadata,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM resume_chunks
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                    """
                ),
                {"embedding": _vec_literal(embedding), "top_k": top_k},
            )
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

    async def delete_by_employee(self, employee_id: UUID | str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM resume_chunks WHERE employee_id = CAST(:eid AS uuid)"),
                {"eid": str(employee_id)},
            )
            await session.commit()
