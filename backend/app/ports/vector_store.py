from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID


class VectorStore(Protocol):
    """Retrieval surface over the resume corpus.

    Dense, sparse and fuzzy-name retrieval are all declared here so callers can
    depend on the protocol instead of probing for optional methods.
    """

    async def upsert(self, *, chunks: list[dict[str, Any]]) -> None: ...

    async def similarity_search(
        self,
        *,
        embedding: list[float],
        top_k: int,
        tenant_id: UUID | str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def lexical_search(
        self,
        query: str,
        *,
        top_k: int = 40,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def resolve_names(
        self,
        name: str,
        *,
        limit: int = 40,
        min_similarity: float = 0.35,
    ) -> list[dict[str, Any]]: ...

    async def fetch_section_chunks(
        self,
        sections: Sequence[str],
        *,
        employee_ids: Sequence[str] | None = None,
        content_hints: Sequence[str] = (),
        limit: int = 2000,
    ) -> list[dict[str, Any]]: ...

    async def count_indexed_employees(self) -> int: ...

    async def search_by_employee_name(
        self,
        name: str,
        *,
        limit: int = 40,
    ) -> list[dict[str, Any]]: ...

    async def delete_by_employee(self, employee_id: UUID | str) -> None: ...
