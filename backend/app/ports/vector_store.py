from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID


class VectorStore(Protocol):
    async def upsert(self, *, chunks: list[dict[str, Any]]) -> None: ...

    async def similarity_search(
        self,
        *,
        embedding: list[float],
        top_k: int,
        tenant_id: UUID | str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def delete_by_employee(self, employee_id: UUID | str) -> None: ...
