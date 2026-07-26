from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.schema_catalog import SchemaCatalog, default_employee_catalog
from app.ports.cache import CachePort


class CatalogService:
    """Serves a schema catalog, optionally refreshed with DISTINCT DB values."""

    CACHE_KEY = "schema_catalog:employees:v1"

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        cache: CachePort | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        self._session_factory = session_factory
        self._cache = cache
        self._ttl = ttl_seconds
        self._local: SchemaCatalog = default_employee_catalog()

    def get(self) -> SchemaCatalog:
        return self._local

    async def warm(self) -> SchemaCatalog:
        if self._cache is not None:
            cached = await self._cache.get(self.CACHE_KEY)
            if isinstance(cached, dict):
                try:
                    self._local = SchemaCatalog.model_validate(cached)
                    return self._local
                except Exception:
                    pass
        catalog = default_employee_catalog()
        if self._session_factory is not None:
            try:
                catalog = await self._refresh_values(catalog)
            except Exception:
                # Keep seed defaults if DB refresh fails (boot / empty DB).
                pass
        self._local = catalog
        if self._cache is not None:
            await self._cache.set(
                self.CACHE_KEY, catalog.model_dump(mode="json"), ttl_seconds=self._ttl
            )
        return catalog

    async def _refresh_values(self, catalog: SchemaCatalog) -> SchemaCatalog:
        assert self._session_factory is not None
        cols = [
            c.name
            for c in catalog.columns
            if c.kind == "enum"
            and c.name
            in {
                "department",
                "country",
                "city",
                "position",
                "education",
                "employment_status",
            }
        ]
        async with self._session_factory() as session:
            for name in cols:
                rows = await session.execute(
                    text(
                        f"SELECT DISTINCT {name} AS v FROM employees "
                        f"WHERE {name} IS NOT NULL ORDER BY 1 LIMIT 80"
                    )
                )
                values = [str(r.v) for r in rows if r.v is not None and str(r.v).strip()]
                col = catalog.by_name(name)
                if col is not None and values:
                    # Preserve seed order preferences, append new DB values
                    merged: list[str] = []
                    seen: set[str] = set()
                    for v in list(col.values) + values:
                        key = v.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        merged.append(v)
                    col.values = merged
        return catalog

    def planner_view(self) -> dict[str, Any]:
        return self.get().planner_view()
