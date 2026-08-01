from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.schema_catalog import SchemaCatalog, default_employee_catalog
from app.ports.cache import CachePort
from app.ports.vector_store import VectorStore
from app.tools.resume_search.location import parse_location

# Employees-table enums are warmed from DISTINCT on employees. Location enums
# are resume-sourced, so they are read from the Location chunk slice instead —
# the application reading its own resume_chunks table, not the sql tool.
_EMPLOYEE_ENUM_COLS = frozenset(
    {"department", "position", "education", "employment_status"}
)
_RESUME_ENUM_COLS = frozenset({"country", "city"})


class CatalogService:
    """Serves a schema catalog, optionally refreshed with live corpus values."""

    CACHE_KEY = "schema_catalog:employees:v3"

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        cache: CachePort | None = None,
        ttl_seconds: int = 3600,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._cache = cache
        self._ttl = ttl_seconds
        self._vector_store = vector_store
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
                catalog = await self._refresh_employee_values(catalog)
            except Exception:
                # Keep seed defaults if DB refresh fails (boot / empty DB).
                pass
        if self._vector_store is not None:
            try:
                catalog = await self._refresh_location_values(catalog)
            except Exception:
                pass
        self._local = catalog
        if self._cache is not None:
            await self._cache.set(
                self.CACHE_KEY, catalog.model_dump(mode="json"), ttl_seconds=self._ttl
            )
        return catalog

    async def _refresh_employee_values(self, catalog: SchemaCatalog) -> SchemaCatalog:
        assert self._session_factory is not None
        async with self._session_factory() as session:
            for name in _EMPLOYEE_ENUM_COLS:
                col = catalog.by_name(name)
                if col is None or col.kind != "enum":
                    continue
                rows = await session.execute(
                    text(
                        f"SELECT DISTINCT {name} AS v FROM employees "
                        f"WHERE {name} IS NOT NULL ORDER BY 1 LIMIT 80"
                    )
                )
                values = [str(r.v) for r in rows if r.v is not None and str(r.v).strip()]
                if values:
                    col.values = _merge_values(col.values, values)
        return catalog

    async def _refresh_location_values(self, catalog: SchemaCatalog) -> SchemaCatalog:
        """Distinct cities/countries parsed from Location chunks in the corpus."""
        assert self._vector_store is not None
        chunks = await self._vector_store.fetch_section_chunks(
            ("Location",), content_hints=("based in", "location")
        )
        cities: list[str] = []
        countries: list[str] = []
        for chunk in chunks:
            place = parse_location(str(chunk.get("content") or ""))
            if place is None:
                continue
            if place.city:
                cities.append(place.city)
            if place.country:
                countries.append(place.country)
        city_col = catalog.by_name("city")
        if city_col is not None and cities:
            city_col.values = _merge_values(city_col.values, cities)
        country_col = catalog.by_name("country")
        if country_col is not None and countries:
            country_col.values = _merge_values(country_col.values, countries)
        return catalog

    def planner_view(self) -> dict[str, Any]:
        return self.get().planner_view()


def _merge_values(seed: list[str], live: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for v in list(seed) + live:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(v)
    return merged
