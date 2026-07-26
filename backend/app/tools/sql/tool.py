from __future__ import annotations

from typing import Any, Callable

from app.adapters.cache import keys as cache_keys
from app.adapters.sql.executor import SqlExecutor
from app.adapters.sql.query_builder import QueryBuilder
from app.adapters.sql.validator import validate_sql
from app.domain.auth import AuthContext
from app.domain.enums import Role, SqlMode
from app.domain.policies.column_policy import allowed_columns
from app.domain.policies.rbac import require_tool
from app.domain.tools.base import SourceRef, ToolMeta, ToolResult
from app.ports.cache import CachePort
from app.ports.llm import LLMClient


class SqlTool:
    def __init__(
        self,
        *,
        executor: SqlExecutor,
        cache: CachePort,
        llm: LLMClient,
        max_rows: int = 200,
        cache_ttl: int = 60,
        prompt_loader: Callable[[str], str] | None = None,
    ) -> None:
        self._executor = executor
        self._cache = cache
        self._llm = llm
        self._max_rows = max_rows
        self._cache_ttl = cache_ttl
        self._builder = QueryBuilder()
        self._prompt_loader = prompt_loader or (lambda name: "")
        self._meta = ToolMeta(
            name="sql",
            description="Query structured employee data via constrained filters or NL2SQL",
            permissions=[Role.RECRUITER, Role.MANAGER],
            estimated_latency_ms=200,
            cache_policy="sql",
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        require_tool(auth, "sql")
        mode = params.get("mode", SqlMode.CONSTRAINED.value)
        if mode == SqlMode.CONSTRAINED.value:
            return await self._constrained(params, auth=auth)
        return await self._nl2sql(params, auth=auth)

    async def _constrained(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        allowed = allowed_columns(
            auth.role, department=auth.department_id, target_department=auth.department_id
        )
        requested = params.get("columns")
        if requested:
            cols = [c for c in requested if c in allowed]
            if not cols:
                cols = sorted(allowed)
        else:
            cols = sorted(allowed)
        count_only = bool(params.get("count_only"))
        distinct = bool(params.get("distinct"))
        count_distinct = params.get("count_distinct")
        filters = dict(params.get("filters") or {})
        if params.get("employee_ids") is not None:
            raw = params.get("employee_ids")
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                cleaned = []
                for x in raw:
                    try:
                        from uuid import UUID

                        cleaned.append(str(UUID(str(x))))
                    except Exception:
                        continue
                filters["employee_ids"] = cleaned
            else:
                filters["employee_ids"] = []
        sql, bind = self._builder.build(
            columns=cols,
            filters=filters,
            max_rows=self._max_rows,
            count_only=count_only,
            distinct=distinct,
            count_distinct=count_distinct,
        )
        key = cache_keys.build("sql", auth, sql=sql, params=bind)
        cached = await self._cache.get(key)
        if cached is not None:
            return ToolResult(data=cached, confidence=1.0, cache_hit=True, sources=[SourceRef(kind="sql", ref=sql)])
        data = await self._executor.execute(sql, bind, auth=auth)
        await self._cache.set(key, data, ttl_seconds=self._cache_ttl)
        return ToolResult(data=data, confidence=1.0, sources=[SourceRef(kind="sql", ref=sql)])

    async def _nl2sql(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        question = params.get("question") or ""
        schema = "TABLE employees(id, first_name, last_name, email, department, position, salary, hire_date, country, city, manager_id, education, employment_status)"
        system = self._prompt_loader("sql_nl2sql") or (
            "Generate a single PostgreSQL SELECT for the employees table. Return only SQL."
        )
        user = f"Schema:\n{schema}\n\nQuestion:\n{question}"
        raw_sql = await self._llm.complete(system=system, user=user, temperature=0.0)
        raw_sql = raw_sql.strip().strip("`")
        if raw_sql.lower().startswith("sql"):
            raw_sql = raw_sql[3:].strip()
        try:
            sql = validate_sql(raw_sql, auth, max_rows=self._max_rows)
            data = await self._executor.execute(sql, auth=auth)
            return ToolResult(data=data, confidence=0.8, sources=[SourceRef(kind="sql", ref=sql)])
        except Exception as exc:
            # one repair attempt
            repair = self._prompt_loader("sql_repair") or "Fix this SQL to be a valid SELECT only."
            fixed = await self._llm.complete(
                system=repair, user=f"Error: {exc}\nSQL: {raw_sql}", temperature=0.0
            )
            fixed = fixed.strip().strip("`")
            sql = validate_sql(fixed, auth, max_rows=self._max_rows)
            data = await self._executor.execute(sql, auth=auth)
            return ToolResult(data=data, confidence=0.7, sources=[SourceRef(kind="sql", ref=sql)])
