from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.persistence.sqlalchemy.rls import apply_rls
from app.domain.auth import AuthContext


class SqlExecutor:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        auth: AuthContext | None = None,
    ) -> dict[str, Any]:
        bind = dict(params or {})
        async with self._session_factory() as session:
            if auth is not None:
                await apply_rls(session, auth)
            result = await session.execute(text(sql), bind)
            if result.returns_rows:
                rows = [dict(r) for r in result.mappings().all()]
                payload: dict[str, Any] = {
                    "rows": rows,
                    "row_count": len(rows),
                    "sql": sql,
                    "truncated": False,
                }
                if len(rows) == 1 and "count" in rows[0]:
                    payload["count"] = int(rows[0]["count"])
                return payload
            await session.commit()
            return {"rows": [], "row_count": 0, "sql": sql, "truncated": False}
