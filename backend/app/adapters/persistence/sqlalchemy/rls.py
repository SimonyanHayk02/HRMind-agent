from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.domain.auth import AuthContext


async def apply_rls(session: AsyncSession | AsyncConnection, auth: AuthContext) -> None:
    statements = [
        ("app.tenant_id", auth.tenant_id),
        ("app.role", auth.role.value),
        ("app.employee_id", auth.employee_id or ""),
        ("app.department_id", auth.department_id or ""),
    ]
    for key, value in statements:
        await session.execute(text("SELECT set_config(:key, :value, true)"), {"key": key, "value": value})
