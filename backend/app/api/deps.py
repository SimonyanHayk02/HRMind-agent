from __future__ import annotations

from fastapi import Header, Request

from app.domain.auth import AuthContext
from app.domain.enums import Role


def get_auth_context(
    request: Request,
    x_user_id: str = Header(default="dev-user", alias="X-User-Id"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_role: str = Header(default="recruiter", alias="X-Role"),
    x_department_id: str | None = Header(default=None, alias="X-Department-Id"),
    x_employee_id: str | None = Header(default=None, alias="X-Employee-Id"),
) -> AuthContext:
    from app.config.settings import get_settings

    container = getattr(request.app.state, "container", None)
    settings = container.settings if container is not None else get_settings()
    return AuthContext(
        user_id=x_user_id,
        tenant_id=x_tenant_id or settings.default_tenant_id,
        role=Role(x_role),
        department_id=x_department_id,
        employee_id=x_employee_id,
    )
