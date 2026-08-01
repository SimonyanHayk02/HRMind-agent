from __future__ import annotations

from fastapi import Header, HTTPException, Request

from app.domain.auth import AuthContext
from app.domain.enums import Role

# Aliases seen in frontends / env typos → canonical Role values.
_ROLE_ALIASES: dict[str, Role] = {
    "recruiter": Role.RECRUITER,
    "recruiters": Role.RECRUITER,
    "admin": Role.RECRUITER,
    "administrator": Role.RECRUITER,
    "hr": Role.RECRUITER,
    "hr_manager": Role.RECRUITER,
    "hrmanager": Role.RECRUITER,
    "hr-manager": Role.RECRUITER,
    "manager": Role.MANAGER,
    "managers": Role.MANAGER,
    "employee": Role.EMPLOYEE,
    "employees": Role.EMPLOYEE,
}


def parse_role(raw: str | None) -> Role:
    """Map X-Role to a Role without raising (invalid → 400, not 500)."""
    key = (raw or "").strip().lower().replace(" ", "_")
    if not key:
        return Role.RECRUITER
    if key in _ROLE_ALIASES:
        return _ROLE_ALIASES[key]
    try:
        return Role(key)
    except ValueError as exc:
        allowed = ", ".join(r.value for r in Role)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X-Role {raw!r}. Allowed: {allowed}",
        ) from exc


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
        role=parse_role(x_role),
        department_id=x_department_id,
        employee_id=x_employee_id,
    )
