from __future__ import annotations

from typing import Any

from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import ForbiddenError

TOOL_PERMISSIONS: dict[str, set[Role]] = {
    "greeting": {Role.RECRUITER, Role.MANAGER, Role.EMPLOYEE},
    "employee": {Role.RECRUITER, Role.MANAGER, Role.EMPLOYEE},
    "sql": {Role.RECRUITER, Role.MANAGER},
    "resume_search": {Role.RECRUITER, Role.MANAGER},
    "clarify": {Role.RECRUITER, Role.MANAGER, Role.EMPLOYEE},
}


def can_use_tool(auth: AuthContext, tool_name: str) -> bool:
    allowed = TOOL_PERMISSIONS.get(tool_name)
    return bool(allowed and auth.role in allowed)


def require_tool(auth: AuthContext, tool_name: str) -> None:
    if not can_use_tool(auth, tool_name):
        raise ForbiddenError(f"Role {auth.role} cannot use tool {tool_name}")


def can_access_employee(auth: AuthContext, *, target_id: str, target_department: str) -> bool:
    if auth.role == Role.RECRUITER:
        return True
    if auth.role == Role.EMPLOYEE:
        return auth.employee_id is not None and auth.employee_id == str(target_id)
    if auth.role == Role.MANAGER:
        if auth.employee_id and auth.employee_id == str(target_id):
            return True
        return auth.department_id is not None and auth.department_id == target_department
    return False


def filter_employee_payload(
    auth: AuthContext, payload: dict[str, Any], *, target_department: str
) -> dict[str, Any]:
    from app.domain.policies.column_policy import allowed_columns

    cols = allowed_columns(
        auth.role, department=auth.department_id, target_department=target_department
    )
    return {k: v for k, v in payload.items() if k in cols or k in {"id", "full_name"}}
