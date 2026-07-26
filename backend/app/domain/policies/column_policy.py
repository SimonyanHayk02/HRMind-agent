from __future__ import annotations

from app.domain.enums import Role

BASE_EMPLOYEE_COLUMNS = {
    "id",
    "tenant_id",
    "first_name",
    "last_name",
    "email",
    "department",
    "position",
    "hire_date",
    "country",
    "city",
    "manager_id",
    "education",
    "employment_status",
    "created_at",
    "updated_at",
}
SENSITIVE_COLUMNS = {"salary"}


def allowed_columns(
    role: Role,
    *,
    department: str | None = None,
    target_department: str | None = None,
) -> set[str]:
    cols = set(BASE_EMPLOYEE_COLUMNS)
    if role == Role.RECRUITER:
        cols |= SENSITIVE_COLUMNS
    elif role == Role.MANAGER:
        if department and target_department and department == target_department:
            cols |= SENSITIVE_COLUMNS
    return cols


def is_column_allowed(
    role: Role,
    column: str,
    *,
    department: str | None = None,
    target_department: str | None = None,
) -> bool:
    return column in allowed_columns(
        role, department=department, target_department=target_department
    )
