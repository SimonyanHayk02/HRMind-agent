from __future__ import annotations

from pydantic import BaseModel

from app.domain.enums import Role


class AuthContext(BaseModel):
    user_id: str
    tenant_id: str
    role: Role
    department_id: str | None = None
    employee_id: str | None = None

    @property
    def permission_hash(self) -> str:
        return "|".join(
            [
                self.tenant_id,
                self.role.value,
                self.department_id or "-",
                self.employee_id or "-",
            ]
        )
