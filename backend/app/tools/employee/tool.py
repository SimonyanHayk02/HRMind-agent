from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import ForbiddenError, NotFoundError
from app.domain.policies.column_policy import allowed_columns
from app.domain.policies.rbac import can_access_employee, require_tool
from app.domain.tools.base import SourceRef, ToolMeta, ToolResult
from app.ports.employee_repository import EmployeeRepository


class EmployeeTool:
    def __init__(self, repo_factory: Callable[[], Any]) -> None:
        """repo_factory returns an async context manager yielding object with .employees"""
        self._repo_factory = repo_factory
        self._meta = ToolMeta(
            name="employee",
            description="Lookup employee profile, manager, or department roster",
            permissions=list(Role),
            estimated_latency_ms=50,
            input_schema={
                "action": "profile|manager|department|by_email|by_id|by_name",
                "employee_id": "uuid?",
                "email": "str?",
                "department": "str?",
                "name": "str?",
            },
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        require_tool(auth, "employee")
        action = params.get("action", "profile")
        async with self._repo_factory(auth) as uow:
            employees: EmployeeRepository = uow.employees
            if action == "by_name":
                name = params.get("name") or params.get("question") or ""
                matches = await employees.search_by_name(name, limit=5)
                accessible = [
                    m
                    for m in matches
                    if can_access_employee(
                        auth, target_id=str(m.id), target_department=m.department
                    )
                ]
                if not accessible:
                    return ToolResult(data={"employees": []}, confidence=0.2)
                if len(accessible) > 1:
                    options = [f"{e.full_name} ({e.department})" for e in accessible]
                    return ToolResult(
                        data={
                            "clarify": (
                                f"I found multiple people named like '{name}'. "
                                f"Which one did you mean: {', '.join(options)}?"
                            ),
                            "candidates": [self._serialize(auth, e) for e in accessible],
                        },
                        confidence=0.4,
                    )
                emp = accessible[0]
                return ToolResult(
                    data=self._serialize(auth, emp),
                    confidence=0.95,
                    sources=[SourceRef(kind="employee", ref=str(emp.id), label=emp.full_name)],
                )
            if action == "by_email":
                emp = await employees.get_by_email(params["email"])
            elif action == "by_id" or action == "profile":
                emp = await employees.get_by_id(UUID(str(params["employee_id"])))
            elif action == "manager":
                # Resolve by id if provided, else by name in question
                if params.get("employee_id"):
                    emp = await employees.get_by_id(UUID(str(params["employee_id"])))
                elif params.get("name"):
                    found = await employees.search_by_name(params["name"], limit=1)
                    emp = found[0] if found else None
                else:
                    q = params.get("question") or ""
                    # crude: take last capitalized token pair
                    parts = q.replace("'s", " ").split()
                    name_guess = " ".join(parts[:2]) if parts else q
                    found = await employees.search_by_name(name_guess, limit=5)
                    if len(found) != 1:
                        return ToolResult(
                            data={
                                "clarify": "Which employee’s manager should I look up?",
                                "candidates": [self._serialize(auth, e) for e in found],
                            },
                            confidence=0.4,
                        )
                    emp = found[0]
                if emp is None:
                    raise NotFoundError("Employee not found")
                if not can_access_employee(
                    auth, target_id=str(emp.id), target_department=emp.department
                ):
                    raise ForbiddenError()
                chain = await employees.get_manager_chain(emp.id)
                return ToolResult(
                    data={"employee": self._serialize(auth, emp), "managers": [self._serialize(auth, m) for m in chain]},
                    confidence=1.0,
                    sources=[SourceRef(kind="employee", ref=str(emp.id))],
                )
            elif action == "department":
                dept = params.get("department") or auth.department_id
                if auth.role == Role.EMPLOYEE:
                    raise ForbiddenError("Employees cannot list departments")
                if auth.role == Role.MANAGER and dept != auth.department_id:
                    raise ForbiddenError("Managers can only list their department")
                rows = await employees.list_by_department(dept)
                return ToolResult(
                    data={"employees": [self._serialize(auth, e) for e in rows]},
                    confidence=1.0,
                )
            else:
                raise NotFoundError(f"Unknown action {action}")

            if emp is None:
                raise NotFoundError("Employee not found")
            if not can_access_employee(
                auth, target_id=str(emp.id), target_department=emp.department
            ):
                raise ForbiddenError()
            return ToolResult(
                data=self._serialize(auth, emp),
                confidence=1.0,
                sources=[SourceRef(kind="employee", ref=str(emp.id), label=emp.full_name)],
            )

    def _serialize(self, auth: AuthContext, emp) -> dict[str, Any]:
        cols = allowed_columns(
            auth.role, department=auth.department_id, target_department=emp.department
        )
        payload = emp.model_dump(mode="json")
        payload["full_name"] = emp.full_name
        return {k: v for k, v in payload.items() if k in cols or k == "full_name"}
