from __future__ import annotations

import re
from typing import Any, Callable
from uuid import UUID

from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import ForbiddenError, NotFoundError
from app.domain.policies.column_policy import allowed_columns
from app.domain.policies.rbac import can_access_employee, require_tool
from app.domain.tools.base import SourceRef, ToolMeta, ToolResult
from app.ports.employee_repository import EmployeeRepository

_MANAGER_OF_RE = re.compile(
    r"\b(?:manager of|manages?|who manages)\s+(?:the\s+)?(.+?)(?:\?|$)",
    re.I,
)
_POSSESSIVE_MANAGER_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\s*'s\s+manager\b",
    re.I,
)
_DEPT_HINT_RE = re.compile(
    r"\b(?:in|from)\s+(Engineering|People|Sales|Finance|Product|Operations)\b",
    re.I,
)


def extract_manager_subject(question: str) -> str | None:
    """Pull the employee name from common manager phrasings."""
    q = (question or "").strip()
    if not q:
        return None
    m = _POSSESSIVE_MANAGER_RE.search(q)
    if m:
        return m.group(1).strip(" .,?!")
    m = _MANAGER_OF_RE.search(q)
    if m:
        name = m.group(1).strip(" .,?!")
        # Drop trailing junk like "please"
        name = re.sub(r"\b(please|now)\b", "", name, flags=re.I).strip(" .,?!")
        return name or None
    return None


def _option_label(emp: Any) -> str:
    bits = [emp.full_name]
    detail = ", ".join(
        x
        for x in (
            getattr(emp, "position", None),
            getattr(emp, "department", None),
            getattr(emp, "city", None),
        )
        if x
    )
    if detail:
        bits.append(detail)
    email = getattr(emp, "email", None)
    if email:
        bits.append(str(email))
    return " — ".join(bits)


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
                dept_hint = params.get("department") or _dept_from_text(name) or _dept_from_text(
                    params.get("question") or ""
                )
                return await self._resolve_by_name(
                    employees, auth, name=name, department=dept_hint
                )
            if action == "by_email":
                emp = await employees.get_by_email(params["email"])
            elif action == "by_id" or action == "profile":
                emp = await employees.get_by_id(UUID(str(params["employee_id"])))
            elif action == "manager":
                return await self._manager(employees, auth, params)
            elif action == "department":
                dept = params.get("department") or auth.department_id
                if auth.role == Role.EMPLOYEE:
                    raise ForbiddenError("Employees cannot list departments")
                if auth.role == Role.MANAGER and dept != auth.department_id:
                    raise ForbiddenError("Managers can only list their department")
                rows = await employees.list_by_department(dept)
                return ToolResult(
                    data={
                        "employees": [self._serialize(auth, e) for e in rows],
                        "employee_ids": [str(e.id) for e in rows],
                    },
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

    async def _manager(
        self, employees: EmployeeRepository, auth: AuthContext, params: dict[str, Any]
    ) -> ToolResult:
        emp = None
        if params.get("employee_id"):
            emp = await employees.get_by_id(UUID(str(params["employee_id"])))
        else:
            question = params.get("question") or ""
            name = (
                params.get("name")
                or extract_manager_subject(question)
                or ""
            )
            if not name:
                return ToolResult(
                    data={"clarify": "Which employee’s manager should I look up?"},
                    confidence=0.4,
                )
            dept = (
                params.get("department")
                or _dept_from_text(question)
                or _dept_from_text(name)
            )
            resolved = await self._resolve_by_name(
                employees, auth, name=name, department=dept
            )
            if resolved.data and isinstance(resolved.data, dict) and resolved.data.get("clarify"):
                return resolved
            if resolved.data and isinstance(resolved.data, dict) and resolved.data.get("id"):
                emp = await employees.get_by_id(UUID(str(resolved.data["id"])))
            else:
                return ToolResult(
                    data={"clarify": f"I couldn't find an employee named '{name}'."},
                    confidence=0.3,
                )

        if emp is None:
            raise NotFoundError("Employee not found")
        if not can_access_employee(
            auth, target_id=str(emp.id), target_department=emp.department
        ):
            raise ForbiddenError()
        chain = await employees.get_manager_chain(emp.id)
        return ToolResult(
            data={
                "employee": self._serialize(auth, emp),
                "managers": [self._serialize(auth, m) for m in chain],
            },
            confidence=1.0,
            sources=[SourceRef(kind="employee", ref=str(emp.id), label=emp.full_name)],
        )

    async def _resolve_by_name(
        self,
        employees: EmployeeRepository,
        auth: AuthContext,
        *,
        name: str,
        department: str | None = None,
    ) -> ToolResult:
        clean = re.sub(
            r"\b(?:in|from)\s+(?:Engineering|People|Sales|Finance|Product|Operations)\b",
            "",
            name or "",
            flags=re.I,
        ).strip(" .,?!")
        matches = await employees.search_by_name(clean, limit=15)
        # Prefer exact full-name matches when available
        exact = [
            m
            for m in matches
            if m.full_name.lower() == clean.lower()
            or f"{m.first_name} {m.last_name}".lower() == clean.lower()
        ]
        pool = exact or matches
        if department:
            dept_pool = [m for m in pool if m.department.lower() == department.lower()]
            if dept_pool:
                pool = dept_pool
        accessible = [
            m
            for m in pool
            if can_access_employee(auth, target_id=str(m.id), target_department=m.department)
        ]
        # Deduplicate by id (search can theoretically overlap)
        seen: set[str] = set()
        unique: list = []
        for e in accessible:
            eid = str(e.id)
            if eid in seen:
                continue
            seen.add(eid)
            unique.append(e)
        accessible = unique

        if not accessible:
            return ToolResult(data={"employees": []}, confidence=0.2)
        if len(accessible) > 1:
            options = [_option_label(e) for e in accessible[:8]]
            return ToolResult(
                data={
                    "clarify": (
                        f"I found multiple people named like '{clean}'. "
                        f"Which one did you mean:\n"
                        + "\n".join(f"- {opt}" for opt in options)
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

    def _serialize(self, auth: AuthContext, emp) -> dict[str, Any]:
        cols = allowed_columns(
            auth.role, department=auth.department_id, target_department=emp.department
        )
        payload = emp.model_dump(mode="json")
        payload["full_name"] = emp.full_name
        return {k: v for k, v in payload.items() if k in cols or k == "full_name"}


def _dept_from_text(text: str) -> str | None:
    m = _DEPT_HINT_RE.search(text or "")
    return m.group(1) if m else None
