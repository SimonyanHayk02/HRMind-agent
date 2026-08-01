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


def _status_write_auth(auth: AuthContext) -> AuthContext:
    """Status flag writes are org-wide — use recruiter RLS visibility for the DB session."""
    return AuthContext(
        user_id=auth.user_id,
        tenant_id=auth.tenant_id,
        role=Role.RECRUITER,
        department_id=auth.department_id,
        employee_id=auth.employee_id,
    )


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if key in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
    return None


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


_REPORTS_TO_RE = re.compile(
    r"\b(?:who\s+)?reports?\s+to\s+(?:the\s+)?"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\b",
    re.I,
)
_POSSESSIVE_TEAM_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)'s\s+"
    r"(?:team|direct\s+reports?|reports?)\b",
    re.I,
)
_ON_TEAM_RE = re.compile(
    r"\bon\s+([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)'s\s+team\b",
    re.I,
)


def extract_reports_subject(question: str) -> str | None:
    """Pull the manager name from direct-reports / team phrasings."""
    q = (question or "").strip()
    if not q:
        return None
    for pat in (_ON_TEAM_RE, _POSSESSIVE_TEAM_RE, _REPORTS_TO_RE):
        m = pat.search(q)
        if m:
            return m.group(1).strip(" .,?!")
    return None


def _option_label(emp: Any) -> str:
    bits = [emp.full_name]
    detail = ", ".join(
        x
        for x in (
            getattr(emp, "position", None),
            getattr(emp, "department", None),
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
            description="Lookup employee profile, manager, department roster, or set status flag",
            permissions=list(Role),
            estimated_latency_ms=50,
            input_schema={
                "action": "profile|manager|reports|department|by_email|by_id|by_name|set_status",
                "employee_id": "uuid?",
                "email": "str?",
                "department": "str?",
                "name": "str?",
                "status": "bool?",
            },
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        require_tool(auth, "employee")
        action = params.get("action", "profile")
        # Any role may set the agent status flag on any employee.
        session_auth = _status_write_auth(auth) if action == "set_status" else auth
        async with self._repo_factory(session_auth) as uow:
            employees: EmployeeRepository = uow.employees
            if action == "set_status":
                result = await self._set_status(employees, auth, params)
                if (
                    isinstance(result.data, dict)
                    and result.data.get("updated")
                    and hasattr(uow, "commit")
                ):
                    await uow.commit()
                return result
            if action == "by_name":
                name = params.get("name") or params.get("question") or ""
                dept_hint = params.get("department") or _dept_from_text(name) or _dept_from_text(
                    params.get("question") or ""
                )
                return await self._resolve_by_name(
                    employees,
                    auth,
                    name=name,
                    department=dept_hint,
                    entity_memory=params.get("entity_memory"),
                )
            if action == "by_email":
                emp = await employees.get_by_email(params["email"])
            elif action == "by_id" or action == "profile":
                emp = await employees.get_by_id(UUID(str(params["employee_id"])))
            elif action == "manager":
                return await self._manager(employees, auth, params)
            elif action == "reports":
                return await self._reports(employees, auth, params)
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

    async def _set_status(
        self,
        employees: EmployeeRepository,
        auth: AuthContext,
        params: dict[str, Any],
    ) -> ToolResult:
        status = _as_bool(params.get("status"))
        if status is None:
            return ToolResult(
                data={"clarify": "Should I set the employee's status to true or false?"},
                confidence=0.4,
            )

        name = (params.get("name") or "").strip()
        emp = None
        to_update: list = []

        if params.get("employee_id"):
            emp = await employees.get_by_id(UUID(str(params["employee_id"])))
            if emp:
                to_update = [emp]
        elif params.get("email"):
            emp = await employees.get_by_email(str(params["email"]))
            if emp is None:
                return ToolResult(
                    data={
                        "clarify": (
                            f"I couldn't find an employee with email "
                            f"'{params['email']}'."
                        )
                    },
                    confidence=0.3,
                )
            to_update = [emp]
        else:
            # Prefer RAG-resolved employee_ids (resume_search → extract_employee_ids).
            raw_ids = params.get("employee_ids") or []
            if isinstance(raw_ids, (str, UUID)):
                raw_ids = [raw_ids]
            if not isinstance(raw_ids, list):
                raw_ids = []

            if raw_ids:
                seen: set[str] = set()
                for x in raw_ids:
                    try:
                        eid = UUID(str(x))
                    except Exception:
                        continue
                    key = str(eid)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = await employees.get_by_id(eid)
                    if row is None:
                        continue
                    if name and not _name_matches(row, name):
                        continue
                    to_update.append(row)
                if not to_update:
                    label = name or "that person"
                    return ToolResult(
                        data={
                            "clarify": (
                                f"I searched resumes but couldn't match '{label}'. "
                                "Try a fuller name, or clarify which person you mean."
                            )
                        },
                        confidence=0.3,
                    )
            elif name:
                # Name-based status updates must come through resume RAG, not SQL lookup.
                return ToolResult(
                    data={
                        "clarify": (
                            f"I couldn't resolve '{name}' from resumes. "
                            "Try again or use their email."
                        )
                    },
                    confidence=0.3,
                )
            else:
                return ToolResult(
                    data={"clarify": "Which employee's status should I update?"},
                    confidence=0.4,
                )

        if not to_update:
            return ToolResult(
                data={"clarify": "I couldn't find that employee."},
                confidence=0.3,
            )

        flag = "true" if status else "false"
        resolve_via = params.get("resolve_via") or ""
        # Location cohort writes need an explicit confirm before any DB write.
        if (
            resolve_via == "location"
            and len(to_update) > 1
            and not params.get("confirmed")
        ):
            return ToolResult(
                data={
                    "clarify": (
                        f"This would update status to {flag} for "
                        f"{len(to_update)} employee(s) matched by location. "
                        "Reply with 'confirm status update' if you want me to proceed."
                    ),
                    "pending_status": status,
                    "employee_ids": [str(r.id) for r in to_update],
                },
                confidence=0.45,
            )

        updated_rows = []
        for row in to_update:
            updated = await employees.update_status(row.id, status)
            if updated is not None:
                updated_rows.append(updated)
        if not updated_rows:
            raise NotFoundError("Employee not found")

        if len(updated_rows) == 1:
            u = updated_rows[0]
            payload = self._serialize(auth, u)
            payload["status"] = bool(u.status)
            payload["full_name"] = u.full_name
            payload["updated"] = True
            payload["answer"] = f"Updated {u.full_name}'s status to {flag}."
            return ToolResult(
                data=payload,
                confidence=1.0,
                sources=[SourceRef(kind="employee", ref=str(u.id), label=u.full_name)],
            )

        if resolve_via == "location":
            answer = (
                f"Updated status to {flag} for {len(updated_rows)} employee(s) "
                f"matched via resumes by location."
            )
        else:
            names = ", ".join(sorted({r.full_name for r in updated_rows}))
            answer = (
                f"Updated status to {flag} for {len(updated_rows)} "
                f"employee(s) named {names} (matched via resumes)."
            )
        return ToolResult(
            data={
                "updated": True,
                "status": status,
                "employee_ids": [str(r.id) for r in updated_rows],
                "answer": answer,
            },
            confidence=1.0,
            sources=[
                SourceRef(kind="employee", ref=str(r.id), label=r.full_name)
                for r in updated_rows
            ],
        )

    async def _reports(
        self, employees: EmployeeRepository, auth: AuthContext, params: dict[str, Any]
    ) -> ToolResult:
        """List direct reports for a manager (by id or name)."""
        manager = None
        if params.get("employee_id"):
            manager = await employees.get_by_id(UUID(str(params["employee_id"])))
        else:
            name = (params.get("name") or params.get("question") or "").strip()
            if not name:
                return ToolResult(
                    data={"clarify": "Whose direct reports should I list?"},
                    confidence=0.4,
                )
            resolved = await self._resolve_by_name(
                employees,
                auth,
                name=name,
                department=params.get("department") or _dept_from_text(name),
                entity_memory=params.get("entity_memory"),
            )
            if resolved.data and isinstance(resolved.data, dict) and resolved.data.get("clarify"):
                return resolved
            if resolved.data and isinstance(resolved.data, dict) and resolved.data.get("id"):
                manager = await employees.get_by_id(UUID(str(resolved.data["id"])))
            else:
                return ToolResult(
                    data={"clarify": f"I couldn't find a manager named '{name}'."},
                    confidence=0.3,
                )
        if manager is None:
            raise NotFoundError("Employee not found")
        if not can_access_employee(
            auth, target_id=str(manager.id), target_department=manager.department
        ):
            raise ForbiddenError()
        rows = await employees.list_by_manager(manager.id)
        # No top-level `answer` — composition DAGs must not short-circuit formatting.
        return ToolResult(
            data={
                "manager": self._serialize(auth, manager),
                "employees": [self._serialize(auth, e) for e in rows],
                "employee_ids": [str(e.id) for e in rows],
            },
            confidence=1.0,
            sources=[
                SourceRef(kind="employee", ref=str(manager.id), label=manager.full_name),
                *[
                    SourceRef(kind="employee", ref=str(e.id), label=e.full_name)
                    for e in rows
                ],
            ],
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
                employees,
                auth,
                name=name,
                department=dept,
                entity_memory=params.get("entity_memory"),
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
        entity_memory: list | None = None,
        skip_access_check: bool = False,
        allow_multi_exact_full_name: bool = False,
    ) -> ToolResult:
        clean = re.sub(
            r"\b(?:in|from)\s+(?:Engineering|People|Sales|Finance|Product|Operations)\b",
            "",
            name or "",
            flags=re.I,
        ).strip(" .,?!")

        # Prefer session entity memory via EntityResolver before DB search
        if entity_memory:
            from app.domain.services.entity_resolver import EntityResolver
            from app.domain.session import EntityRef

            refs: list[EntityRef] = []
            for raw in entity_memory:
                try:
                    if isinstance(raw, EntityRef):
                        refs.append(raw)
                    elif isinstance(raw, dict):
                        refs.append(EntityRef.model_validate(raw))
                except Exception:
                    continue
            eid, conf = EntityResolver().resolve_from_refs(clean, refs)
            if eid is not None:
                emp = await employees.get_by_id(eid)
                if emp and (
                    skip_access_check
                    or can_access_employee(
                        auth, target_id=str(emp.id), target_department=emp.department
                    )
                ):
                    return ToolResult(
                        data=self._serialize(auth, emp),
                        confidence=conf,
                        sources=[
                            SourceRef(
                                kind="employee", ref=str(emp.id), label=emp.full_name
                            )
                        ],
                    )

        matches = await employees.search_by_name(clean, limit=15)
        # Prefer exact full-name matches when available
        exact = [
            m
            for m in matches
            if m.full_name.lower() == clean.lower()
            or f"{m.first_name} {m.last_name}".lower() == clean.lower()
        ]
        # Auto-pick only on exact full name; partial hits must clarify or be unique
        # two-token fuzzy matches (never a lone first-name guess from DB).
        tokens = [t for t in clean.split() if t]
        pool = exact if exact else (matches if len(tokens) >= 2 else [])
        if department:
            dept_pool = [m for m in pool if m.department.lower() == department.lower()]
            if dept_pool:
                pool = dept_pool
        if skip_access_check:
            accessible = list(pool)
        else:
            accessible = [
                m
                for m in pool
                if can_access_employee(
                    auth, target_id=str(m.id), target_department=m.department
                )
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
            # First-name-only DB miss: ask rather than guessing a namesake.
            if matches and not exact:
                options = [_option_label(e) for e in matches[:8]]
                return ToolResult(
                    data={
                        "clarify": (
                            f"I found multiple people named like '{clean}'. "
                            f"Which one did you mean:\n"
                            + "\n".join(f"- {opt}" for opt in options)
                        ),
                        "candidates": [self._serialize(auth, e) for e in matches[:8]],
                    },
                    confidence=0.4,
                )
            return ToolResult(data={"employees": []}, confidence=0.2)
        if len(accessible) > 1:
            # For status writes: if every match shares the exact full name, update all.
            if allow_multi_exact_full_name:
                canon = clean.lower()
                exact_all = [
                    e
                    for e in accessible
                    if e.full_name.lower() == canon
                    or f"{e.first_name} {e.last_name}".lower() == canon
                ]
                if len(exact_all) == len(accessible) and exact_all:
                    return ToolResult(
                        data={
                            "multi_ids": [str(e.id) for e in exact_all],
                            "full_name": exact_all[0].full_name,
                        },
                        confidence=0.9,
                    )
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


def _name_matches(emp: Any, name: str) -> bool:
    """Exact full-name match for status writes — never substring fan-out."""
    n = (name or "").strip().lower()
    if not n:
        return True
    full = f"{getattr(emp, 'first_name', '')} {getattr(emp, 'last_name', '')}".strip().lower()
    display = str(getattr(emp, "full_name", "") or "").strip().lower()
    if not full and not display:
        return False
    return n == full or (bool(display) and n == display)
