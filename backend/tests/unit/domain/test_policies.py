from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.policies.column_policy import allowed_columns, is_column_allowed
from app.domain.policies.rbac import can_access_employee, can_use_tool


def test_employee_cannot_see_salary() -> None:
    cols = allowed_columns(Role.EMPLOYEE)
    assert "salary" not in cols
    assert not is_column_allowed(Role.EMPLOYEE, "salary")


def test_recruiter_can_see_salary() -> None:
    assert "salary" in allowed_columns(Role.RECRUITER)


def test_manager_salary_own_dept_only() -> None:
    assert "salary" in allowed_columns(
        Role.MANAGER, department="Engineering", target_department="Engineering"
    )
    assert "salary" not in allowed_columns(
        Role.MANAGER, department="Engineering", target_department="Sales"
    )


def test_tool_permissions() -> None:
    emp = AuthContext(user_id="u", tenant_id="t", role=Role.EMPLOYEE, employee_id="1")
    assert can_use_tool(emp, "greeting")
    assert not can_use_tool(emp, "sql")
    assert not can_use_tool(emp, "resume_search")


def test_access_employee() -> None:
    auth = AuthContext(
        user_id="u", tenant_id="t", role=Role.EMPLOYEE, employee_id="abc"
    )
    assert can_access_employee(auth, target_id="abc", target_department="Engineering")
    assert not can_access_employee(auth, target_id="zzz", target_department="Engineering")
