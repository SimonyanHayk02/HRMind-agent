import pytest

from app.adapters.sql.query_builder import QueryBuilder
from app.adapters.sql.validator import validate_sql
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import ValidationFailedError
from app.tools.greeting.tool import GreetingTool


@pytest.mark.asyncio
async def test_greeting_hello() -> None:
    tool = GreetingTool()
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    result = await tool.run({"message": "hello"}, auth=auth)
    assert "Hello" in result.data["answer"]


def test_query_builder_ids() -> None:
    sql, params = QueryBuilder().build(
        columns=["id", "first_name"],
        filters={"employee_ids": ["00000000-0000-0000-0000-000000000099"], "hire_date_gt": "2023-01-01"},
        count_only=True,
    )
    assert "COUNT(*)" in sql
    assert "eid_0" in params
    assert params["eid_0"] == "00000000-0000-0000-0000-000000000099"
    assert ":eid_0" in sql
    assert "hire_date_gt" in params


def test_query_builder_count_distinct_country() -> None:
    sql, params = QueryBuilder().build(
        columns=["country"],
        filters={},
        count_distinct="country",
    )
    assert "COUNT(DISTINCT e.country)" in sql
    assert "LIMIT" not in sql.upper()
    assert params == {}


def test_query_builder_distinct_country() -> None:
    sql, params = QueryBuilder().build(
        columns=["country"],
        filters={},
        distinct=True,
    )
    assert "SELECT DISTINCT e.country" in sql
    assert "ORDER BY e.country" in sql
    assert "LIMIT" in sql.upper()


def test_sql_validator_rejects_drop() -> None:
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    with pytest.raises(ValidationFailedError):
        validate_sql("DROP TABLE employees", auth)


def test_sql_validator_adds_limit() -> None:
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    sql = validate_sql("SELECT id, first_name FROM employees", auth, max_rows=50)
    assert "LIMIT 50" in sql.upper()


def test_sql_validator_blocks_salary_for_employee_role_via_column() -> None:
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.EMPLOYEE, employee_id="1")
    with pytest.raises(ValidationFailedError):
        validate_sql("SELECT salary FROM employees", auth)
