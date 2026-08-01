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
    answer = result.data["answer"].lower()
    assert result.data["intent"] == "hello"
    assert "help" in answer
    assert any(w in answer for w in ("hello", "hi", "hey"))


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
    assert "array_position" not in sql  # count-only needs no display order


def test_query_builder_preserves_employee_id_order() -> None:
    id_a = "00000000-0000-0000-0000-00000000000a"
    id_b = "00000000-0000-0000-0000-00000000000b"
    sql, params = QueryBuilder().build(
        columns=["id", "first_name", "last_name"],
        filters={"employee_ids": [id_a, id_b]},
    )
    assert "array_position" in sql
    assert params["eid_0"] == id_a
    assert params["eid_1"] == id_b
    # ORDER BY must appear before LIMIT
    assert sql.upper().index("ORDER BY") < sql.upper().index("LIMIT")


def test_query_builder_filters_before_order_by() -> None:
    """Regression: ORDER BY mid-WHERE made Postgres treat array_position as AND arg."""
    id_a = "00000000-0000-0000-0000-00000000000a"
    sql, _ = QueryBuilder().build(
        columns=["id", "first_name", "last_name", "department", "position"],
        filters={"employee_ids": [id_a], "department": "Engineering"},
    )
    assert "AND e.department = :department" in sql
    assert sql.index("AND e.department") < sql.index("ORDER BY")
    assert sql.index("ORDER BY") < sql.upper().index("LIMIT")


def test_query_builder_count_distinct_department() -> None:
    sql, params = QueryBuilder().build(
        columns=["department"],
        filters={},
        count_distinct="department",
    )
    assert "COUNT(DISTINCT e.department)" in sql
    assert "JOIN" not in sql
    assert "LIMIT" not in sql.upper()
    assert params == {}


def test_query_builder_distinct_department() -> None:
    sql, _ = QueryBuilder().build(
        columns=["department"],
        filters={},
        distinct=True,
    )
    assert "SELECT DISTINCT e.department" in sql
    assert "ORDER BY e.department" in sql
    assert "LIMIT" in sql.upper()


@pytest.mark.parametrize("field", ["city", "country"])
def test_query_builder_cannot_touch_resume_sourced_fields(field: str) -> None:
    """Location is not a column, so the builder has no way to express it."""
    builder = QueryBuilder()
    with pytest.raises(ValueError, match="Unsupported filters"):
        builder.build(columns=["id"], filters={field: "Berlin"})
    with pytest.raises(ValueError, match="Unsupported count_distinct"):
        builder.build(columns=[field], filters={}, count_distinct=field)
    with pytest.raises(ValueError, match="Unsupported distinct column"):
        builder.build(columns=[field], filters={}, distinct=True)


def test_sql_validator_rejects_the_resumes_table() -> None:
    """The last route to a resume-sourced column through generated SQL."""
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    with pytest.raises(ValidationFailedError):
        validate_sql("SELECT city FROM resumes", auth)
    with pytest.raises(ValidationFailedError):
        validate_sql(
            "SELECT e.id FROM employees e JOIN resumes r ON r.employee_id = e.id", auth
        )


def test_query_builder_status_bool() -> None:
    sql, params = QueryBuilder().build(
        columns=["id"],
        filters={"status": "true"},
        count_only=True,
    )
    assert "e.status = :status" in sql
    assert params["status"] is True
    assert "JOIN resumes" not in sql


def test_query_builder_no_join_without_location() -> None:
    sql, _ = QueryBuilder().build(
        columns=["id", "first_name"],
        filters={"department": "Engineering"},
    )
    assert "JOIN resumes" not in sql
    assert "e.department = :department" in sql


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
