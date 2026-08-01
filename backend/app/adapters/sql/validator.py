from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from app.domain.auth import AuthContext
from app.domain.errors import ValidationFailedError
from app.domain.policies.column_policy import allowed_columns

FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"}
# Generated SQL reads employees and nothing else. Resume-sourced facts (location,
# birth date, skills) are retrieved from the chunk corpus, so neither `resumes`
# nor `resume_chunks` is reachable from here.
ALLOWED_TABLES = {"employees"}


def validate_sql(sql: str, auth: AuthContext, *, max_rows: int = 200) -> str:
    stripped = sql.strip().rstrip(";")
    upper = stripped.upper()
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", upper):
            raise ValidationFailedError(f"Forbidden SQL keyword: {word}")

    try:
        parsed = sqlglot.parse_one(stripped, read="postgres")
    except Exception as exc:
        raise ValidationFailedError(f"SQL parse error: {exc}") from exc

    if not isinstance(parsed, exp.Select):
        raise ValidationFailedError("Only SELECT statements are allowed")

    tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
    if not tables.issubset(ALLOWED_TABLES):
        raise ValidationFailedError(f"Table not allowed: {tables - ALLOWED_TABLES}")

    cols = allowed_columns(auth.role, department=auth.department_id, target_department=auth.department_id)
    # SELECT * bypasses column ACL — reject and require an explicit projection.
    if parsed.find(exp.Star):
        raise ValidationFailedError(
            "SELECT * is not allowed; project explicit columns only"
        )
    for col in parsed.find_all(exp.Column):
        name = col.name.lower()
        if name not in cols and name not in {"id", "count"}:
            # allow aliases like count
            if name not in cols:
                raise ValidationFailedError(f"Column not allowed for role: {name}")

    # Ensure LIMIT
    if parsed.args.get("limit") is None:
        stripped = f"{stripped} LIMIT {max_rows}"
    else:
        # clamp limit
        limit_exp = parsed.args["limit"]
        try:
            val = int(limit_exp.expression.this)
            if val > max_rows:
                stripped = re.sub(r"LIMIT\s+\d+", f"LIMIT {max_rows}", stripped, flags=re.I)
        except Exception:
            stripped = f"{stripped} LIMIT {max_rows}"

    return stripped
