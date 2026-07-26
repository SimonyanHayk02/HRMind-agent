from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"Cannot convert {value!r} to date")


class QueryBuilder:
    """Deterministic constrained SQL for hybrid flows. No LLM."""

    ALLOWED_FILTERS = {
        "employee_ids",
        "hire_date_gt",
        "hire_date_gte",
        "hire_date_lt",
        "department",
        "country",
        "city",
        "employment_status",
        "position",
        "education",
        "position_ilike",
    }

    ALLOWED_FACETS = frozenset(
        {"department", "country", "city", "position", "employment_status", "education"}
    )

    def build(
        self,
        *,
        columns: list[str],
        filters: dict[str, Any],
        max_rows: int = 200,
        count_only: bool = False,
        distinct: bool = False,
        count_distinct: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        bad = set(filters) - self.ALLOWED_FILTERS
        if bad:
            raise ValueError(f"Unsupported filters: {bad}")
        if count_distinct is not None and count_distinct not in self.ALLOWED_FACETS:
            raise ValueError(f"Unsupported count_distinct column: {count_distinct}")
        if distinct and count_only:
            raise ValueError("distinct and count_only cannot both be true")
        if count_distinct and (count_only or distinct):
            raise ValueError("count_distinct cannot combine with count_only/distinct")

        if count_distinct:
            select_cols = f"COUNT(DISTINCT e.{count_distinct}) AS count"
        elif count_only:
            select_cols = "COUNT(*) AS count"
        elif distinct:
            if len(columns) != 1:
                raise ValueError("distinct requires exactly one column")
            if columns[0] not in self.ALLOWED_FACETS:
                raise ValueError(f"Unsupported distinct column: {columns[0]}")
            select_cols = f"DISTINCT e.{columns[0]}"
        else:
            select_cols = ", ".join(f"e.{c}" for c in columns)

        sql = f"SELECT {select_cols} FROM employees e WHERE 1=1"
        params: dict[str, Any] = {}

        if "employee_ids" in filters:
            raw_ids = filters["employee_ids"] or []
            if not raw_ids:
                # Empty ID list means "no matches" — never fall through to all rows.
                sql += " AND 1=0"
            else:
                ids: list[str] = []
                for x in raw_ids:
                    try:
                        ids.append(str(UUID(str(x))))
                    except Exception:
                        continue
                if ids:
                    placeholders = []
                    for i, eid in enumerate(ids):
                        key = f"eid_{i}"
                        placeholders.append(f"CAST(:{key} AS uuid)")
                        params[key] = eid
                    sql += " AND e.id IN (" + ", ".join(placeholders) + ")"
                # If every provided id was invalid (LLM junk), ignore the filter.
        if "hire_date_gt" in filters:
            sql += " AND e.hire_date > :hire_date_gt"
            params["hire_date_gt"] = _as_date(filters["hire_date_gt"])
        if "hire_date_gte" in filters:
            sql += " AND e.hire_date >= :hire_date_gte"
            params["hire_date_gte"] = _as_date(filters["hire_date_gte"])
        if "hire_date_lt" in filters:
            sql += " AND e.hire_date < :hire_date_lt"
            params["hire_date_lt"] = _as_date(filters["hire_date_lt"])
        for col in ("department", "country", "city", "employment_status", "position", "education"):
            if col in filters and filters[col] is not None:
                val = filters[col]
                if isinstance(val, (list, tuple, set)):
                    vals = [str(v) for v in val if v is not None and str(v).strip()]
                    if not vals:
                        sql += " AND 1=0"
                    else:
                        placeholders = []
                        for i, item in enumerate(vals):
                            key = f"{col}_{i}"
                            placeholders.append(f":{key}")
                            params[key] = item
                        sql += f" AND e.{col} IN (" + ", ".join(placeholders) + ")"
                else:
                    sql += f" AND e.{col} = :{col}"
                    params[col] = val
        if filters.get("position_ilike"):
            sql += " AND e.position ILIKE :position_ilike"
            params["position_ilike"] = str(filters["position_ilike"])

        if not count_only and count_distinct is None:
            if distinct and columns:
                sql += f" ORDER BY e.{columns[0]}"
            sql += f" LIMIT {int(max_rows)}"
        return sql, params
