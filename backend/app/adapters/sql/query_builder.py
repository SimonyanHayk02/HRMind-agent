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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class QueryBuilder:
    """Deterministic constrained SQL for hybrid flows. No LLM.

    Every statement reads the ``employees`` table alone. City and country are
    written in resume documents, so they are resolved by ``resume_search`` and
    arrive here as an ``employee_ids`` filter — never as a column.
    """

    EMPLOYEE_EQ_FILTERS = frozenset(
        {"department", "employment_status", "position", "education"}
    )

    ALLOWED_FILTERS = {
        "employee_ids",
        "hire_date_gt",
        "hire_date_gte",
        "hire_date_lt",
        "department",
        "employment_status",
        "position",
        "education",
        "position_ilike",
        "status",
    }

    ALLOWED_FACETS = frozenset(
        {"department", "position", "employment_status", "education"}
    )

    ALLOWED_TEMPLATES = frozenset(
        {
            "agg_tenure_by_dept",
            "agg_tenure",
            "longest_tenured",
        }
    )

    def _qualify(self, col: str) -> str:
        return f"e.{col}"

    def build(
        self,
        *,
        columns: list[str],
        filters: dict[str, Any],
        max_rows: int = 200,
        count_only: bool = False,
        distinct: bool = False,
        count_distinct: str | None = None,
        template: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if template:
            return self.build_template(
                template, filters=filters, max_rows=max_rows, limit=limit
            )
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
            select_cols = f"COUNT(DISTINCT {self._qualify(count_distinct)}) AS count"
        elif count_only:
            select_cols = "COUNT(*) AS count"
        elif distinct:
            if len(columns) != 1:
                raise ValueError("distinct requires exactly one column")
            if columns[0] not in self.ALLOWED_FACETS:
                raise ValueError(f"Unsupported distinct column: {columns[0]}")
            select_cols = f"DISTINCT {self._qualify(columns[0])}"
        else:
            select_cols = ", ".join(self._qualify(c) for c in columns)

        sql = f"SELECT {select_cols} FROM employees e WHERE 1=1"
        params: dict[str, Any] = {}
        # ORDER BY must be emitted after every AND filter — never mid-WHERE.
        id_order_keys: list[str] = []
        sql, params, id_order_keys = self._apply_filters(sql, params, filters, id_order_keys)

        if not count_only and count_distinct is None:
            # Preserve cohort order so bullets match "first/second".
            if id_order_keys and not distinct:
                sql += (
                    " ORDER BY array_position(ARRAY["
                    + ", ".join(id_order_keys)
                    + "]::uuid[], e.id)"
                )
            elif distinct and columns:
                sql += f" ORDER BY {self._qualify(columns[0])}"
            sql += f" LIMIT {int(max_rows)}"
        return sql, params

    def build_template(
        self,
        template: str,
        *,
        filters: dict[str, Any] | None = None,
        max_rows: int = 200,
        limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Allowlisted analytics statements — never free-form SQL."""
        if template not in self.ALLOWED_TEMPLATES:
            raise ValueError(f"Unsupported SQL template: {template}")
        filters = dict(filters or {})
        bad = set(filters) - self.ALLOWED_FILTERS
        if bad:
            raise ValueError(f"Unsupported filters: {bad}")
        params: dict[str, Any] = {}

        if template == "agg_tenure_by_dept":
            sql = (
                "SELECT e.department AS department, "
                "AVG(CURRENT_DATE - e.hire_date) AS avg_tenure_days, "
                "COUNT(*) AS headcount "
                "FROM employees e WHERE 1=1"
            )
            sql, params, _ = self._apply_filters(sql, params, filters, [])
            sql += " GROUP BY e.department ORDER BY e.department"
            return sql, params

        if template == "agg_tenure":
            sql = (
                "SELECT AVG(CURRENT_DATE - e.hire_date) AS avg_tenure_days, "
                "COUNT(*) AS headcount "
                "FROM employees e WHERE 1=1"
            )
            sql, params, _ = self._apply_filters(sql, params, filters, [])
            return sql, params

        # longest_tenured — proxy for "most senior" by earliest hire_date.
        k = int(limit or max_rows or 5)
        k = max(1, min(k, 50))
        sql = (
            "SELECT e.id, e.first_name, e.last_name, e.department, e.position, "
            "e.hire_date, (CURRENT_DATE - e.hire_date) AS tenure_days "
            "FROM employees e WHERE 1=1"
        )
        sql, params, _ = self._apply_filters(sql, params, filters, [])
        sql += f" ORDER BY e.hire_date ASC NULLS LAST LIMIT {k}"
        return sql, params

    def _apply_filters(
        self,
        sql: str,
        params: dict[str, Any],
        filters: dict[str, Any],
        id_order_keys: list[str],
    ) -> tuple[str, dict[str, Any], list[str]]:
        if "employee_ids" in filters:
            raw_ids = filters["employee_ids"] or []
            if not raw_ids:
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
                        id_order_keys.append(f"CAST(:{key} AS uuid)")
                        params[key] = eid
                    sql += " AND e.id IN (" + ", ".join(placeholders) + ")"
                else:
                    sql += " AND 1=0"
        if "hire_date_gt" in filters:
            sql += " AND e.hire_date > :hire_date_gt"
            params["hire_date_gt"] = _as_date(filters["hire_date_gt"])
        if "hire_date_gte" in filters:
            sql += " AND e.hire_date >= :hire_date_gte"
            params["hire_date_gte"] = _as_date(filters["hire_date_gte"])
        if "hire_date_lt" in filters:
            sql += " AND e.hire_date < :hire_date_lt"
            params["hire_date_lt"] = _as_date(filters["hire_date_lt"])

        for col in ("department", "employment_status", "position", "education"):
            if col in filters and filters[col] is not None:
                val = filters[col]
                qualified = self._qualify(col)
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
                        sql += f" AND {qualified} IN (" + ", ".join(placeholders) + ")"
                else:
                    sql += f" AND {qualified} = :{col}"
                    params[col] = val

        if "status" in filters and filters["status"] is not None:
            sql += " AND e.status = :status"
            params["status"] = _as_bool(filters["status"])

        if filters.get("position_ilike"):
            sql += " AND e.position ILIKE :position_ilike"
            params["position_ilike"] = str(filters["position_ilike"])

        return sql, params, id_order_keys
