from __future__ import annotations

from typing import Any


class ProjectFields:
    name = "project_fields"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> Any:
        fields = (params or {}).get("fields") or []
        if isinstance(data, list):
            return [{k: item.get(k) for k in fields if isinstance(item, dict)} for item in data]
        if isinstance(data, dict):
            if "rows" in data and isinstance(data["rows"], list):
                return {
                    **data,
                    "rows": [{k: row.get(k) for k in fields} for row in data["rows"] if isinstance(row, dict)],
                }
            return {k: data.get(k) for k in fields}
        return data
