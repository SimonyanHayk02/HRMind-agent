from __future__ import annotations

from typing import Any


class ExtractEmployeeIds:
    name = "extract_employee_ids"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[str]:
        ids: list[str] = []
        if data is None:
            return ids
        if isinstance(data, dict):
            if "employee_id" in data:
                ids.append(str(data["employee_id"]))
            if "employee_ids" in data:
                ids.extend(str(x) for x in data["employee_ids"])
            if "hits" in data and isinstance(data["hits"], list):
                for hit in data["hits"]:
                    if isinstance(hit, dict) and "employee_id" in hit:
                        ids.append(str(hit["employee_id"]))
            if "rows" in data and isinstance(data["rows"], list):
                for row in data["rows"]:
                    if isinstance(row, dict) and "id" in row:
                        ids.append(str(row["id"]))
                    if isinstance(row, dict) and "employee_id" in row:
                        ids.append(str(row["employee_id"]))
        elif isinstance(data, list):
            for item in data:
                ids.extend(self.run(item, params))
        # unique preserve order
        seen: set[str] = set()
        out: list[str] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out
