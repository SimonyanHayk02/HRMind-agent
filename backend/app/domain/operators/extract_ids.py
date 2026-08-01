from __future__ import annotations

from typing import Any

_DEFAULT_MIN_HIT_SCORE = 0.32


class ExtractEmployeeIds:
    name = "extract_employee_ids"

    def run(self, data: Any, params: dict[str, Any] | None = None) -> list[str]:
        params = params or {}
        min_score = float(params.get("min_score", _DEFAULT_MIN_HIT_SCORE))
        # Prefer pre-gated ids from resume_search when present.
        if isinstance(data, dict) and data.get("score_gated") and not data.get("hits"):
            return []
        ids: list[str] = []
        if data is None:
            return ids
        if isinstance(data, dict):
            if "employee_id" in data:
                ids.append(str(data["employee_id"]))
            if "employee_ids" in data and not data.get("hits"):
                ids.extend(str(x) for x in data["employee_ids"])
            if "hits" in data and isinstance(data["hits"], list):
                for hit in data["hits"]:
                    if not isinstance(hit, dict) or "employee_id" not in hit:
                        continue
                    scores = [
                        s.get("score")
                        for s in (hit.get("snippets") or [])
                        if isinstance(s, dict)
                    ]
                    raw_score = hit.get("score")
                    try:
                        best = float(
                            raw_score
                            if raw_score is not None
                            else (max(scores) if scores else min_score)
                        )
                    except (TypeError, ValueError):
                        best = min_score
                    # Structured people blobs without scores (SQL rows) keep their ids.
                    if raw_score is None and not scores:
                        ids.append(str(hit["employee_id"]))
                    elif best >= min_score:
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
