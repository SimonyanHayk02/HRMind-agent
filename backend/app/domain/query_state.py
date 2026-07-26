from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FilterSlot(BaseModel):
    field: str
    op: Literal["eq", "contains", "gt", "gte", "lt", "lte"] = "eq"
    value: Any
    confidence: float = 1.0


class QueryState(BaseModel):
    """Structured understanding of one user turn, merged with dialog memory."""

    intent: Literal[
        "count",
        "list",
        "facet_count",
        "facet_list",
        "skill_search",
        "profile",
        "manager",
        "clarify",
        "unsupported",
        "unknown",
    ] = "unknown"
    refers_to_prior: bool = False
    filters: list[FilterSlot] = Field(default_factory=list)
    facet_dimension: str | None = None
    skill: str | None = None
    person_name: str | None = None
    hire_year_gt: int | None = None
    want_count: bool = False
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)

    def filters_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self.filters:
            if f.op == "eq":
                out[f.field] = f.value
            elif f.op == "gt" and f.field == "hire_date":
                out["hire_date_gt"] = f.value
            elif f.op == "gte" and f.field == "hire_date":
                out["hire_date_gte"] = f.value
            elif f.op == "lt" and f.field == "hire_date":
                out["hire_date_lt"] = f.value
        return out
