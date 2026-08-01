from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FilterSlot(BaseModel):
    field: str
    op: Literal["eq", "in", "contains", "gt", "gte", "lt", "lte"] = "eq"
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
        "reports",
        "set_status",
        "birthday",
        "languages",
        "certifications",
        "tenure_agg",
        "longest_tenured",
        "clarify",
        "unsupported",
        "unknown",
    ] = "unknown"
    refers_to_prior: bool = False
    filters: list[FilterSlot] = Field(default_factory=list)
    facet_dimension: str | None = None
    skill: str | None = None
    person_name: str | None = None
    # When set, person-attribute plans bind to these ids (ordinal / pronoun).
    person_employee_ids: list[str] = Field(default_factory=list)
    # Target value for intent=set_status (employees.status boolean flag).
    status_value: bool | None = None
    status_email: str | None = None
    status_employee_id: str | None = None
    # intent=birthday: dates exist only in resume text, so these route to RAG.
    birthday_scope: Literal["person", "today", "month", "upcoming", "closest"] | None = None
    birthday_month: int | None = None
    wants_age: bool = False
    wants_wish: bool = False
    # Resume attribute filters (languages / certifications).
    language: str | None = None
    certification: str | None = None
    hire_year_gt: int | None = None
    want_count: bool = False
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)

    def filters_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self.filters:
            if f.op == "eq":
                out[f.field] = f.value
            elif f.op == "in":
                out[f.field] = list(f.value) if not isinstance(f.value, list) else f.value
            elif f.op == "gt" and f.field == "hire_date":
                out["hire_date_gt"] = f.value
            elif f.op == "gte" and f.field == "hire_date":
                out["hire_date_gte"] = f.value
            elif f.op == "lt" and f.field == "hire_date":
                out["hire_date_lt"] = f.value
        return out
