"""Structured slots from residual LLM NLU — never a free-form ExecutionPlan."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PersonRefKind = Literal["name", "pronoun", "ordinal", "id", "none"]
SlotAttribute = Literal[
    "none",
    "dob",
    "location",
    "profile",
    "manager",
    "status",
    "skill",
    "facet",
    "count",
    "list",
]
SlotIntent = Literal[
    "count",
    "list",
    "facet_count",
    "facet_list",
    "skill_search",
    "profile",
    "manager",
    "set_status",
    "birthday",
    "location_cohort",
    "location_person",
    "clarify",
    "unsupported",
    "unknown",
]


class PersonRef(BaseModel):
    kind: PersonRefKind = "none"
    value: str | None = None
    index: int | None = None  # 1-based when kind=ordinal


class SlotBundle(BaseModel):
    """Compact DST extract for residual turns (after regex/heuristics miss)."""

    intent: SlotIntent = "unknown"
    attribute: SlotAttribute = "none"
    person_ref: PersonRef = Field(default_factory=PersonRef)
    refers_to_prior: bool = False
    skill: str | None = None
    facet_dimension: str | None = None
    # Place hints — resume retrieval only; never SQL column filters.
    city: str | None = None
    country: str | None = None
    department: str | None = None
    position: str | None = None
    status_value: bool | None = None
    want_count: bool = False
    wants_age: bool = False
    wants_wish: bool = False
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)
    source: Literal["llm_slots"] = "llm_slots"

    def model_dump_compact(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
