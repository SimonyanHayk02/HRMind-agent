from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanNode(BaseModel):
    id: str
    kind: Literal["tool", "operator"]
    name: str
    input_bindings: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    version: Literal["1"] = "1"
    nodes: list[PlanNode]
    response_strategy: Literal["template", "llm_format"] = "llm_format"
    clarify_question: str | None = None
    # Node id whose output defines the active "them" cohort for the next turn.
    active_cohort_node: str | None = None
