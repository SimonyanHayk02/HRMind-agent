from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PlanNode(BaseModel):
    id: str
    kind: Literal["tool", "operator"]
    name: str
    input_bindings: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_llm_shape(cls, data: Any) -> Any:
        """Salvage common LLM mistakes before field validation."""
        if not isinstance(data, dict):
            return data
        bindings = data.get("input_bindings")
        params = data.get("params")
        if not isinstance(bindings, dict):
            bindings = {}
        if not isinstance(params, dict):
            params = {} if params is None else {}
        # LLM sometimes nests tool args under input_bindings.params (a dict).
        nested = bindings.get("params")
        if isinstance(nested, dict):
            params = {**nested, **params}
            bindings = {k: v for k, v in bindings.items() if k != "params"}
        data["input_bindings"] = {
            str(k): v for k, v in bindings.items() if isinstance(v, str)
        }
        data["params"] = params
        return data

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: Any) -> Any:
        # LLMs sometimes emit numeric node ids.
        return str(value) if value is not None and not isinstance(value, str) else value

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("input_bindings", mode="before")
    @classmethod
    def _coerce_bindings(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("depends_on", mode="before")
    @classmethod
    def _coerce_depends(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, value: Any) -> Any:
        return {} if value is None else value


class ExecutionPlan(BaseModel):
    version: Literal["1"] = "1"
    nodes: list[PlanNode]
    response_strategy: Literal["template", "llm_format"] = "llm_format"
    clarify_question: str | None = None
    # Node id whose output defines the active "them" cohort for the next turn.
    active_cohort_node: str | None = None
    # Optional planner hint for the runtime refusal resolver (not client-facing).
    refusal_code: str | None = None
    # Optional tool-fact payload for HITL (e.g. pending status confirm).
    pending_tool_fact: dict[str, Any] | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, value: Any) -> Any:
        if value is None:
            return "1"
        return str(value)

    @field_validator("response_strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, value: Any) -> Any:
        if value is None:
            return "llm_format"
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("nodes", mode="before")
    @classmethod
    def _coerce_nodes(cls, value: Any) -> Any:
        return [] if value is None else value
