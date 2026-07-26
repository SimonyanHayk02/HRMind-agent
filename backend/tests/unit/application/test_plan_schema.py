from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.application.planning.plan_schema import ExecutionPlan


def test_execution_plan_coerces_null_optional_fields() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "version": "1",
            "nodes": [
                {
                    "id": 1,
                    "kind": "Tool",
                    "name": "sql",
                    "depends_on": None,
                    "input_bindings": None,
                    "params": None,
                }
            ],
            "response_strategy": None,
        }
    )
    assert plan.nodes[0].id == "1"
    assert plan.nodes[0].kind == "tool"
    assert plan.nodes[0].depends_on == []
    assert plan.nodes[0].input_bindings == {}
    assert plan.nodes[0].params == {}
    assert plan.response_strategy == "llm_format"


def test_execution_plan_rejects_bad_kind() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(
            {
                "nodes": [{"id": "x", "kind": "magic", "name": "sql"}],
            }
        )
