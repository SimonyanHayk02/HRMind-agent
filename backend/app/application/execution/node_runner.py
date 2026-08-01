from __future__ import annotations

import asyncio
from typing import Any

from app.application.execution.circuit_breaker import CircuitBreaker
from app.application.execution.graph_state import GraphState
from app.application.planning.plan_schema import PlanNode
from app.domain.operators.base import OperatorRegistry
from app.domain.tools.base import ToolResult
from app.domain.tools.registry import ToolRegistry


def _resolve_path(state: GraphState, path: str) -> Any:
    if path == "question":
        return state.question
    if path.startswith("nodes."):
        # nodes.<id> or nodes.<id>.data...
        parts = path.split(".")
        node_id = parts[1]
        value = state.node_results.get(node_id)
        if isinstance(value, ToolResult):
            cur: Any = value.data
        else:
            cur = value
        for p in parts[2:]:
            if p == "data" and isinstance(value, ToolResult) and len(parts) == 3:
                return value.data
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        return cur
    return path


def _as_id_list(raw: Any) -> list[str]:
    """Normalise bound ids from a list, UUID, or employee/tool payload dict."""
    if raw is None:
        return []
    if isinstance(raw, (str, int)):
        return [str(raw)] if str(raw).strip() else []
    if isinstance(raw, dict):
        # Employee profile payloads expose `id`, not `employee_ids`.
        if raw.get("id") is not None and str(raw.get("id")).strip():
            return [str(raw["id"])]
        emp = raw.get("employee")
        if isinstance(emp, dict) and emp.get("id") is not None:
            return [str(emp["id"])]
        if isinstance(raw.get("employee_ids"), list):
            return _as_id_list(raw["employee_ids"])
        if raw.get("employee_id") is not None and str(raw.get("employee_id")).strip():
            return [str(raw["employee_id"])]
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if isinstance(x, dict):
                out.extend(_as_id_list(x))
            else:
                s = str(x).strip()
                if s:
                    out.append(s)
        return out
    return []


class NodeRunner:
    def __init__(
        self,
        tools: ToolRegistry,
        operators: OperatorRegistry,
        *,
        timeout_ms: int = 15000,
    ) -> None:
        self._tools = tools
        self._operators = operators
        self._timeout = timeout_ms / 1000
        self._breakers: dict[str, CircuitBreaker] = {}

    def _breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker()
        return self._breakers[name]

    async def run_node(self, node: PlanNode, state: GraphState) -> Any:
        # Resolve bindings into params
        params = dict(node.params)
        for key, path in node.input_bindings.items():
            params[key] = _resolve_path(state, path)
        if "question" not in params:
            params.setdefault("question", state.question)

        # Inject session context so tools can recover cohort / entity refs
        if node.name == "employee" and state.session_entities:
            params.setdefault(
                "entity_memory",
                [e.model_dump(mode="json") for e in state.session_entities],
            )
            if state.person_bindings:
                params.setdefault("person_bindings", state.person_bindings)
        if node.name in {"sql", "resume_search"}:
            filters = dict(params.get("filters") or {})
            if "employee_ids" in node.input_bindings:
                raw = params.get("employee_ids")
                bound_ids = _as_id_list(raw)
                # Empty binding ⇒ empty cohort (QueryBuilder AND 1=0), never org-wide.
                params["employee_ids"] = bound_ids
                filters["employee_ids"] = bound_ids
                params["filters"] = filters
            elif params.get("use_session_cohort") and state.last_employee_ids:
                # Explicit recovery only when the plan asks for the session cohort.
                params["employee_ids"] = list(state.last_employee_ids)
                filters["employee_ids"] = list(state.last_employee_ids)
                params["filters"] = filters

        if node.kind == "operator":
            op = self._operators.get(node.name)
            # Prefer bound "data" else first dependency result
            data = params.get("data")
            if data is None and node.depends_on:
                prev = state.node_results.get(node.depends_on[0])
                data = prev.data if isinstance(prev, ToolResult) else prev
            try:
                return op.run(data, params)
            except Exception as exc:
                state.degraded = True
                state.errors.append(str(exc))
                return ToolResult(
                    data=None, confidence=0.0, error=str(exc), degraded=True
                )

        tool = self._tools.get(node.name)
        breaker = self._breaker(node.name)
        if not breaker.allow():
            result = ToolResult(data=None, confidence=0.0, error="circuit_open", degraded=True)
            state.degraded = True
            return result
        try:
            result = await asyncio.wait_for(tool.run(params, auth=state.auth), timeout=self._timeout)
            if result.error:
                breaker.record_failure()
                state.degraded = state.degraded or result.degraded
                # Do not let failed tools keep usable-looking payloads.
                if result.data is not None and not result.degraded:
                    result = ToolResult(
                        data=None,
                        confidence=0.0,
                        error=result.error,
                        degraded=True,
                        sources=result.sources,
                    )
            else:
                breaker.record_success()
            return result
        except Exception as exc:
            breaker.record_failure()
            state.degraded = True
            state.errors.append(str(exc))
            return ToolResult(data=None, confidence=0.0, error=str(exc), degraded=True)
