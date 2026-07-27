from __future__ import annotations

import asyncio

from app.application.execution.graph_state import GraphState
from app.application.execution.node_runner import NodeRunner
from app.application.planning.plan_schema import ExecutionPlan
from app.domain.auth import AuthContext
from app.domain.session import EntityRef


class LangGraphExecutor:
    """DAG executor (LangGraph-compatible naming). Runs ready nodes in parallel waves."""

    def __init__(self, node_runner: NodeRunner) -> None:
        self._runner = node_runner

    async def execute(
        self,
        plan: ExecutionPlan,
        *,
        question: str,
        auth: AuthContext,
        session_entities: list[EntityRef] | None = None,
        last_employee_ids: list[str] | None = None,
        person_bindings: dict[str, str] | None = None,
    ) -> GraphState:
        state = GraphState(
            question=question,
            auth=auth,
            session_entities=list(session_entities or []),
            last_employee_ids=list(last_employee_ids or []),
            person_bindings=dict(person_bindings or {}),
        )
        pending = {n.id: n for n in plan.nodes}
        completed: set[str] = set()

        while pending:
            ready = [
                n
                for n in pending.values()
                if all(dep in completed for dep in n.depends_on)
            ]
            if not ready:
                state.errors.append("Deadlock in plan execution")
                break

            results = await asyncio.gather(
                *[self._runner.run_node(n, state) for n in ready], return_exceptions=False
            )
            for node, result in zip(ready, results):
                state.node_results[node.id] = result
                completed.add(node.id)
                pending.pop(node.id)
        return state
