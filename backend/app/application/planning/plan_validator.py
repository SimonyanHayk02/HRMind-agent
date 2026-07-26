from __future__ import annotations

from app.application.planning.plan_schema import ExecutionPlan
from app.domain.auth import AuthContext
from app.domain.errors import PlanInvalidError
from app.domain.operators.base import OperatorRegistry
from app.domain.policies.rbac import can_use_tool
from app.domain.tools.registry import ToolRegistry


class PlanValidator:
    def __init__(
        self,
        tools: ToolRegistry,
        operators: OperatorRegistry,
        *,
        max_nodes: int = 12,
    ) -> None:
        self._tools = tools
        self._operators = operators
        self._max_nodes = max_nodes

    def validate(self, plan: ExecutionPlan, auth: AuthContext) -> None:
        if not plan.nodes:
            if plan.clarify_question:
                return
            raise PlanInvalidError("Plan has no nodes")
        if len(plan.nodes) > self._max_nodes:
            raise PlanInvalidError("Plan exceeds max nodes")

        ids = [n.id for n in plan.nodes]
        if len(ids) != len(set(ids)):
            raise PlanInvalidError("Duplicate node ids")

        id_set = set(ids)
        for node in plan.nodes:
            for dep in node.depends_on:
                if dep not in id_set:
                    raise PlanInvalidError(f"Unknown dependency {dep}")
            if node.kind == "tool":
                if node.name not in self._tools.names():
                    raise PlanInvalidError(f"Unknown tool {node.name}")
                if not can_use_tool(auth, node.name):
                    raise PlanInvalidError(f"Tool not permitted: {node.name}")
            elif node.kind == "operator":
                if node.name not in self._operators.names():
                    raise PlanInvalidError(f"Unknown operator {node.name}")
            else:
                raise PlanInvalidError(f"Invalid kind {node.kind}")

        # cycle detection
        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {n.id: n.depends_on for n in plan.nodes}

        def dfs(node_id: str) -> None:
            if node_id in visiting:
                raise PlanInvalidError("Plan contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dep in graph[node_id]:
                dfs(dep)
            visiting.remove(node_id)
            visited.add(node_id)

        for nid in ids:
            dfs(nid)
