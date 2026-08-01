from __future__ import annotations

from app.application.planning.plan_schema import ExecutionPlan
from app.domain.auth import AuthContext
from app.domain.errors import PlanInvalidError
from app.domain.operators.base import OperatorRegistry
from app.domain.policies.rbac import can_use_tool
from app.domain.schema_catalog import resume_sourced_fields
from app.domain.tools.registry import ToolRegistry
_ALLOWED_SQL_MODES = frozenset({"constrained", "nl2sql"})
_ALLOWED_EMPLOYEE_ACTIONS = frozenset(
    {
        "profile",
        "manager",
        "reports",
        "department",
        "by_email",
        "by_id",
        "by_name",
        "set_status",
    }
)
_ALLOWED_RESUME_PURPOSES = frozenset(
    {
        "birthday_person",
        "birthday_cohort",
        "location_person",
        "location_cohort",
        "location_facet",
        "languages_person",
        "languages_cohort",
        "certifications_person",
        "certifications_cohort",
        "status_resolve",
    }
)


def _sql_fields(params: dict) -> set[str]:
    """Every column a constrained sql node would put into its statement."""
    fields = {str(f) for f in (params.get("filters") or {})}
    fields |= {str(c) for c in (params.get("columns") or [])}
    if params.get("count_distinct"):
        fields.add(str(params["count_distinct"]))
    return fields


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
        self._resume_sourced = resume_sourced_fields()

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
                # Resume-sourced facts must be retrieved, not queried. Enforced
                # here because every planner tier — including the LLM — ends up
                # in this loop, and failing at planning time beats a tool error
                # halfway through execution.
                params = node.params or {}
                if node.name == "sql":
                    mode = str(params.get("mode") or "constrained")
                    if mode not in _ALLOWED_SQL_MODES:
                        raise PlanInvalidError(f"Unknown sql mode: {mode}")
                    leaked = _sql_fields(params) & self._resume_sourced
                    if leaked:
                        raise PlanInvalidError(
                            "Resume-sourced field cannot be queried in SQL: "
                            f"{sorted(leaked)}. Use resume_search instead."
                        )
                if node.name == "employee":
                    action = str(params.get("action") or "profile")
                    if action not in _ALLOWED_EMPLOYEE_ACTIONS:
                        raise PlanInvalidError(f"Unknown employee action: {action}")
                if node.name == "resume_search":
                    purpose = params.get("purpose")
                    if purpose is not None and str(purpose) not in _ALLOWED_RESUME_PURPOSES:
                        # Empty/missing purpose is generic semantic RAG (allowed).
                        if str(purpose).strip():
                            raise PlanInvalidError(
                                f"Unknown resume_search purpose: {purpose}"
                            )
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
