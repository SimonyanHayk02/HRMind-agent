from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.application.planning.heuristic_planner import try_heuristic_plan
from app.application.planning.plan_schema import ExecutionPlan
from app.application.schema.catalog_service import CatalogService
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.plan_from_query_state import plan_from_query_state
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.query_state import QueryState
from app.domain.schema_catalog import default_employee_catalog
from app.domain.session import SessionMemory
from app.domain.tools.registry import ToolRegistry
from app.ports.llm import LLMClient

logger = get_logger(__name__)

_CLARIFY_FALLBACK = (
    "I couldn't build a reliable plan for that. "
    "Try asking with a department, country, city, or job title "
    "(for example: list Software Engineers in Engineering)."
)


class PlanCompiler:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        prompts_dir: Path | None = None,
        *,
        max_context: int = 8,
        catalog_service: CatalogService | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._prompts_dir = prompts_dir or Path(__file__).resolve().parents[2] / "prompts"
        self._max_context = max_context
        self._catalog = catalog_service

    def _load_prompt(self) -> str:
        path = self._prompts_dir / "planner.md"
        if path.exists():
            return path.read_text()
        return (
            "You are an HR agent planner. Return JSON ExecutionPlan with nodes "
            "(tool|operator), depends_on, response_strategy."
        )

    def _catalog_view(self) -> dict:
        if self._catalog is not None:
            return self._catalog.planner_view()
        return default_employee_catalog().planner_view()

    def _context_packet(
        self,
        question: str,
        auth: AuthContext,
        memory: SessionMemory | None,
        query_state: QueryState | None = None,
    ) -> dict:
        recent: list[dict[str, str]] = []
        entities: list[dict] = []
        constraints: list[dict] = []
        last_ids: list[str] = []
        summary = ""
        last_focus: dict | None = None
        if memory:
            for msg in memory.messages[-self._max_context :]:
                recent.append({"role": msg.role, "content": msg.content[:800]})
            entities = [e.model_dump(mode="json") for e in memory.entity_memory[-20:]]
            constraints = [c.model_dump(mode="json") for c in memory.constraint_memory]
            last_ids = list(memory.last_employee_ids)
            summary = memory.summary or ""
            if memory.last_focus:
                last_focus = memory.last_focus.model_dump(mode="json")
        return {
            "question": question,
            "role": auth.role.value,
            "tools": self._tools.discover(),
            "schema_catalog": self._catalog_view(),
            "query_state": query_state.model_dump(mode="json") if query_state else None,
            "recent_messages": recent,
            "last_employee_ids": last_ids,
            "last_focus": last_focus,
            "constraints": constraints,
            "entities": entities,
            "summary": summary,
        }

    async def compile(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None = None,
    ) -> ExecutionPlan:
        catalog = self._catalog.get() if self._catalog else default_employee_catalog()
        query_state = extract_query_state(question, catalog=catalog, memory=memory)

        # Prefer QueryState even at moderate confidence when filters are grounded.
        structured = plan_from_query_state(query_state, memory=memory)
        if structured is None and query_state.filters:
            structured = plan_from_query_state(
                query_state, memory=memory, min_confidence=0.5
            )
        if structured is not None:
            return structured

        heuristic = try_heuristic_plan(question, memory=memory)
        if heuristic is not None:
            return heuristic

        user = json.dumps(
            self._context_packet(question, auth, memory, query_state), default=str
        )
        try:
            raw = await self._llm.complete(
                system=self._load_prompt(), user=user, temperature=0.0, response_json=True
            )
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)
            return ExecutionPlan.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "planner_llm_invalid_plan",
                question=question[:200],
                error_type=type(exc).__name__,
                error=str(exc)[:400],
            )
            return ExecutionPlan(
                nodes=[],
                response_strategy="template",
                clarify_question=_CLARIFY_FALLBACK,
            )
