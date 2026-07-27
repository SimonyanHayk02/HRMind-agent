from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.application.planning.heuristic_planner import (
    _META_COUNT_RE,
    _PRONOUN_ONLY_RE,
    _employee_by_id_plan,
    _meta_count_plan,
    _pronoun_clarify_plan,
    _resolve_pronoun_employee_id,
    try_heuristic_plan,
    wants_person_lookup,
)
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

if TYPE_CHECKING:
    from app.application.memory.context_manager import ContextManager

logger = get_logger(__name__)

_CLARIFY_FALLBACK = (
    "I couldn't build a reliable plan for that. "
    "Try asking with a department, country, city, or job title "
    "(for example: list Software Engineers in Engineering)."
)


class PlanCompiler:
    def __init__(
        self,
        llm,
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
        context_manager: ContextManager | None = None,
    ) -> dict:
        if context_manager is not None and memory is not None:
            return context_manager.build_for_planner(
                question=question,
                auth=auth,
                memory=memory,
                tools=self._tools.discover(),
                schema_catalog=self._catalog_view(),
                query_state=query_state,
            )
        # Fallback (tests without ContextManager)
        from app.application.memory.context_budget import build_planner_packet

        return build_planner_packet(
            question=question,
            auth=auth,
            memory=memory,
            tools=self._tools.discover(),
            schema_catalog=self._catalog_view(),
            query_state=query_state.model_dump(mode="json") if query_state else None,
            max_context=self._max_context,
        )

    async def compile(
        self,
        question: str,
        *,
        auth: AuthContext,
        memory: SessionMemory | None = None,
        context_manager: ContextManager | None = None,
    ) -> tuple[ExecutionPlan, str]:
        """Compile a plan. Returns (plan, planner_mode)."""
        q = (question or "").strip()

        # Deterministic escapes that must beat QueryState / nl2sql.
        if _META_COUNT_RE.search(q):
            plan = _meta_count_plan(memory)
            if plan is not None:
                return plan, "heuristic_meta_count"

        pronoun_id = _resolve_pronoun_employee_id(q, memory)
        if _PRONOUN_ONLY_RE.search(q) and wants_person_lookup(q):
            # Bare pronoun person questions — never treat "she"/"he" as a name.
            has_proper_name = bool(
                re.search(
                    r"\bwhere\s+(?:does|is)\s+(?!she\b|he\b|her\b|him\b)([A-Z][a-z]+)",
                    q,
                )
            )
            if not has_proper_name:
                if pronoun_id:
                    return _employee_by_id_plan(pronoun_id), "heuristic_pronoun"
                if memory and len(memory.entity_memory) == 1:
                    return (
                        _employee_by_id_plan(str(memory.entity_memory[0].employee_id)),
                        "heuristic_pronoun",
                    )
                # Prefer the most recently mentioned named person when several exist
                if memory and memory.entity_memory:
                    return (
                        _employee_by_id_plan(str(memory.entity_memory[-1].employee_id)),
                        "heuristic_pronoun",
                    )
                return _pronoun_clarify_plan(), "heuristic_pronoun"

        catalog = self._catalog.get() if self._catalog else default_employee_catalog()
        query_state = extract_query_state(question, catalog=catalog, memory=memory)

        structured = plan_from_query_state(query_state, memory=memory)
        if structured is None and query_state.filters:
            structured = plan_from_query_state(
                query_state, memory=memory, min_confidence=0.5
            )
        if structured is not None:
            return structured, "query_state"

        heuristic = try_heuristic_plan(question, memory=memory)
        if heuristic is not None:
            return heuristic, "heuristic"

        packet = self._context_packet(
            question, auth, memory, query_state, context_manager
        )
        user = json.dumps(packet, default=str)
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
            return ExecutionPlan.model_validate(data), "llm"
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "planner_llm_invalid_plan",
                question=question[:200],
                error_type=type(exc).__name__,
                error=str(exc)[:400],
                approx_tokens=packet.get("approx_tokens"),
            )
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=_CLARIFY_FALLBACK,
                ),
                "llm",
            )
