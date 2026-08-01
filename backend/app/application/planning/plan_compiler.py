from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.application.memory.context_view import (
    ContextNeed,
    classify_context_need,
    lint_fresh_scope,
    minimal_planner_packet,
)
from app.application.planning.bound_person import bound_person_attribute_plan
from app.application.planning.heuristic_planner import (
    _LOCATION_TOPIC_RE,
    _META_COUNT_RE,
    _PERSON_LOCATION_RE,
    _PRONOUN_ONLY_RE,
    _employee_by_id_plan,
    _entity_display_name,
    _meta_count_plan,
    _pronoun_clarify_plan,
    _resolve_pronoun_employee_id,
    try_heuristic_plan,
    wants_person_lookup,
)
from app.application.planning.location_plans import location_person_plan
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.salary_acl import salary_acl_plan
from app.application.planning.unsupported import is_unsupported_topic
from app.application.response.refusal import RefusalCode, out_of_scope_plan
from app.application.schema.catalog_service import CatalogService
from app.application.understanding.birthday import extract_birthday
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.list_referents import resolve_list_referent
from app.application.understanding.llm_slot_extractor import LlmSlotExtractor
from app.application.understanding.plan_from_query_state import (
    _set_status_plan,
    plan_from_query_state,
)
from app.application.understanding.plan_from_slots import plan_from_slots
from app.application.understanding.status_change import (
    PENDING_SET_STATUS_KEY,
    PENDING_STATUS_MUTATION_KEY,
    extract_status_change,
    is_confirm_status_update,
    looks_like_bare_person_name,
)
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
        self._slot_extractor = LlmSlotExtractor(llm, prompts_dir=self._prompts_dir)

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
        *,
        context_need: str | None = None,
    ) -> dict:
        if context_manager is not None and memory is not None:
            return context_manager.build_for_planner(
                question=question,
                auth=auth,
                memory=memory,
                tools=self._tools.discover(),
                schema_catalog=self._catalog_view(),
                query_state=query_state,
                context_need=context_need,
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
            context_need=context_need,
        )

    async def _llm_plan_once(
        self,
        question: str,
        *,
        packet: dict,
        context_need: ContextNeed,
    ) -> ExecutionPlan:
        """Call planner LLM once; raise on invalid JSON/schema/fresh-scope lint."""
        user = json.dumps(packet, default=str)
        raw = await self._llm.complete(
            system=self._load_prompt(), user=user, temperature=0.0, response_json=True
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        plan = ExecutionPlan.model_validate(data)
        lint_err = lint_fresh_scope(plan, context_need, question)
        if lint_err:
            raise ValueError(f"fresh_scope_lint: {lint_err}")
        return plan

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

        # Continue a prior location-cohort status write after explicit confirm.
        mutation = (
            memory.tool_fact_cache.get(PENDING_STATUS_MUTATION_KEY) if memory else None
        )
        if mutation is not None and is_confirm_status_update(q):
            payload = mutation.value if isinstance(mutation.value, dict) else None
            ids = list((payload or {}).get("employee_ids") or [])
            status = (payload or {}).get("status")
            if ids and isinstance(status, bool):
                return (
                    ExecutionPlan(
                        nodes=[
                            PlanNode(
                                id="e1",
                                kind="tool",
                                name="employee",
                                params={
                                    "action": "set_status",
                                    "status": status,
                                    "employee_ids": ids,
                                    "resolve_via": (payload or {}).get("resolve_via")
                                    or "location",
                                    "confirmed": True,
                                },
                            )
                        ],
                        response_strategy="template",
                        active_cohort_node="e1",
                    ),
                    "heuristic_confirm_status",
                )

        # Continue a prior "which employee's status?" clarify with a bare name.
        pending = (
            memory.tool_fact_cache.get(PENDING_SET_STATUS_KEY) if memory else None
        )
        if (
            pending is not None
            and isinstance(pending.value, bool)
            and looks_like_bare_person_name(q)
        ):
            return (
                _set_status_plan(
                    QueryState(
                        intent="set_status",
                        person_name=q.strip().strip(" .,?!"),
                        status_value=bool(pending.value),
                        confidence=0.95,
                    ),
                    filters={},
                ),
                "heuristic_pending_set_status",
            )

        # Salary/comp ACL before any SQL / LLM path (employees soft-refuse).
        acl = salary_acl_plan(q, auth=auth, memory=memory)
        if acl is not None:
            return acl, "salary_acl"

        # OOS before pronoun→profile dump ("her travelling preferences", PTO, …).
        if is_unsupported_topic(q):
            from app.application.planning.unsupported import unsupported_answer_for

            return out_of_scope_plan(unsupported_answer_for(q)), "unsupported"

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
                bound_id = pronoun_id
                if not bound_id and memory and len(memory.entity_memory) == 1:
                    bound_id = str(memory.entity_memory[0].employee_id)
                asks_where = bool(_PERSON_LOCATION_RE.search(q)) or bool(
                    _LOCATION_TOPIC_RE.search(q)
                )
                if asks_where:
                    # Place lives in the resume — never the employee profile row.
                    if not bound_id:
                        return _pronoun_clarify_plan(), "heuristic_pronoun"
                    name = _entity_display_name(memory, bound_id)
                    return (
                        location_person_plan(name, employee_ids=[bound_id]),
                        "heuristic_pronoun",
                    )
                # Birthday / status with a pronoun must bind before profile fallback.
                birthday = extract_birthday(q)
                if birthday.matched and birthday.scope == "person":
                    if not bound_id:
                        return _pronoun_clarify_plan(), "heuristic_pronoun"
                    name = _entity_display_name(memory, bound_id)
                    return (
                        bound_person_attribute_plan(
                            q, employee_id=bound_id, display_name=name
                        ),
                        "heuristic_pronoun",
                    )
                status = extract_status_change(q)
                if status.matched:
                    if not bound_id:
                        return _pronoun_clarify_plan(), "heuristic_pronoun"
                    name = _entity_display_name(memory, bound_id)
                    return (
                        bound_person_attribute_plan(
                            q, employee_id=bound_id, display_name=name
                        ),
                        "heuristic_pronoun",
                    )
                if bound_id:
                    name = _entity_display_name(memory, bound_id)
                    if re.search(r"\bmanagers?\b", q, re.I):
                        from app.application.planning.heuristic_planner import (
                            _manager_plan,
                        )

                        return (
                            _manager_plan(q, name=name, employee_id=bound_id),
                            "heuristic_pronoun",
                        )
                    return (
                        bound_person_attribute_plan(
                            q, employee_id=bound_id, display_name=name
                        ),
                        "heuristic_pronoun",
                    )
                return _pronoun_clarify_plan(), "heuristic_pronoun"

        # Ordinals / list deixis — before QueryState so "first person" is never a name.
        list_ref = resolve_list_referent(q, memory)
        if list_ref.kind == "clarify":
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=list_ref.clarify_question,
                    refusal_code=RefusalCode.AMBIGUOUS.value,
                ),
                "heuristic_list_referent",
            )
        if list_ref.kind == "resolved" and list_ref.employee_id:
            return (
                bound_person_attribute_plan(
                    q,
                    employee_id=list_ref.employee_id,
                    display_name=list_ref.display_name or "that employee",
                ),
                "heuristic_list_referent",
            )

        # Bare elliptical SQL attrs ("what is the education?") about the bound person.
        from app.application.planning.heuristic_planner import (
            elliptical_bound_person_plan,
        )

        elliptical = elliptical_bound_person_plan(q, memory=memory)
        if elliptical is not None:
            return elliptical, "heuristic_elliptical_person"

        # Birthday + pronoun without other profile cues (e.g. "when was she born").
        birthday_early = extract_birthday(q)
        if (
            birthday_early.matched
            and birthday_early.scope == "person"
            and _PRONOUN_ONLY_RE.search(q)
            and not birthday_early.person_name
        ):
            bound_id = pronoun_id
            if not bound_id and memory and len(memory.entity_memory) == 1:
                bound_id = str(memory.entity_memory[0].employee_id)
            if bound_id:
                name = _entity_display_name(memory, bound_id)
                return (
                    bound_person_attribute_plan(
                        q, employee_id=bound_id, display_name=name
                    ),
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

        # One shared context view for residual slots + LLM planner.
        context_need = classify_context_need(q, memory)
        logger.info(
            "planner_context_need",
            question=q[:200],
            context_need=context_need.value,
        )

        # Residual hybrid DST: structured slots before free-form LLM plans.
        # Regex/heuristic happy path never reaches here (no extra LLM cost).
        try:
            bundle = await self._slot_extractor.extract(
                question, memory, context_need=context_need
            )
        except Exception as exc:  # noqa: BLE001 — fall through to LLM planner
            logger.warning(
                "llm_slots_extract_error",
                question=question[:200],
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
            bundle = None
        if bundle is not None:
            slot_result = plan_from_slots(bundle, memory=memory, question=q)
            if slot_result.handled and slot_result.plan is not None:
                lint_err = lint_fresh_scope(slot_result.plan, context_need, q)
                if lint_err:
                    logger.warning(
                        "planner_fresh_scope_lint_failed",
                        source="llm_slots",
                        error=lint_err,
                        context_need=context_need.value,
                    )
                    # Do not open free-form planner with a cohort-bleeding slot plan.
                    return (
                        ExecutionPlan(
                            nodes=[],
                            response_strategy="template",
                            clarify_question=(
                                "I need a clearer ask without relying on the previous "
                                "result set. Name the person, department, city, or skill."
                            ),
                            refusal_code=RefusalCode.AMBIGUOUS.value,
                        ),
                        "llm_slots_lint_clarify",
                    )
                logger.info(
                    "planner_mode_llm_slots",
                    intent=bundle.intent,
                    confidence=bundle.confidence,
                    context_need=context_need.value,
                )
                return slot_result.plan, "llm_slots"
            if slot_result.handled and slot_result.plan is None and slot_result.clarify:
                return (
                    ExecutionPlan(
                        nodes=[],
                        response_strategy="template",
                        clarify_question=slot_result.clarify,
                        refusal_code=RefusalCode.AMBIGUOUS.value,
                    ),
                    "llm_slots",
                )

        packet = self._context_packet(
            question,
            auth,
            memory,
            query_state,
            context_manager,
            context_need=context_need.value,
        )
        try:
            plan = await self._llm_plan_once(
                q, packet=packet, context_need=context_need
            )
            return plan, "llm"
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "planner_llm_invalid_plan",
                question=question[:200],
                error_type=type(exc).__name__,
                error=str(exc)[:400],
                approx_tokens=packet.get("approx_tokens"),
                context_need=context_need.value,
            )
            # One stripped retry — no history/cohort — then clarify.
            stripped = minimal_planner_packet(
                question=q,
                role=auth.role.value,
                tools=self._tools.discover(),
                schema_catalog=self._catalog_view(),
                query_state=(
                    query_state.model_dump(mode="json") if query_state else None
                ),
                context_need=ContextNeed.FRESH.value,
            )
            try:
                plan = await self._llm_plan_once(
                    q, packet=stripped, context_need=ContextNeed.FRESH
                )
                logger.info(
                    "planner_llm_retry_stripped",
                    question=q[:200],
                    prior_need=context_need.value,
                )
                return plan, "llm_retry_stripped"
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc2:
                logger.warning(
                    "planner_llm_retry_failed",
                    question=question[:200],
                    error_type=type(exc2).__name__,
                    error=str(exc2)[:400],
                )
            return (
                ExecutionPlan(
                    nodes=[],
                    response_strategy="template",
                    clarify_question=_CLARIFY_FALLBACK,
                ),
                "llm",
            )
