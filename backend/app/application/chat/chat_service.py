from __future__ import annotations

from uuid import uuid4

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.application.execution.langgraph_executor import LangGraphExecutor
from app.application.memory.cohort import should_update_last_employee_ids
from app.application.memory.context_updates import (
    extract_entities_from_state,
    extract_last_focus,
    infer_constraints_from_question,
)
from app.application.memory.memory_service import MemoryService
from app.application.memory.result_ids import extract_cohort_ids
from app.application.planning.heuristic_planner import refers_to_prior_set
from app.application.planning.plan_compiler import PlanCompiler
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.plan_validator import PlanValidator
from app.application.response.response_formatter import ResponseFormatter
from app.application.routing.embedding_router import EmbeddingRouter
from app.application.routing.rule_router import RuleRouter
from app.application.schema.catalog_service import CatalogService
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.merge_query_state import apply_universe_marker
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.enums import RouterLabel
from app.domain.schema_catalog import default_employee_catalog

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        *,
        memory: MemoryService,
        rule_router: RuleRouter,
        embedding_router: EmbeddingRouter,
        planner: PlanCompiler,
        validator: PlanValidator,
        executor: LangGraphExecutor,
        formatter: ResponseFormatter,
        catalog_service: CatalogService | None = None,
    ) -> None:
        self._memory = memory
        self._rule_router = rule_router
        self._embedding_router = embedding_router
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._formatter = formatter
        self._catalog = catalog_service

    async def handle(self, body: ChatRequest, *, auth: AuthContext) -> ChatResponse:
        trace_id = str(uuid4())
        session = await self._memory.get_or_create(body.session_id, auth)
        session = await self._memory.append_user(session, body.question)

        rule = self._rule_router.route(body.question)
        if rule == RouterLabel.GREETING:
            plan = ExecutionPlan(
                nodes=[
                    PlanNode(
                        id="greet",
                        kind="tool",
                        name="greeting",
                        params={"message": body.question},
                    )
                ],
                response_strategy="template",
            )
        else:
            label = await self._embedding_router.route(body.question)
            # Promote chitchat→tools for short follow-ups with any saved context.
            has_context = bool(
                session.last_employee_ids
                or session.last_focus
                or session.constraint_memory
            )
            if (
                label == RouterLabel.CHITCHAT
                and has_context
                and refers_to_prior_set(body.question)
            ):
                label = RouterLabel.NEEDS_TOOLS
            if label == RouterLabel.CHITCHAT:
                plan = ExecutionPlan(
                    nodes=[
                        PlanNode(
                            id="greet",
                            kind="tool",
                            name="greeting",
                            params={"message": body.question},
                        )
                    ],
                    response_strategy="template",
                )
            else:
                plan = await self._planner.compile(body.question, auth=auth, memory=session)
                self._validator.validate(plan, auth)

        # Understand + persist dialog filters (even when heuristics/LLM planned)
        catalog = self._catalog.get() if self._catalog else default_employee_catalog()
        query_state = extract_query_state(
            body.question, catalog=catalog, memory=session
        )
        qs_constraints = apply_universe_marker(session, query_state)
        if qs_constraints:
            session = await self._memory.merge_constraints(session, qs_constraints)

        logger.info(
            "chat_plan_ready",
            trace_id=trace_id,
            session_id=session.session_id,
            question=body.question[:200],
            plan_nodes=[n.name for n in plan.nodes],
            clarify=bool(plan.clarify_question),
            last_employee_ids=len(session.last_employee_ids),
            query_intent=query_state.intent,
            query_filters=[f"{f.field}={f.value}" for f in query_state.filters],
            last_focus=(
                f"{session.last_focus.kind}:{session.last_focus.dimension}"
                if session.last_focus
                else None
            ),
        )

        state = await self._executor.execute(plan, question=body.question, auth=auth)

        ids = extract_cohort_ids(state, plan)
        saved_ids = ids if ids and should_update_last_employee_ids(plan, body.question, ids) else []
        if saved_ids:
            session = await self._memory.set_last_employee_ids(session, saved_ids)

        entities = extract_entities_from_state(state)
        if entities:
            session = await self._memory.upsert_entities(session, entities)

        constraints = infer_constraints_from_question(body.question)
        if constraints:
            session = await self._memory.merge_constraints(session, constraints)

        focus = extract_last_focus(state, plan, employee_ids=saved_ids or None)
        if focus:
            session = await self._memory.set_last_focus(session, focus)

        answer, confidence, sources, clarify = await self._formatter.format(
            body.question, plan, state
        )
        await self._memory.append_assistant(session, answer)
        return ChatResponse(
            session_id=session.session_id,
            answer=answer,
            confidence=confidence,
            sources=sources or [],
            clarify=clarify,
            trace_id=trace_id,
            degraded=state.degraded,
        )
