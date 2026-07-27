from __future__ import annotations

from uuid import uuid4

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.application.execution.langgraph_executor import LangGraphExecutor
from app.application.memory.context_manager import ContextManager, TurnTrace
from app.application.planning.heuristic_planner import refers_to_prior_set
from app.application.planning.plan_compiler import PlanCompiler
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.plan_validator import PlanValidator
from app.application.response.response_formatter import ResponseFormatter
from app.application.routing.embedding_router import EmbeddingRouter
from app.application.routing.rule_router import RuleRouter
from app.application.schema.catalog_service import CatalogService
from app.application.understanding.extract_query_state import extract_query_state
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.enums import RouterLabel
from app.domain.schema_catalog import default_employee_catalog

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        *,
        context: ContextManager,
        rule_router: RuleRouter,
        embedding_router: EmbeddingRouter,
        planner: PlanCompiler,
        validator: PlanValidator,
        executor: LangGraphExecutor,
        formatter: ResponseFormatter,
        catalog_service: CatalogService | None = None,
    ) -> None:
        self._context = context
        self._rule_router = rule_router
        self._embedding_router = embedding_router
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._formatter = formatter
        self._catalog = catalog_service

    async def handle(self, body: ChatRequest, *, auth: AuthContext) -> ChatResponse:
        trace_id = str(uuid4())
        session = await self._context.load(body.session_id, auth)
        session = await self._context.append_user(session, body.question)
        session, working = await self._context.prepare_turn(body.question, session)

        trace = TurnTrace(
            trace_id=trace_id,
            session_id=session.session_id,
            refers_to_prior=working.resolved.refers_to_prior,
            referent_before=len(session.last_employee_ids),
        )

        planner_mode = "greeting"
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
            trace.router_label = RouterLabel.GREETING.value
        else:
            label = await self._embedding_router.route(body.question)
            has_context = bool(
                session.last_employee_ids
                or session.last_focus
                or session.constraint_memory
                or session.active_referent
            )
            if (
                label == RouterLabel.CHITCHAT
                and has_context
                and refers_to_prior_set(body.question)
            ):
                label = RouterLabel.NEEDS_TOOLS
            trace.router_label = label.value
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
                planner_mode = "greeting"
            else:
                plan, planner_mode = await self._planner.compile(
                    body.question,
                    auth=auth,
                    memory=session,
                    context_manager=self._context,
                )
                self._validator.validate(plan, auth)

        trace.planner_mode = planner_mode

        catalog = self._catalog.get() if self._catalog else default_employee_catalog()
        query_state = extract_query_state(
            body.question, catalog=catalog, memory=session
        )
        session = await self._context.merge_query_constraints(session, query_state)

        logger.info(
            "chat_plan_ready",
            **{
                **trace.as_dict(),
                "question": body.question[:200],
                "plan_nodes": [n.name for n in plan.nodes],
                "clarify": bool(plan.clarify_question),
                "query_intent": query_state.intent,
                "query_filters": [f"{f.field}={f.value}" for f in query_state.filters],
                "last_focus": (
                    f"{session.last_focus.kind}:{session.last_focus.dimension}"
                    if session.last_focus
                    else None
                ),
            },
        )

        state = await self._executor.execute(
            plan,
            question=body.question,
            auth=auth,
            session_entities=session.entity_memory,
            last_employee_ids=session.last_employee_ids,
            person_bindings=session.person_bindings,
        )

        session = await self._context.commit(
            session, question=body.question, plan=plan, state=state
        )
        trace.referent_after = len(session.last_employee_ids)
        trace.tools_called = [n.name for n in plan.nodes]
        trace.degraded = state.degraded

        answer, confidence, sources, clarify = await self._formatter.format(
            body.question,
            plan,
            state,
            context_manager=self._context,
            auth=auth,
        )
        safe_sources = [s for s in (sources or []) if getattr(s, "ref", None) is not None]
        await self._context.append_assistant(session, answer or "")

        logger.info("chat_turn_complete", **trace.as_dict())

        return ChatResponse(
            session_id=session.session_id,
            answer=answer or "",
            confidence=confidence if confidence is not None else 0.0,
            sources=safe_sources,
            clarify=clarify,
            trace_id=trace_id,
            degraded=state.degraded,
        )
