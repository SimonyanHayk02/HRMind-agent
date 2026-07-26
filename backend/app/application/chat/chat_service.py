from __future__ import annotations

from uuid import uuid4

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.application.execution.langgraph_executor import LangGraphExecutor
from app.application.memory.cohort import should_update_last_employee_ids
from app.application.memory.context_updates import (
    extract_entities_from_state,
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
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.enums import RouterLabel

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
    ) -> None:
        self._memory = memory
        self._rule_router = rule_router
        self._embedding_router = embedding_router
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._formatter = formatter

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
            # Only promote chitchat→tools when the user clearly refers to a prior set.
            if (
                label == RouterLabel.CHITCHAT
                and session.last_employee_ids
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

        logger.info(
            "chat_plan_ready",
            trace_id=trace_id,
            session_id=session.session_id,
            question=body.question[:200],
            plan_nodes=[n.name for n in plan.nodes],
            clarify=bool(plan.clarify_question),
            last_employee_ids=len(session.last_employee_ids),
        )

        state = await self._executor.execute(plan, question=body.question, auth=auth)

        ids = extract_cohort_ids(state, plan)
        if ids and should_update_last_employee_ids(plan, body.question, ids):
            session = await self._memory.set_last_employee_ids(session, ids)

        entities = extract_entities_from_state(state)
        if entities:
            session = await self._memory.upsert_entities(session, entities)

        constraints = infer_constraints_from_question(body.question)
        if constraints:
            session = await self._memory.merge_constraints(session, constraints)

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
