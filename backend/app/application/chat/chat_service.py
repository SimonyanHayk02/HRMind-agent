from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.application.execution.graph_state import GraphState
from app.application.execution.langgraph_executor import LangGraphExecutor
from app.application.memory.context_manager import ContextManager, TurnTrace
from app.application.memory.context_view import (
    ContextNeed,
    _ELLIPTICAL_ATTR_RE,
    classify_context_need,
)
from app.application.planning.heuristic_planner import (
    _PRONOUN_ONLY_RE,
    refers_to_prior_set,
)
from app.application.planning.plan_compiler import PlanCompiler
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.planning.plan_validator import PlanValidator
from app.application.response.refusal import (
    is_soft_unauthorized_error,
    resolve_refusal,
    unauthorized_plan,
)
from app.application.response.response_formatter import ResponseFormatter
from app.application.routing.embedding_router import EmbeddingRouter
from app.application.routing.rule_router import RuleRouter
from app.application.schema.catalog_service import CatalogService
from app.application.understanding.birthday import extract_birthday
from app.application.understanding.extract_query_state import extract_query_state
from app.application.understanding.list_referents import has_list_referent_phrase
from app.application.understanding.status_change import (
    PENDING_SET_STATUS_KEY,
    PENDING_STATUS_MUTATION_KEY,
    extract_status_change,
)
from app.config.logging import get_logger
from app.domain.auth import AuthContext
from app.domain.enums import RouterLabel
from app.domain.errors import PlanInvalidError
from app.domain.query_state import QueryState
from app.domain.schema_catalog import default_employee_catalog
from app.domain.session import SessionMemory, ToolFact
from app.domain.tools.base import ToolResult

logger = get_logger(__name__)


def tools_that_answered(plan: ExecutionPlan, state: GraphState) -> str | None:
    """Name the tools whose output the answer is built from, in plan order.

    Operators are left out: they reshape another tool's result rather than
    fetching anything, so listing them would misreport where the answer came
    from. Tools that errored are left out for the same reason — `degraded`
    already tells the client the turn was incomplete.
    """
    names: list[str] = []
    for node in plan.nodes:
        if node.kind != "tool" or node.name in names:
            continue
        result = state.node_results.get(node.id)
        if isinstance(result, ToolResult) and not result.error:
            names.append(node.name)
    return "+".join(names) or None


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
            status_req = extract_status_change(body.question)
            if label == RouterLabel.CHITCHAT and (
                status_req.matched
                or has_list_referent_phrase(body.question)
                or extract_birthday(body.question).matched
            ):
                label = RouterLabel.NEEDS_TOOLS
            has_person_focus = bool(
                session.person_bindings
                or (session.active_referent and session.active_referent.ids)
                or len(session.last_listed) == 1
                or len(session.entity_memory) == 1
            )
            need = (
                classify_context_need(body.question, session)
                if has_context or has_person_focus
                else None
            )
            followup_attr = bool(
                has_person_focus
                and (
                    _PRONOUN_ONLY_RE.search(body.question)
                    or _ELLIPTICAL_ATTR_RE.search(body.question)
                    or (
                        need
                        in {
                            ContextNeed.ANAPHORA,
                            ContextNeed.ANAPHORA_NAMED,
                            ContextNeed.ELLIPTICAL_PERSON,
                            ContextNeed.LIST_ORDINAL,
                        }
                    )
                )
            )
            if label == RouterLabel.CHITCHAT and has_context and (
                refers_to_prior_set(body.question) or followup_attr
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
                try:
                    self._validator.validate(plan, auth)
                except PlanInvalidError as exc:
                    # Soft-chat whitelist only — ACL denials become chat refuses.
                    if is_soft_unauthorized_error(str(exc)):
                        plan = unauthorized_plan()
                        planner_mode = "soft_unauthorized"
                    else:
                        raise

        trace.planner_mode = planner_mode

        catalog = self._catalog.get() if self._catalog else default_employee_catalog()
        query_state = extract_query_state(
            body.question, catalog=catalog, memory=session
        )
        session = await self._context.merge_query_constraints(session, query_state)
        session = await self._sync_pending_set_status(
            session, plan=plan, query_state=query_state
        )

        logger.info(
            "chat_plan_ready",
            **{
                **trace.as_dict(),
                "question": body.question[:200],
                "plan_nodes": [n.name for n in plan.nodes],
                "clarify": bool(plan.clarify_question),
                "refusal_code": plan.refusal_code,
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

        # Resolve before commit so memory uses the typed code, not answer text.
        refusal = resolve_refusal(plan, state, body.question, auth=auth)
        refusal_code = refusal.code.value if refusal else (plan.refusal_code or "ok")

        session = await self._context.commit(
            session,
            question=body.question,
            plan=plan,
            state=state,
            refusal_code=refusal_code,
        )
        session = await self._sync_pending_status_mutation(session, state=state)
        trace.referent_after = len(session.last_employee_ids)
        trace.tools_called = [n.name for n in plan.nodes]
        trace.degraded = state.degraded
        tool = tools_that_answered(plan, state)

        answer, confidence, sources, clarify = await self._formatter.format(
            body.question,
            plan,
            state,
            context_manager=self._context,
            auth=auth,
        )
        safe_sources = [s for s in (sources or []) if getattr(s, "ref", None) is not None]
        await self._context.append_assistant(session, answer or "")

        logger.info(
            "chat_turn_complete",
            **{**trace.as_dict(), "tool": tool, "refusal_code": refusal_code},
        )

        return ChatResponse(
            session_id=session.session_id,
            answer=answer or "",
            confidence=confidence if confidence is not None else 0.0,
            sources=safe_sources,
            clarify=clarify,
            tool=tool,
            trace_id=trace_id,
            degraded=state.degraded,
        )

    async def _sync_pending_set_status(
        self,
        session: SessionMemory,
        *,
        plan: ExecutionPlan,
        query_state: QueryState,
    ) -> SessionMemory:
        """Remember a missing-person status write so a bare-name reply can finish it."""
        clarify = (plan.clarify_question or "").lower()
        asking_who = "which employee" in clarify and "status" in clarify
        if (
            query_state.intent == "set_status"
            and asking_who
            and query_state.status_value is not None
            and not plan.nodes
        ):
            return await self._context.put_tool_fact(
                session,
                ToolFact(
                    key=PENDING_SET_STATUS_KEY,
                    value=bool(query_state.status_value),
                    created_at=datetime.now(UTC),
                    ttl_seconds=180,
                ),
            )
        has_status_write = any(
            n.name == "employee" and (n.params or {}).get("action") == "set_status"
            for n in plan.nodes
        )
        # Bare-name pending bool clears when any set_status node is planned.
        # Location mutation facts clear after a successful write (or topic shift).
        if has_status_write:
            return await self._context.clear_tool_fact(
                session, PENDING_SET_STATUS_KEY
            )
        if (
            plan.nodes
            and query_state.intent not in {"set_status", "unknown"}
            and (
                PENDING_SET_STATUS_KEY in session.tool_fact_cache
                or PENDING_STATUS_MUTATION_KEY in session.tool_fact_cache
            )
        ):
            session = await self._context.clear_tool_fact(
                session, PENDING_SET_STATUS_KEY
            )
            return await self._context.clear_tool_fact(
                session, PENDING_STATUS_MUTATION_KEY
            )
        return session

    async def _sync_pending_status_mutation(
        self,
        session: SessionMemory,
        *,
        state: GraphState,
    ) -> SessionMemory:
        """Persist a location-cohort status propose so 'confirm status update' can finish it."""
        for result in state.node_results.values():
            data = getattr(result, "data", result)
            if not isinstance(data, dict):
                continue
            if data.get("updated"):
                if PENDING_STATUS_MUTATION_KEY in session.tool_fact_cache:
                    return await self._context.clear_tool_fact(
                        session, PENDING_STATUS_MUTATION_KEY
                    )
                return session
            pending = data.get("pending_status")
            ids = data.get("employee_ids") or []
            if (
                data.get("clarify")
                and isinstance(pending, bool)
                and isinstance(ids, list)
                and ids
            ):
                return await self._context.put_tool_fact(
                    session,
                    ToolFact(
                        key=PENDING_STATUS_MUTATION_KEY,
                        value={
                            "kind": "set_status",
                            "status": pending,
                            "employee_ids": [str(x) for x in ids],
                            "resolve_via": "location",
                        },
                        created_at=datetime.now(UTC),
                        ttl_seconds=180,
                    ),
                )
        return session
