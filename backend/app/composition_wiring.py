from __future__ import annotations

from pathlib import Path

from app.adapters.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.adapters.sql.executor import SqlExecutor
from app.adapters.vectorstore.memory_vector_store import MemoryVectorStore
from app.application.chat.chat_service import ChatService
from app.application.execution.langgraph_executor import LangGraphExecutor
from app.application.execution.node_runner import NodeRunner
from app.application.memory.memory_service import MemoryService
from app.application.memory.summarizer import Summarizer
from app.application.planning.plan_compiler import PlanCompiler
from app.application.planning.plan_validator import PlanValidator
from app.application.response.response_formatter import ResponseFormatter
from app.application.routing.embedding_router import EmbeddingRouter
from app.application.routing.rule_router import RuleRouter
from app.domain.operators.base import OperatorRegistry
from app.domain.operators.count import Count
from app.domain.operators.dedupe import Dedupe
from app.domain.operators.extract_ids import ExtractEmployeeIds
from app.domain.operators.intersect_ids import IntersectIds
from app.domain.operators.merge_results import MergeResults
from app.domain.operators.project_fields import ProjectFields
from app.domain.tools.registry import ToolRegistry
from app.tools.clarify.tool import ClarifyTool
from app.tools.employee.tool import EmployeeTool
from app.tools.greeting.tool import GreetingTool
from app.tools.resume_search.tool import ResumeSearchTool
from app.tools.sql.tool import SqlTool


def wire_application_stack(container) -> None:
    settings = container.settings
    prompts_dir = Path(__file__).resolve().parent / "prompts"

    tools = ToolRegistry()
    operators = OperatorRegistry()
    for op in (ExtractEmployeeIds(), Count(), IntersectIds(), MergeResults(), ProjectFields(), Dedupe()):
        operators.register(op)

    def uow_factory(auth):
        return SqlAlchemyUnitOfWork(container.session_factory, auth=auth)

    tools.register(GreetingTool())
    tools.register(ClarifyTool())
    tools.register(EmployeeTool(uow_factory))

    executor = SqlExecutor(container.session_factory)
    if container.session_factory is not None:
        from app.adapters.vectorstore.pgvector_store import PgVectorStore

        vector_store = container.extras.get("vector_store") or PgVectorStore(container.session_factory)
    else:
        vector_store = container.extras.get("vector_store") or MemoryVectorStore()
    container.extras["vector_store"] = vector_store

    def prompt_loader(name: str) -> str:
        path = prompts_dir / f"{name}.md"
        return path.read_text() if path.exists() else ""

    tools.register(
        SqlTool(
            executor=executor,
            cache=container.cache,
            llm=container.llm,
            max_rows=settings.sql_max_rows,
            cache_ttl=settings.sql_cache_ttl,
            prompt_loader=prompt_loader,
        )
    )
    tools.register(
        ResumeSearchTool(
            embeddings=container.embeddings,
            vector_store=vector_store,
            cache=container.cache,
            top_k=settings.top_k,
            rerank_top_k=settings.rerank_top_k,
            cache_ttl=settings.retrieval_cache_ttl,
        )
    )

    memory = MemoryService(
        container.session_store,
        Summarizer(container.llm, prompts_dir),
        max_history=settings.max_history,
        summary_trigger=settings.summary_trigger,
    )
    planner = PlanCompiler(container.llm, tools, prompts_dir)
    validator = PlanValidator(tools, operators, max_nodes=settings.max_plan_nodes)
    node_runner = NodeRunner(tools, operators, timeout_ms=settings.tool_timeout_ms)
    graph_executor = LangGraphExecutor(node_runner)
    formatter = ResponseFormatter(container.llm, prompts_dir)

    chat_service = ChatService(
        memory=memory,
        rule_router=RuleRouter(),
        embedding_router=EmbeddingRouter(container.embeddings, threshold=settings.router_threshold),
        planner=planner,
        validator=validator,
        executor=graph_executor,
        formatter=formatter,
    )
    container.extras["chat_service"] = chat_service
    container.extras["tool_registry"] = tools
    container.extras["operator_registry"] = operators
