import asyncio
from uuid import uuid4

import pytest

from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.persistence.memory_session_store import MemorySessionStore
from app.application.execution.graph_state import GraphState
from app.application.memory.memory_service import MemoryService
from app.application.memory.result_ids import extract_employee_ids_from_state
from app.application.response.response_formatter import _format_employee_rows
from app.application.routing.embedding_router import EmbeddingRouter
from app.application.routing.rule_router import RuleRouter
from app.composition import build_container
from app.config.settings import Settings
from app.domain.auth import AuthContext
from app.domain.enums import Role, RouterLabel
from app.domain.tools.base import ToolResult


def test_extract_ids_from_resume_and_operator() -> None:
    state = GraphState(
        question="q",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "r1": ToolResult(
                data={
                    "employee_ids": ["00000000-0000-0000-0000-000000000001"],
                    "hits": [],
                }
            ),
            "ids": ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"],
        },
    )
    ids = extract_employee_ids_from_state(state)
    assert ids == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]


def test_format_employee_rows() -> None:
    text = _format_employee_rows(
        [
            {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "department": "Engineering",
                "position": "Engineer",
            },
            {
                "first_name": "Alan",
                "last_name": "Turing",
                "department": "Engineering",
                "position": "Scientist",
            },
        ]
    )
    assert text is not None
    assert "Ada Lovelace" in text
    assert "Alan Turing" in text


def test_embedding_router_names_followup_is_needs_tools() -> None:
    router = EmbeddingRouter(FakeEmbeddings(dims=32), threshold=0.99)
    label = asyncio.run(router.route("please say there names"))
    assert label == RouterLabel.NEEDS_TOOLS


def test_rule_router_pure_greeting_only() -> None:
    router = RuleRouter()
    assert router.route("hello") == RouterLabel.GREETING
    assert router.route("thanks") == RouterLabel.GREETING
    assert router.route("thanks, say their names") is None
    assert router.route("hi who knows python") is None


@pytest.mark.asyncio
async def test_memory_persists_last_ids_and_constraints() -> None:
    store = MemorySessionStore()
    mem = MemoryService(store)
    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    session = await mem.get_or_create(None, auth)
    session = await mem.set_last_employee_ids(session, ["00000000-0000-0000-0000-000000000099"])
    from app.domain.session import ConstraintRef

    session = await mem.merge_constraints(
        session, [ConstraintRef(field="skill", op="contains", value="Python")]
    )
    loaded = await store.get(session.session_id)
    assert loaded is not None
    assert loaded.last_employee_ids == ["00000000-0000-0000-0000-000000000099"]
    assert loaded.constraint_memory[0].value == "Python"


@pytest.mark.asyncio
async def test_redis_required_in_production_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):  # noqa: ANN001
        raise ConnectionError("redis down")

    monkeypatch.setattr("app.composition.Redis.from_url", boom)
    with pytest.raises(RuntimeError, match="Redis is required"):
        await build_container(
            Settings(
                openai_api_key="",
                app_env="production",
                require_redis=True,
                database_url="postgresql+asyncpg://u:p@localhost:5433/hrmind",
            ),
            use_fakes=False,
        )


from app.application.memory.cohort import should_update_last_employee_ids
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
from app.application.response.response_formatter import ResponseFormatter
from app.adapters.llm.fake_llm import FakeLLM


def test_formatter_rejects_spurious_headcount_for_vacation() -> None:
    formatter = ResponseFormatter(FakeLLM())
    plan = ExecutionPlan(
        nodes=[PlanNode(id="sql1", kind="tool", name="sql", params={"mode": "constrained"})],
        response_strategy="llm_format",
    )
    state = GraphState(
        question="how much vacation are developers taking?",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "sql1": ToolResult(
                data={
                    "rows": [{"id": "1", "first_name": "A", "last_name": "B"}] * 20,
                    "row_count": 100,
                    "sql": "SELECT ... FROM employees e WHERE 1=1 LIMIT 200",
                },
                confidence=1.0,
            )
        },
    )

    async def _run():
        return await formatter.format(
            "how much vacation are developers taking?", plan, state
        )

    answer, conf, _, clarify = asyncio.run(_run())
    assert "don't have that information" in answer.lower()
    assert "100" not in answer
    assert clarify is None


def test_unsupported_plan_does_not_duplicate_clarify() -> None:
    formatter = ResponseFormatter(FakeLLM())
    msg = "I don't have that information in the HR data I can access."
    plan = ExecutionPlan(nodes=[], response_strategy="template", clarify_question=msg)
    state = GraphState(
        question="do we have information about the vacations ?",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
    )
    answer, _, _, clarify = asyncio.run(formatter.format(state.question, plan, state))
    assert answer == msg
    assert clarify is None



def test_cohort_rejects_unscoped_headcount() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={"mode": "nl2sql", "count_only": False, "question": "how many"},
            )
        ]
    )
    ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(100)]
    assert not should_update_last_employee_ids(plan, "How many employees do we have?", ids)


def test_cohort_keeps_resume_matches() -> None:
    plan = ExecutionPlan(
        nodes=[
            PlanNode(id="r1", kind="tool", name="resume_search", params={}),
            PlanNode(
                id="sql1",
                kind="tool",
                name="sql",
                params={"mode": "constrained", "count_only": True, "filters": {}},
                input_bindings={"employee_ids": "nodes.ids"},
            ),
        ]
    )
    ids = ["00000000-0000-0000-0000-000000000001"]
    assert should_update_last_employee_ids(plan, "how many know python", ids)


@pytest.mark.asyncio
async def test_session_tenant_mismatch_forbidden() -> None:
    store = MemorySessionStore()
    mem = MemoryService(store)
    sid = str(uuid4())
    await mem.get_or_create(
        sid, AuthContext(user_id="u", tenant_id="tenant-a", role=Role.RECRUITER)
    )
    from app.domain.errors import ForbiddenError

    with pytest.raises(ForbiddenError):
        await mem.get_or_create(
            sid, AuthContext(user_id="u", tenant_id="tenant-b", role=Role.RECRUITER)
        )
