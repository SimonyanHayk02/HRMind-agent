import asyncio
from uuid import uuid4

import pytest

from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.persistence.memory_session_store import MemorySessionStore
from app.application.execution.graph_state import GraphState
from app.application.memory.memory_service import MemoryService
from app.application.memory.result_ids import extract_cohort_ids, extract_employee_ids_from_state
from app.application.memory.context_updates import extract_entities_from_state
from app.application.planning.plan_schema import ExecutionPlan, PlanNode
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


def test_extract_cohort_prefers_intersect_not_union() -> None:
    rag = "00000000-0000-0000-0000-000000000001"
    prior = "00000000-0000-0000-0000-000000000002"
    intersected = "00000000-0000-0000-0000-000000000003"
    plan = ExecutionPlan(
        nodes=[
            PlanNode(id="r1", kind="tool", name="resume_search", params={}),
            PlanNode(id="ids", kind="operator", name="extract_employee_ids"),
            PlanNode(id="ix", kind="operator", name="intersect_ids"),
            PlanNode(id="sql1", kind="tool", name="sql", params={"count_only": True}),
        ],
        active_cohort_node="ix",
    )
    state = GraphState(
        question="q",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "r1": ToolResult(data={"employee_ids": [rag, intersected], "hits": []}),
            "ids": [rag, intersected],
            "ix": [intersected],
            "sql1": ToolResult(data={"count": 1, "rows": [{"count": 1}]}),
        },
    )
    assert extract_cohort_ids(state, plan) == [intersected]
    # Without plan, last non-empty wins — still should not invent a union across nodes
    assert prior not in extract_cohort_ids(state, None)


def test_org_wide_headcount_resets_stale_cohort_and_filters() -> None:
    from app.application.understanding.merge_query_state import apply_universe_marker
    from app.domain.query_state import QueryState
    from app.domain.session import ActiveReferent, ConstraintRef, SessionMemory

    mem = SessionMemory(
        session_id="s",
        tenant_id="t",
        user_id="u",
        role=Role.RECRUITER,
        last_employee_ids=["00000000-0000-0000-0000-0000000000aa"],
        constraint_memory=[ConstraintRef(field="department", op="eq", value="Product")],
        active_referent=ActiveReferent(
            ids=["00000000-0000-0000-0000-0000000000aa"],
            label="alice",
            source_turn=1,
        ),
    )
    out = apply_universe_marker(mem, QueryState(intent="count", confidence=0.9))
    assert mem.last_employee_ids == []
    assert mem.active_referent is None
    assert mem.constraint_memory == []
    assert [(c.field, c.value) for c in out] == [("_universe", "all")]


def test_extract_cohort_empty_intersect_does_not_widen() -> None:
    """Empty intersect must not fall through to the pre-intersect location ids."""
    dubai_a = "00000000-0000-0000-0000-0000000000aa"
    dubai_b = "00000000-0000-0000-0000-0000000000bb"
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="loc",
                kind="tool",
                name="resume_search",
                params={"purpose": "location_cohort", "city": "Dubai"},
            ),
            PlanNode(id="locids", kind="operator", name="extract_employee_ids"),
            PlanNode(id="locix", kind="operator", name="intersect_ids"),
            PlanNode(id="sql1", kind="tool", name="sql", params={"count_only": True}),
        ],
        active_cohort_node="locix",
    )
    state = GraphState(
        question="how much of them are from dubai",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "loc": ToolResult(data={"employee_ids": [dubai_a, dubai_b], "hits": []}),
            "locids": [dubai_a, dubai_b],
            "locix": [],
            "sql1": ToolResult(data={"count": 0, "rows": [{"count": 0}]}),
        },
    )
    assert extract_cohort_ids(state, plan) == []


def test_entities_from_resume_hits() -> None:
    state = GraphState(
        question="q",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "r1": ToolResult(
                data={
                    "hits": [
                        {
                            "employee_id": "00000000-0000-0000-0000-000000000001",
                            "employee_name": "Ada Lovelace",
                            "snippets": [],
                        }
                    ],
                    "employee_ids": ["00000000-0000-0000-0000-000000000001"],
                }
            )
        },
    )
    ents = extract_entities_from_state(state)
    assert len(ents) == 1
    assert ents[0].display_name == "Ada Lovelace"


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
    assert router.route("how are you") == RouterLabel.GREETING
    assert router.route("who are you") == RouterLabel.GREETING
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


def test_extract_last_focus_from_location_facet_plan() -> None:
    from app.application.memory.context_updates import extract_last_focus

    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="resume_search",
                params={"purpose": "location_facet", "facet": "country", "facet_count": True},
            ),
        ]
    )
    state = GraphState(
        question="q",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "facet": ToolResult(
                data={
                    "rows": [
                        {"country": "Germany"},
                        {"country": "USA"},
                        {"country": "UK"},
                    ],
                    "count": 3,
                    "employee_ids": [],
                }
            ),
        },
    )
    focus = extract_last_focus(state, plan)
    assert focus is not None
    assert focus.kind == "facet"
    assert focus.dimension == "country"
    assert focus.values == ["Germany", "USA", "UK"]


def test_formatter_lists_countries_from_resume_facet() -> None:
    formatter = ResponseFormatter(FakeLLM())
    plan = ExecutionPlan(
        nodes=[
            PlanNode(
                id="facet",
                kind="tool",
                name="resume_search",
                params={"purpose": "location_facet", "facet": "country"},
            )
        ],
        response_strategy="template",
    )
    state = GraphState(
        question="names please",
        auth=AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER),
        node_results={
            "facet": ToolResult(
                data={
                    "rows": [{"country": "Germany"}, {"country": "France"}],
                    "count": 2,
                    "employee_ids": [],
                }
            )
        },
    )
    answer, _, _, _ = asyncio.run(formatter.format("names please", plan, state))
    assert "Germany" in answer
    assert "France" in answer
    assert "countries" in answer.lower()


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
