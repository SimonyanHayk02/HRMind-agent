from app.application.execution.graph_state import GraphState
from app.application.memory.result_ids import extract_employee_ids_from_state
from app.application.response.response_formatter import _format_employee_rows
from app.application.routing.embedding_router import EmbeddingRouter
from app.domain.auth import AuthContext
from app.domain.enums import Role, RouterLabel
from app.domain.tools.base import ToolResult
from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
import asyncio


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
            {"first_name": "Ada", "last_name": "Lovelace", "department": "Engineering", "position": "Engineer"},
            {"first_name": "Alan", "last_name": "Turing", "department": "Engineering", "position": "Scientist"},
        ]
    )
    assert text is not None
    assert "Ada Lovelace" in text
    assert "Alan Turing" in text


def test_embedding_router_names_followup_is_needs_tools() -> None:
    router = EmbeddingRouter(FakeEmbeddings(dims=32), threshold=0.99)
    label = asyncio.run(router.route("please say there names"))
    assert label == RouterLabel.NEEDS_TOOLS
