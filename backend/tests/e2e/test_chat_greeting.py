import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.persistence.memory_session_store import MemorySessionStore
from app.application.memory.memory_service import MemoryService
from app.composition import build_container
from app.config.settings import Settings
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.main import create_app


@pytest.mark.asyncio
async def test_chat_greeting() -> None:
    app = create_app()
    app.state.container = await build_container(
        Settings(openai_api_key="", app_env="test"), use_fakes=True
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "hello"},
            headers={"X-Role": "recruiter", "X-User-Id": "u1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "Hello" in body["answer"]
    assert body["session_id"]


@pytest.mark.asyncio
async def test_multi_turn_names_followup_uses_session_ids() -> None:
    """Simulate prior result set in session, then ask for names."""
    app = create_app()
    container = await build_container(Settings(openai_api_key="", app_env="test"), use_fakes=True)
    app.state.container = container

    store = container.session_store
    assert isinstance(store, MemorySessionStore)
    mem = MemoryService(store)
    auth = AuthContext(
        user_id="u1",
        tenant_id=Settings().default_tenant_id,
        role=Role.RECRUITER,
    )
    # Seed a session with prior employee IDs (as resume/SQL would).
    session = await mem.get_or_create(None, auth)
    session = await mem.set_last_employee_ids(
        session,
        [
            # IDs that may not exist in DB — SQL should still run; empty/0 rows ok.
            # Use nil uuid pattern; constrained query with 1=0 if invalid — use valid UUIDs.
            "00000000-0000-0000-0000-000000000001",
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "please say their names", "session_id": session.session_id},
            headers={
                "X-Role": "recruiter",
                "X-User-Id": "u1",
                "X-Tenant-Id": Settings().default_tenant_id,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    # Must not fall back to greeting
    assert "Hello!" not in body["answer"]
    assert body["session_id"] == session.session_id
