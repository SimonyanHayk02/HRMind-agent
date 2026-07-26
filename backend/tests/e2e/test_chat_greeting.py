import pytest
from httpx import ASGITransport, AsyncClient

from app.composition import build_container
from app.config.settings import Settings
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
