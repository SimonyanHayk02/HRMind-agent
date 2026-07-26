import pytest
from httpx import ASGITransport, AsyncClient

from app.composition import build_container
from app.config.settings import Settings
from app.main import create_app


@pytest.mark.asyncio
async def test_unhandled_error_includes_request_id_and_logs(capfd: pytest.CaptureFixture[str]) -> None:
    app = create_app()
    container = await build_container(Settings(openai_api_key="", app_env="test"), use_fakes=True)
    app.state.container = container

    class Boom:
        async def handle(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom-for-logging")

    container.extras["chat_service"] = Boom()

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "hello"},
            headers={
                "X-Role": "recruiter",
                "X-User-Id": "u1",
                "X-Request-Id": "test-req-123",
            },
        )

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal_error"
    assert body["request_id"] == "test-req-123"
    assert body["error_type"] == "RuntimeError"
    assert resp.headers.get("X-Request-Id") == "test-req-123"

    captured = capfd.readouterr().out
    assert "chat_handler_failed" in captured
    assert "boom-for-logging" in captured
