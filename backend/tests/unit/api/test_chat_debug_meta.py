"""Opt-in ChatResponse.meta and development-only force-repair header."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.composition import build_container
from app.config.settings import Settings
from app.main import create_app


async def _client(app_env: str = "development") -> AsyncClient:
    app = create_app()
    app.state.container = await build_container(
        Settings(openai_api_key="", app_env=app_env, chat_debug_meta=False),
        use_fakes=True,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_meta_absent_without_debug_header() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "hello"},
            headers={"X-Role": "recruiter", "X-User-Id": "u1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("meta") is None


@pytest.mark.asyncio
async def test_meta_present_with_debug_header() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "hello"},
            headers={
                "X-Role": "recruiter",
                "X-User-Id": "u1",
                "X-HRMind-Debug": "1",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    meta = body.get("meta")
    assert isinstance(meta, dict)
    assert meta.get("planner_mode") == "greeting"
    assert "greeting" in (meta.get("plan_nodes") or [])
    assert meta.get("tools_answered") == "greeting" or body.get("tool") == "greeting"


@pytest.mark.asyncio
async def test_force_repair_ignored_outside_development() -> None:
    async with await _client(app_env="production") as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "How many employees do we have?"},
            headers={
                "X-Role": "recruiter",
                "X-User-Id": "u1",
                "X-HRMind-Debug": "1",
                "X-HRMind-Force-Repair": "1",
            },
        )
    assert resp.status_code == 200
    meta = resp.json().get("meta") or {}
    # Production must not honor the probe — repairs stay 0 when selection succeeds once.
    assert int(meta.get("repairs") or 0) == 0


@pytest.mark.asyncio
async def test_force_repair_increments_in_development() -> None:
    async with await _client(app_env="development") as client:
        resp = await client.post(
            "/v1/chat",
            json={"question": "How many employees do we have?"},
            headers={
                "X-Role": "recruiter",
                "X-User-Id": "u1",
                "X-HRMind-Debug": "1",
                "X-HRMind-Force-Repair": "1",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    meta = body.get("meta") or {}
    mode = str(meta.get("planner_mode") or "")
    # Contract under test: force-repair is honored in development.
    # Do not require degraded=False — CI has no Postgres, so sql may still
    # fail after a successful tool-select repair.
    assert mode.startswith("tool_select"), mode
    assert int(meta.get("repairs") or 0) >= 1 or "repair" in mode
