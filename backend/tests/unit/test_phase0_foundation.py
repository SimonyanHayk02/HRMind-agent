import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.cache.keys import build, embedding_key
from app.adapters.cache.memory_cache import MemoryCache
from app.config.settings import Settings
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.errors import CacheKeyError
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="", app_env="test")


@pytest.mark.asyncio
async def test_health_ok() -> None:
    from app.composition import build_container

    app = create_app()
    app.state.container = await build_container(Settings(openai_api_key="", app_env="test"), use_fakes=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_settings_load(settings: Settings) -> None:
    assert settings.chat_model
    assert settings.embedding_model
    assert settings.sql_max_rows > 0


@pytest.mark.asyncio
async def test_memory_cache_roundtrip() -> None:
    cache = MemoryCache()
    await cache.set("k", {"a": 1}, ttl_seconds=60)
    assert await cache.get("k") == {"a": 1}
    await cache.delete("k")
    assert await cache.get("k") is None


def test_cache_key_requires_auth() -> None:
    with pytest.raises(CacheKeyError):
        build("sql", None, q="x")

    auth = AuthContext(user_id="u", tenant_id="t", role=Role.RECRUITER)
    key = build("sql", auth, q="x")
    assert key.startswith("hrmind:sql:")
    emb = embedding_key(model="m", text="hello")
    assert "embedding" in emb
