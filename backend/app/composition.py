from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.cache.memory_cache import MemoryCache
from app.adapters.cache.redis_cache import RedisCache
from app.adapters.embeddings.fake_embeddings import FakeEmbeddings
from app.adapters.llm.fake_llm import FakeLLM
from app.adapters.persistence.memory_session_store import MemorySessionStore
from app.config.settings import Settings
from app.ports.cache import CachePort
from app.ports.embeddings import EmbeddingClient
from app.ports.llm import LLMClient
from app.ports.session_store import SessionStore


@dataclass
class AppContainer:
    settings: Settings
    cache: CachePort
    embeddings: EmbeddingClient
    llm: LLMClient
    session_store: SessionStore
    redis: Redis | None = None
    engine: AsyncEngine | None = None
    session_factory: async_sessionmaker[AsyncSession] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    async def aclose(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()
        if self.engine is not None:
            await self.engine.dispose()


async def build_container(settings: Settings, *, use_fakes: bool = False) -> AppContainer:
    redis: Redis | None = None
    cache: CachePort
    session_store: SessionStore
    session_backend = "memory"

    if use_fakes:
        cache = MemoryCache()
        session_store = MemorySessionStore()
        session_backend = "memory"
    else:
        try:
            redis = Redis.from_url(settings.redis_url, decode_responses=True)
            await redis.ping()
            cache = RedisCache(redis)
            from app.adapters.persistence.redis_session_store import RedisSessionStore

            session_store = RedisSessionStore(redis, ttl_seconds=settings.session_ttl_seconds)
            session_backend = "redis"
        except Exception as exc:
            if settings.redis_required:
                raise RuntimeError(
                    f"Redis is required in app_env={settings.app_env!r} for durable chat sessions. "
                    f"Connection failed: {exc}"
                ) from exc
            cache = MemoryCache()
            session_store = MemorySessionStore()
            redis = None
            session_backend = "memory"

    if use_fakes or not settings.openai_api_key:
        embeddings: EmbeddingClient = FakeEmbeddings(
            model_name=settings.embedding_model, dims=min(settings.embedding_dims, 64)
        )
        llm: LLMClient = FakeLLM()
    else:
        from app.adapters.embeddings.cached_embeddings import CachedEmbeddings
        from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
        from app.adapters.llm.openai_chat import OpenAIChatLLM

        base = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dims=settings.embedding_dims,
        )
        embeddings = CachedEmbeddings(base, cache)
        llm = OpenAIChatLLM(api_key=settings.openai_api_key, model=settings.chat_model)

    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    container = AppContainer(
        settings=settings,
        cache=cache,
        embeddings=embeddings,
        llm=llm,
        session_store=session_store,
        redis=redis,
        engine=engine,
        session_factory=session_factory,
        extras={"session_backend": session_backend},
    )
    # Wire chat stack lazily-safe
    try:
        from app.composition_wiring import wire_application_stack

        wire_application_stack(container)
    except Exception:
        pass
    return container
