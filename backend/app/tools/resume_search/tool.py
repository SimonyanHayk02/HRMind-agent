from __future__ import annotations

from typing import Any

from app.adapters.cache import keys as cache_keys
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.policies.rbac import require_tool
from app.domain.tools.base import SourceRef, ToolMeta, ToolResult
from app.ports.cache import CachePort
from app.ports.embeddings import EmbeddingClient
from app.ports.vector_store import VectorStore
from app.tools.resume_search.context_builder import build_structured_context
from app.tools.resume_search.reranker import rerank


class ResumeSearchTool:
    def __init__(
        self,
        *,
        embeddings: EmbeddingClient,
        vector_store: VectorStore,
        cache: CachePort,
        top_k: int = 30,
        rerank_top_k: int = 5,
        cache_ttl: int = 1800,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._cache = cache
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self._cache_ttl = cache_ttl
        self._meta = ToolMeta(
            name="resume_search",
            description="Semantic search over employee resumes",
            permissions=[Role.RECRUITER, Role.MANAGER],
            estimated_latency_ms=400,
            cache_policy="retrieval",
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        require_tool(auth, "resume_search")
        question = params.get("question") or params.get("query") or ""
        # Optional cohort filter (SQL→RAG): only keep hits in this employee set.
        scope_ids = {
            str(x)
            for x in (params.get("employee_ids") or [])
            if x
        }
        scope_key = ",".join(sorted(scope_ids)[:80]) if scope_ids else ""
        key = cache_keys.build(
            "retrieval",
            auth,
            q=question,
            model=self._embeddings.model_name,
            top_k=self._top_k,
            scope=scope_key,
        )
        cached = await self._cache.get(key)
        if cached is not None:
            return ToolResult(data=cached, confidence=0.85, cache_hit=True)

        try:
            emb = await self._embeddings.embed(question)
            # Over-fetch when scoping so filtering still yields enough candidates.
            fetch_k = self._top_k * 3 if scope_ids else self._top_k
            hits = await self._vector_store.similarity_search(
                embedding=emb, top_k=fetch_k, tenant_id=auth.tenant_id
            )
            if scope_ids:
                hits = [h for h in hits if str(h.get("employee_id")) in scope_ids]
            ranked = rerank(question, hits, self._rerank_top_k)
            context = build_structured_context(ranked)
            await self._cache.set(key, context, ttl_seconds=self._cache_ttl)
            sources = [
                SourceRef(kind="resume_chunk", ref=str(h.get("id", "")), label=str(h.get("employee_id")))
                for h in ranked
            ]
            return ToolResult(data=context, confidence=0.85, sources=sources)
        except Exception as exc:
            return ToolResult(data={"hits": [], "employee_ids": []}, confidence=0.0, error=str(exc), degraded=True)
