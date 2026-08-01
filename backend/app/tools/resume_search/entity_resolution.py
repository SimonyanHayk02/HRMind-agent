"""Stage 1 of resume retrieval: decide *who* the question is about.

Identity resolution and attribute retrieval are separate problems. An attribute
chunk such as ``Date of Birth: 12 March 1991`` is near-identical across the whole
corpus, so embeddings cannot distinguish people from it; names are matched
lexically and fuzzily instead, while dense retrieval handles descriptive
references ("the Kubernetes engineer in Dubai") over the prose sections.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.adapters.vectorstore.pgvector_store import EXACT_NAME_SCORE
from app.ports.embeddings import EmbeddingClient
from app.ports.vector_store import VectorStore
from app.tools.resume_search.fusion import DEFAULT_K, rank_of, reciprocal_rank_fusion

# Name matches are the strongest identity signal, then exact tokens, then embeddings.
ARM_WEIGHTS = (2.0, 1.5, 1.0)


@dataclass(frozen=True)
class ResolvedEmployee:
    employee_id: str
    name: str
    score: float
    name_score: float = 0.0


@dataclass(frozen=True)
class Resolution:
    """Candidate employees plus how they were found."""

    candidates: list[ResolvedEmployee] = field(default_factory=list)
    via: str = "none"
    note: str | None = None
    # Ranked employee ids per arm, for evaluation and logging.
    arms: dict[str, list[str]] = field(default_factory=dict)

    @property
    def employee_ids(self) -> list[str]:
        return [c.employee_id for c in self.candidates]


def _employee_order(hits: Sequence[dict[str, Any]]) -> list[str]:
    """Employee ids in hit order, first occurrence wins."""
    out: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


async def resolve_employees(
    *,
    query: str,
    store: VectorStore,
    embeddings: EmbeddingClient | None = None,
    exclude_sections: Sequence[str] = (),
    name_limit: int = 40,
    lexical_top_k: int = 40,
    dense_top_k: int = 60,
    max_dense_candidates: int = 3,
    min_similarity: float = 0.35,
    rrf_k: int = DEFAULT_K,
    descriptive_fallback: bool = True,
) -> Resolution:
    """Resolve a person reference to candidate employees.

    An exact name match short-circuits the other arms — including the embedding
    call — and returns every namesake so the caller can disambiguate instead of
    guessing.

    ``descriptive_fallback`` controls what happens when no name matched. It suits
    references like "the Kubernetes engineer in Dubai", where the retrieved prose
    is the only identity signal. Callers holding an actual name must switch it
    off: a name matching nobody has to resolve to nobody, not to whichever
    resumes happened to rank highest.
    """
    q = (query or "").strip()
    if not q:
        return Resolution()

    name_rows = await store.resolve_names(
        q, limit=name_limit, min_similarity=min_similarity
    )
    name_ranked = [str(r["employee_id"]) for r in name_rows]
    exact = [r for r in name_rows if float(r.get("score") or 0) >= EXACT_NAME_SCORE]
    if exact:
        return Resolution(
            candidates=[
                ResolvedEmployee(
                    employee_id=str(r["employee_id"]),
                    name=str(r.get("employee_name") or ""),
                    score=float(r["score"]),
                    name_score=float(r["score"]),
                )
                for r in exact
            ],
            via="name",
            arms={"name": name_ranked},
        )

    fuzzy_rows = [r for r in name_rows if float(r.get("score") or 0) > 0]
    if not fuzzy_rows and not descriptive_fallback:
        return Resolution(via="none", arms={"name": name_ranked})

    filters = {"exclude_sections": list(exclude_sections)} if exclude_sections else None
    lexical_hits = await store.lexical_search(q, top_k=lexical_top_k, filters=filters)
    lexical_ranked = _employee_order(lexical_hits)

    dense_hits: list[dict[str, Any]] = []
    note: str | None = None
    if embeddings is not None:
        try:
            embedding = await embeddings.embed(q)
            dense_hits = await store.similarity_search(
                embedding=embedding, top_k=dense_top_k, filters=filters
            )
        except Exception:  # noqa: BLE001 - keep the lexical arms usable if embeddings fail
            note = "embedding search unavailable; matched on text only"
    dense_ranked = _employee_order(dense_hits)

    fused = reciprocal_rank_fusion(
        [name_ranked, lexical_ranked, dense_ranked], k=rrf_k, weights=ARM_WEIGHTS
    )
    ranks = rank_of(fused)
    arms = {"name": name_ranked, "lexical": lexical_ranked, "dense": dense_ranked}
    names = {str(r["employee_id"]): str(r.get("employee_name") or "") for r in name_rows}
    for hit in (*lexical_hits, *dense_hits):
        eid = str(hit.get("employee_id") or "")
        if eid:
            names.setdefault(eid, str(hit.get("employee_name") or ""))

    fuzzy = {str(r["employee_id"]): float(r["score"]) for r in name_rows}
    if fuzzy:
        candidates = sorted(
            (
                ResolvedEmployee(
                    employee_id=eid,
                    name=names.get(eid, ""),
                    score=dict(fused).get(eid, 0.0),
                    name_score=score,
                )
                for eid, score in fuzzy.items()
            ),
            key=lambda c: ranks.get(c.employee_id, len(ranks) + 1),
        )
        best = candidates[0].name if candidates else ""
        return Resolution(
            candidates=candidates,
            via="fuzzy",
            note=note
            or (f'No exact match for "{q}"; closest is {best}.' if best else None),
            arms=arms,
        )

    candidates = [
        ResolvedEmployee(employee_id=eid, name=names.get(eid, ""), score=score)
        for eid, score in fused[:max_dense_candidates]
    ]
    return Resolution(
        candidates=candidates,
        via="fused" if candidates else "none",
        note=note,
        arms=arms,
    )
