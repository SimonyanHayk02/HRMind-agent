from __future__ import annotations

from typing import Any


def rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Lightweight lexical reranker (cross-encoder substitute for v1/tests)."""
    q_terms = {t.lower() for t in query.split() if len(t) > 2}

    def score(hit: dict[str, Any]) -> float:
        text = (hit.get("content") or "").lower()
        overlap = sum(1 for t in q_terms if t in text)
        base = float(hit.get("score") or 0.0)
        return base + 0.1 * overlap

    ranked = sorted(hits, key=score, reverse=True)
    return ranked[:top_k]
