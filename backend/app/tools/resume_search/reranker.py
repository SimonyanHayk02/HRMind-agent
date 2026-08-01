from __future__ import annotations

from typing import Any


def rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Lightweight lexical reranker (cross-encoder substitute for v1/tests)."""
    q_terms = {t.lower() for t in query.split() if len(t) > 1}
    q_full = (query or "").strip().lower()

    def score(hit: dict[str, Any]) -> float:
        text = (hit.get("content") or "").lower()
        name = (
            hit.get("employee_name")
            or (hit.get("metadata") or {}).get("employee_name")
            or ""
        ).lower()
        overlap = sum(1 for t in q_terms if t in text)
        name_overlap = sum(1 for t in q_terms if t in name)
        base = float(hit.get("score") or 0.0)
        boost = 0.0
        if q_full and name:
            if q_full == name:
                boost += 5.0
            elif q_full in name or name in q_full:
                boost += 3.0
            boost += 1.5 * name_overlap
        return base + 0.1 * overlap + boost

    ranked = sorted(hits, key=score, reverse=True)
    return ranked[:top_k]
