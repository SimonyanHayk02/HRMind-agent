"""Reciprocal Rank Fusion over independently ranked retrieval arms.

RRF combines rankings without needing their scores to be comparable, which is
what makes it safe to merge a trigram name score, a full-text rank and a cosine
similarity. Each arm contributes ``weight / (k + rank)``.
"""
from __future__ import annotations

from collections.abc import Sequence

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ranking of ``(id, score)``, best first.

    Ties keep the order in which ids were first seen, so the result is stable
    across runs.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for arm_index, ranking in enumerate(rankings):
        weight = 1.0
        if weights is not None and arm_index < len(weights):
            weight = float(weights[arm_index])
        for rank, item in enumerate(ranking, start=1):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def rank_of(fused: Sequence[tuple[str, float]]) -> dict[str, int]:
    """Position of each id in a fused ranking, 1-based."""
    return {item: index for index, (item, _) in enumerate(fused, start=1)}
