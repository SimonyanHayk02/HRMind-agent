from __future__ import annotations

import re

import numpy as np

from app.domain.enums import RouterLabel
from app.ports.embeddings import EmbeddingClient

CHITCHAT_EXAMPLES = [
    "how are you",
    "what's up",
    "tell me a joke",
    "who made you",
]
NEEDS_TOOLS_EXAMPLES = [
    "how many engineers in berlin",
    "find python developers",
    "who is alice's manager",
    "show employees hired after 2023",
    "search resumes for kubernetes",
]

_HR_HINTS = re.compile(
    r"\b("
    r"employee|employees|department|engineering|sales|finance|product|operations|people|"
    r"manager|salary|hired|hire|resume|skill|skills|developer|headcount|count|"
    r"how many|how much|python|kubernetes|aws|berlin|dubai|roster|profile|"
    r"who|about|tell me|find|search|list|show|names?|them|those|their|there|"
    r"of them|of those|among them|name them|say their"
    r")\b",
    re.I,
)


def _cos(a: list[float], b: list[float]) -> float:
    va = np.asarray(a)
    vb = np.asarray(b)
    return float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9))


class EmbeddingRouter:
    def __init__(self, embeddings: EmbeddingClient, *, threshold: float = 0.75) -> None:
        self._embeddings = embeddings
        self._threshold = threshold
        self._centroids: dict[RouterLabel, list[float]] | None = None

    async def _ensure(self) -> None:
        if self._centroids is not None:
            return
        chat_vecs = await self._embeddings.embed_many(CHITCHAT_EXAMPLES)
        tool_vecs = await self._embeddings.embed_many(NEEDS_TOOLS_EXAMPLES)
        self._centroids = {
            RouterLabel.CHITCHAT: np.mean(np.asarray(chat_vecs), axis=0).tolist(),
            RouterLabel.NEEDS_TOOLS: np.mean(np.asarray(tool_vecs), axis=0).tolist(),
        }

    async def route(self, question: str) -> RouterLabel:
        # Strong keyword prior: HR analytics/search should never fall into chitchat.
        if _HR_HINTS.search(question):
            return RouterLabel.NEEDS_TOOLS

        await self._ensure()
        assert self._centroids is not None
        q = await self._embeddings.embed(question)
        scores = {label: _cos(q, vec) for label, vec in self._centroids.items()}
        return max(scores, key=scores.get)
