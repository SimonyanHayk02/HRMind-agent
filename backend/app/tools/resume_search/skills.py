"""Closed-vocab skill tokens for post-RAG precision on skill search.

Embeddings still retrieve candidates; before we publish employee_ids we require
the retrieved resume text to mention the skill (or a known alias). That stops
near-neighbors like Docker ranking into a Kubernetes cohort.
"""
from __future__ import annotations

import re
from typing import Any

# Canonical label -> match terms (longest / most specific first in values).
# Keep labels aligned with scripts/generate_resumes.py SKILLS.
_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "Java": ("java",),
    "Go": ("golang", "go"),
    "SQL": ("sql",),
    "Kubernetes": (
        "kubernetes",
        "kubernetees",
        "k8s",
        "ckad",
        "cka",
    ),
    "React": ("react.js", "reactjs", "react"),
    "Machine Learning": ("machine learning", "machine-learning", "ml"),
    "NLP": ("natural language processing", "nlp"),
    "AWS": ("amazon web services", "aws"),
    "Docker": ("docker",),
    "Recruiting": ("recruiting", "recruitment"),
    "Salesforce": ("salesforce",),
    "Accounting": ("accounting",),
    "Product Strategy": ("product strategy",),
}

# Longer labels first so "Machine Learning" wins over a bare "Learning" miss,
# and multi-word aliases are tried before short ones like "ml" / "go".
_CANONICAL_BY_LENGTH = sorted(_SKILL_ALIASES.keys(), key=len, reverse=True)


def canonicalize_skill(raw: str | None) -> str | None:
    """Map a free-text skill mention to its canonical label, if known."""
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip().lower()
    for label, terms in _SKILL_ALIASES.items():
        if text == label.lower() or text in terms:
            return label
    return None


def extract_skill(question: str, *, explicit: str | None = None) -> str | None:
    """Resolve the skill from an explicit param or from the RAG question text."""
    if explicit:
        return canonicalize_skill(explicit) or str(explicit).strip() or None
    q = question or ""
    if not q.strip():
        return None
    # Prefer longest canonical / alias span in the question.
    best: tuple[int, str] | None = None
    for label in _CANONICAL_BY_LENGTH:
        for term in _SKILL_ALIASES[label]:
            pattern = _term_pattern(term)
            match = pattern.search(q)
            if match:
                span = match.end() - match.start()
                if best is None or span > best[0]:
                    best = (span, label)
    return best[1] if best else None


def content_mentions_skill(content: str, skill: str) -> bool:
    """True when resume chunk text contains the skill or one of its aliases."""
    canon = canonicalize_skill(skill) or skill.strip()
    terms = _SKILL_ALIASES.get(canon) if canon in _SKILL_ALIASES else (canon.lower(),)
    text = content or ""
    if not text or not terms:
        return False
    return any(_term_pattern(term).search(text) for term in terms)


def filter_hits_for_skill(
    hits: list[dict[str, Any]], skill: str | None
) -> list[dict[str, Any]]:
    """Drop RAG hits whose text does not lexically mention ``skill``.

    When ``skill`` is missing/unknown we leave hits unchanged so open-ended
    resume search still returns embedding neighbors.
    """
    if not skill:
        return hits
    return [h for h in hits if content_mentions_skill(str(h.get("content") or ""), skill)]


def _term_pattern(term: str) -> re.Pattern[str]:
    """Word-ish boundary match; spaces in multi-word terms also allow hyphens."""
    escaped = re.escape(term.lower()).replace(r"\ ", r"[\s\-]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)
