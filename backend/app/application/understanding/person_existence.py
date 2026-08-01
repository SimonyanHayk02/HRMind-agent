"""Detect "do we have Sofia?"-style person existence asks."""
from __future__ import annotations

import re

# Whole-utterance existence / membership checks (not skill/dept headcount).
_HAVE_PERSON_RE = re.compile(
    r"^\s*(?:"
    r"do\s+we\s+have|"
    r"have\s+we\s+(?:got|hired)|"
    r"is\s+there(?:\s+anyone\s+named)?|"
    r"anyone\s+named|"
    r"do\s+you\s+have|"
    r"is"
    r")\s+"
    r"(?:an?\s+)?"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"
    r"(?:\s+(?:here|on\s+(?:the\s+)?(?:team|staff)|"
    r"in\s+(?:the\s+)?(?:company|org(?:anization)?|directory)|"
    r"an?\s+employee|working\s+here))?"
    r"\s*\??\s*$",
    re.I,
)

NON_PERSON = frozenset(
    {
        "anyone",
        "someone",
        "everybody",
        "everyone",
        "anybody",
        "people",
        "employees",
        "employee",
        "engineers",
        "engineer",
        "developers",
        "developer",
        "staff",
        "them",
        "any",
        "openings",
        "roles",
        "headcount",
        "engineering",
        "sales",
        "finance",
        "product",
        "operations",
        "python",
        "java",
        "react",
        "kubernetes",
        "aws",
        "docker",
        "sql",
        "remote",
        "managers",
        "manager",
        # Question / count phrasing after "find …" must not become a person name.
        "how",
        "many",
        "much",
        "which",
        "what",
        "where",
        "when",
        "why",
        "who",
        "instead",
        "work",
        "works",
        "working",
        "know",
        "knows",
        "list",
        "show",
        "names",
        "name",
    }
)


def extract_person_existence_name(question: str) -> str | None:
    """Return a person name when the question asks if that person exists."""
    q = (question or "").strip()
    if not q:
        return None
    m = _HAVE_PERSON_RE.match(q)
    if not m:
        return None
    name = " ".join(m.group(1).split())
    tokens = [t.lower() for t in re.split(r"[\s\-]+", name) if t]
    if not tokens or any(t in NON_PERSON for t in tokens):
        return None
    # Avoid treating department-only or skill-only tokens as people.
    if len(tokens) == 1 and tokens[0] in NON_PERSON:
        return None
    return name
