"""Resolve ordinal / list deixis against the last displayed employee name list.

Ordinals index ``SessionMemory.last_listed`` only — never the raw cohort
``last_employee_ids``, which may update on count-only turns with no visible order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.domain.session import EntityRef, SessionMemory

ReferentKind = Literal["resolved", "clarify", "none"]

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
}

_ORDINAL_ALT = "|".join(
    sorted(_ORDINAL_WORDS.keys(), key=len, reverse=True)
)

# Require a person-like noun after word ordinals so "first name" / "last week"
# never bind. Numeric forms (#2, 2nd) may omit the noun.
_PERSON_NOUN = r"(?:person|persons|people|employee|employees|one|ones)"
_ORDINAL_RE = re.compile(
    rf"(?:^|(?<!\w))(?:the\s+)?(?:"
    rf"(?P<ord>{_ORDINAL_ALT})\s+{_PERSON_NOUN}"
    # Hash ordinals: "#1" has no word-char before the digit, so avoid \b before #.
    rf"|#(?P<num>\d+)(?:\s+{_PERSON_NOUN})?"
    rf"|(?P<ord_num>\d+)(?:st|nd|rd|th)(?:\s+{_PERSON_NOUN})?"
    rf")\b",
    re.IGNORECASE,
)
_LAST_RE = re.compile(
    rf"\b(?:the\s+)?last\s+{_PERSON_NOUN}\b",
    re.IGNORECASE,
)
_FORMER_ONE_RE = re.compile(
    r"\b(?:the\s+)?former(?:\s+(?:one|person|employee))\b",
    re.IGNORECASE,
)
_LATTER_ONE_RE = re.compile(
    r"\b(?:the\s+)?latter(?:\s+(?:one|person|employee))\b",
    re.IGNORECASE,
)
_OTHER_ONE_RE = re.compile(
    r"\b(?:the\s+)?other(?:\s+(?:one|person|employee))\b",
    re.IGNORECASE,
)
_THAT_ONE_RE = re.compile(
    r"\bthat(?:\s+(?:one|person|employee))\b",
    re.IGNORECASE,
)
_TOP_ONE_RE = re.compile(
    r"\b(?:the\s+)?top\s+(?:one|ones|person|employee)\b",
    re.IGNORECASE,
)
# Bare "former" / employment-status language — do not treat as list ordinal.
_BARE_FORMER_RE = re.compile(r"\bformer\b", re.IGNORECASE)

# Plural ordinals → clarify (this pass does not multi-bind).
# "first persons" (typo for possessive) is treated as singular via _ORDINAL_RE;
# "first ones" / "first people" are plural.
_PLURAL_ORDINAL_RE = re.compile(
    rf"\b(?:the\s+)?(?:{_ORDINAL_ALT}|last)\s+(?:ones|people)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ListReferentResult:
    kind: ReferentKind
    employee_id: str | None = None
    display_name: str | None = None
    index: int | None = None  # 1-based when resolved
    clarify_question: str | None = None
    phrase_matched: bool = False


def has_list_referent_phrase(question: str) -> bool:
    """True when the question looks like list deixis / ordinal (even if unbound)."""
    q = question or ""
    if _PLURAL_ORDINAL_RE.search(q):
        return True
    if _FORMER_ONE_RE.search(q) or _LATTER_ONE_RE.search(q) or _OTHER_ONE_RE.search(q):
        return True
    if _THAT_ONE_RE.search(q) or _TOP_ONE_RE.search(q):
        return True
    if _LAST_RE.search(q) or _ORDINAL_RE.search(q):
        return True
    return False


def resolve_list_referent(
    question: str, memory: SessionMemory | None
) -> ListReferentResult:
    """Bind ordinal/deixis to ``last_listed``, or clarify. Never invent a name."""
    q = (question or "").strip()
    if not q or not has_list_referent_phrase(q):
        return ListReferentResult(kind="none")

    listed = list(memory.last_listed) if memory and memory.last_listed else []

    if _PLURAL_ORDINAL_RE.search(q):
        return _clarify(
            listed,
            "Which people did you mean? I can answer about one person at a time.",
            phrase_matched=True,
        )

    # Contrast pairs (need a 2-person list).
    if _FORMER_ONE_RE.search(q):
        return _index(listed, 1, phrase_matched=True)
    if _TOP_ONE_RE.search(q):
        return _index(listed, 1, phrase_matched=True)
    if _LATTER_ONE_RE.search(q):
        return _index(listed, 2 if len(listed) >= 2 else None, phrase_matched=True)
    if _OTHER_ONE_RE.search(q):
        if len(listed) == 2:
            # Without a prior focal person, "the other" is ambiguous — clarify.
            return _clarify(
                listed,
                "Which of the two did you mean?",
                phrase_matched=True,
            )
        return _clarify(
            listed,
            "I need a two-person list to interpret “the other one”.",
            phrase_matched=True,
        )

    if _THAT_ONE_RE.search(q):
        if len(listed) == 1:
            return _resolved(listed[0], 1, phrase_matched=True)
        return _clarify(
            listed,
            "Which person did you mean?",
            phrase_matched=True,
        )

    if _LAST_RE.search(q):
        if not listed:
            return _clarify(listed, None, phrase_matched=True)
        return _resolved(listed[-1], len(listed), phrase_matched=True)

    m = _ORDINAL_RE.search(q)
    if m:
        if m.group("num"):
            n = int(m.group("num"))
        elif m.group("ord_num"):
            n = int(m.group("ord_num"))
        else:
            word = (m.group("ord") or "").lower()
            n = _ORDINAL_WORDS.get(word)
        if n is None:
            return ListReferentResult(kind="none")
        return _index(listed, n, phrase_matched=True)

    # Bare "former" without one/person — not a list referent.
    if _BARE_FORMER_RE.search(q):
        return ListReferentResult(kind="none")

    return ListReferentResult(kind="none")


def _index(
    listed: list[EntityRef], n: int | None, *, phrase_matched: bool
) -> ListReferentResult:
    if n is None or n < 1:
        return _clarify(listed, None, phrase_matched=phrase_matched)
    if not listed:
        return _clarify(
            listed,
            "Which person did you mean? Ask for their names first, then I can "
            "answer about the first, second, and so on.",
            phrase_matched=phrase_matched,
        )
    if n > len(listed):
        return _clarify(
            listed,
            f"I only have {len(listed)} people in that list.",
            phrase_matched=phrase_matched,
        )
    return _resolved(listed[n - 1], n, phrase_matched=phrase_matched)


def _resolved(
    ent: EntityRef, index: int, *, phrase_matched: bool
) -> ListReferentResult:
    return ListReferentResult(
        kind="resolved",
        employee_id=str(ent.employee_id),
        display_name=ent.display_name,
        index=index,
        phrase_matched=phrase_matched,
    )


def _clarify(
    listed: list[EntityRef],
    prefix: str | None,
    *,
    phrase_matched: bool,
) -> ListReferentResult:
    names = [e.display_name for e in listed[:5] if e.display_name]
    if names:
        shown = ", ".join(names)
        if len(listed) > 5:
            shown += f", and {len(listed) - 5} more"
        base = prefix or "Which person did you mean?"
        msg = f"{base} I have: {shown}."
    else:
        msg = prefix or (
            "Which person did you mean? Ask for their names first, then I can "
            "answer about the first, second, and so on."
        )
    return ListReferentResult(
        kind="clarify",
        clarify_question=msg,
        phrase_matched=phrase_matched,
    )
