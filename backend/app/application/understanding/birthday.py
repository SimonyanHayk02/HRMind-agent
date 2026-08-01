"""Detect birthday questions. Birth dates live only in resume text, so these
always resolve through resume_search, never SQL."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.tools.resume_search.birthday import today_utc

BirthdayScope = Literal["person", "today", "month", "upcoming", "closest"]

# First + optional last name; leading articles are handled outside the group.
_NAME = r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"

_RESERVED_NAMES = frozenset(
    {
        "birthday",
        "birthdays",
        "birth",
        "born",
        "date",
        "dates",
        "dob",
        "age",
        "old",
        "happy",
        "wish",
        "wishes",
        "say",
        "send",
        "tell",
        "show",
        "list",
        "find",
        "get",
        "when",
        "who",
        "whose",
        "which",
        "what",
        "that",
        "this",
        "there",
        "any",
        "all",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "has",
        "have",
        "to",
        "of",
        "for",
        "in",
        "on",
        "and",
        "or",
        "me",
        "us",
        "employee",
        "employees",
        "people",
        "staff",
        "everyone",
        "everybody",
        "someone",
        "anyone",
        "somebody",
        "team",
        "today",
        "tomorrow",
        "month",
        "week",
        "year",
        "upcoming",
        "next",
        "coming",
        "soon",
        "closest",
        "nearest",
        "soonest",
        "current",
        # List deixis / ordinals — never treat as a person name.
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "last",
        "person",
        "persons",
        "former",
        "latter",
        "other",
        "one",
        "ones",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
    }
)

_MONTH_WORDS = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_MONTH_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Any mention of birthday / date of birth / age.
_BIRTHDAY_HINT_RE = re.compile(
    r"\b(?:birth\s*day|birthdays?|date\s+of\s+birth|birth\s*date|\bdob\b|"
    r"\bbday\b|born|how\s+old)\b",
    re.IGNORECASE,
)

_TODAY_RE = re.compile(
    r"\b(?:today|todays|today's|this\s+day|right\s+now)\b", re.IGNORECASE
)
_UPCOMING_RE = re.compile(
    r"\b(?:upcoming|coming\s+up|next|soon|this\s+week|next\s+week|"
    r"this\s+month|next\s+month)\b",
    re.IGNORECASE,
)
# "closest / nearest birthday to today" — single nearest person, not a 30-day list.
_CLOSEST_RE = re.compile(
    r"\b(?:"
    r"closest|nearest|soonest|"
    r"(?:closest|nearest|soonest)\s+(?:one|birthday|birthdays?|person|people)|"
    r"birthday\s+(?:that\s+is\s+)?(?:the\s+)?(?:closest|nearest|soonest)|"
    r"next\s+(?:upcoming\s+)?birthday"
    r")\b",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(rf"\b(?:in|during|for)\s+({_MONTH_WORDS})\b", re.IGNORECASE)
_COHORT_SUBJECT_RE = re.compile(
    r"\b(?:whose|who|anyone|anybody|any|"
    r"which\s+(?:employees?|persons?|people)|"
    r"employees?|persons?|people|staff|"
    r"team|everyone|everybody|list|show|all)\b",
    re.IGNORECASE,
)
# Ordinal / pronoun deixis is a single bound person, not an upcoming cohort.
_PERSON_DEIXIS_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"
    r"\s+(?:person|persons|one|ones)\b|"
    r"\b(?:he|she|him|her|his|hers|they|them|their|theirs)\b",
    re.IGNORECASE,
)
_AGE_RE = re.compile(r"\bhow\s+old\b|\bage\s+of\b|\bwhat\s+age\b", re.IGNORECASE)
_WISH_RE = re.compile(
    r"\b(?:wish|wishes|say|send|greet|congratulate)\b|\bhappy\s+birth\s*day\b",
    re.IGNORECASE,
)

# Ordered: most specific phrasings first.
_PERSON_PATTERNS: list[re.Pattern[str]] = [
    # wish Eva Kim a happy birthday / say happy birthday to Eva Kim
    re.compile(
        rf"\b(?:wish|greet|congratulate)\s+(?:the\s+)?{_NAME}\s+"
        rf"(?:a\s+)?happy\s+birth\s*day\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:say|send|wish)\s+(?:a\s+)?happy\s+birth\s*day\s+(?:to|for)\s+"
        rf"(?:the\s+)?{_NAME}\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\bhappy\s+birth\s*day\s+(?:to|for)\s+(?:the\s+)?{_NAME}\b", re.IGNORECASE),
    # when is Eva Kim's birthday / what is the birthday of Eva Kim
    re.compile(
        rf"\b{_NAME}\s*'s\s+(?:birth\s*day|date\s+of\s+birth|birth\s*date|age)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:birth\s*day|date\s+of\s+birth|birth\s*date)\s+(?:of|for)\s+"
        rf"(?:the\s+)?{_NAME}\b",
        re.IGNORECASE,
    ),
    # how old is Eva Kim / when was Eva Kim born
    re.compile(rf"\bhow\s+old\s+is\s+(?:the\s+)?{_NAME}\b", re.IGNORECASE),
    re.compile(rf"\bwhen\s+(?:was|is)\s+(?:the\s+)?{_NAME}\s+born\b", re.IGNORECASE),
    re.compile(rf"\bwhen\s+(?:is|was)\s+(?:the\s+)?{_NAME}\s+birth\s*day\b", re.IGNORECASE),
    re.compile(rf"\b{_NAME}\s+(?:birth\s*day|date\s+of\s+birth)\b", re.IGNORECASE),
    re.compile(rf"\bage\s+of\s+(?:the\s+)?{_NAME}\b", re.IGNORECASE),
]


@dataclass
class BirthdayRequest:
    matched: bool = False
    scope: BirthdayScope = "person"
    person_name: str | None = None
    month: int | None = None
    wants_age: bool = False
    wants_wish: bool = False


def _normalize_name_token(token: str) -> str:
    """Strip trailing possessive / contraction so "what's" → "what" (reserved)."""
    t = token.strip().lower()
    if t.endswith("'s"):
        t = t[:-2]
    elif t.endswith("'"):
        t = t[:-1]
    return t


def _clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip(" .,?!'")
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.IGNORECASE).strip()
    if not name:
        return None
    tokens = name.split()
    # Drop reserved tokens ("employee Eva", "what's their") — never keep a
    # fully reserved / pronoun-only phrase as a person name.
    kept = [t for t in tokens if _normalize_name_token(t) not in _RESERVED_NAMES]
    if not kept:
        return None
    return " ".join(kept)


def _month_from(question: str) -> int | None:
    m = _MONTH_RE.search(question)
    if not m:
        return None
    key = m.group(1).lower()[:4]
    return _MONTH_NUM.get(key) or _MONTH_NUM.get(key[:3])


_PRIOR_SET_PHRASE_RE = re.compile(
    r"\b("
    r"from there|of them|among them|from those|from them|"
    r"from (?:the )?(?:previous|prior|last) (?:list|set|group|search|results?)|"
    r"(?:the )?(?:previous|prior|last) (?:list|set|group)|"
    r"which ones?"
    r")\b",
    re.IGNORECASE,
)


def extract_birthday(question: str) -> BirthdayRequest:
    """Detect birthday/age questions for one person or for a cohort."""
    q = (question or "").strip()
    if not q:
        return BirthdayRequest()

    has_hint = bool(_BIRTHDAY_HINT_RE.search(q))
    wants_age = bool(_AGE_RE.search(q))
    wants_wish = bool(_WISH_RE.search(q))

    # Closest + prior-set phrasing ("from there", "previous list") even without
    # repeating the word birthday — users clarify that way after a DOB answer.
    if _CLOSEST_RE.search(q) and (has_hint or _PRIOR_SET_PHRASE_RE.search(q)):
        return BirthdayRequest(matched=True, scope="closest")

    if not has_hint:
        return BirthdayRequest()

    for pat in _PERSON_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        name = _clean_name(next((g for g in m.groups() if g), None))
        if name:
            return BirthdayRequest(
                matched=True,
                scope="person",
                person_name=name,
                wants_age=wants_age,
                wants_wish=wants_wish,
            )

    month = _month_from(q)
    if month:
        return BirthdayRequest(matched=True, scope="month", month=month)

    if _TODAY_RE.search(q):
        return BirthdayRequest(matched=True, scope="today", wants_wish=wants_wish)

    if _UPCOMING_RE.search(q):
        # "birthdays this month" is a month question about the current month.
        if re.search(r"\bthis\s+month\b", q, re.IGNORECASE):
            return BirthdayRequest(
                matched=True, scope="month", month=today_utc().month
            )
        return BirthdayRequest(matched=True, scope="upcoming")

    # "first person's date of birth" / "her birthday" — person scope; list
    # referent / pronoun binder supplies the id later.
    if _PERSON_DEIXIS_RE.search(q):
        return BirthdayRequest(matched=True, scope="person", wants_age=wants_age)

    if _COHORT_SUBJECT_RE.search(q):
        return BirthdayRequest(matched=True, scope="upcoming")

    # Birthday wording with no resolvable subject — ask which employee.
    return BirthdayRequest(matched=True, scope="person", wants_age=wants_age)
