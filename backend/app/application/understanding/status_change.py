from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.places import PLACE_ALT, canon_place
from app.domain.session import SessionMemory

PENDING_SET_STATUS_KEY = "pending_set_status"
# Location cohort writes that asked for an explicit confirm before committing.
PENDING_STATUS_MUTATION_KEY = "pending_status_mutation"

_TRUE_WORDS = frozenset(
    {"true", "yes", "on", "1", "enabled", "enable", "active", "activate"}
)
_FALSE_WORDS = frozenset(
    {
        "false",
        "no",
        "off",
        "0",
        "disabled",
        "disable",
        "inactive",
        "deactivate",
    }
)
_BOOL = (
    r"(true|false|yes|no|on|off|1|0|enabled|enable|disabled|disable|"
    r"active|inactive|activate|deactivate)"
)
# First + optional last; "the" is handled outside the capture group.
_NAME = r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)"
_EMAIL = r"([A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+)"
_UUID = (
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

_RESERVED_NAMES = frozenset(
    {
        "set",
        "change",
        "update",
        "mark",
        "enable",
        "disable",
        "status",
        "flag",
        "employee",
        "employees",
        "people",
        "staff",
        "everyone",
        "everybody",
        "someone",
        "anyone",
        "all",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "and",
        "or",
        "which",
        "who",
        "that",
        "this",
        "these",
        "those",
        "current",
        "same",
        "said",
        "above",
        "are",
        "is",
        "living",
        "based",
        "located",
        "working",
        "from",
        "in",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
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
    }
)

# "the current employee", "this one", … — bind from session focus, not as a name.
_DEICTIC_SUBJECT_RE = re.compile(
    r"^(?:the\s+)?"
    r"(?:current|this|that|same|said|above)\s+"
    r"(?:employee|person|one|candidate|profile)$"
    r"|^(?:the\s+)?(?:current|this|that)\s+one$"
    r"|^(?:him|her|them|they|he|she)$",
    re.I,
)

_LOC_WORDS = PLACE_ALT

# Location cohort: update status of employees living in Dubai to true
_LOCATION_STATUS_RE = re.compile(
    rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+(?:of|for)\s+"
    rf"(?:(?:all\s+)?(?:the\s+)?(?:employees?|people|staff|everyone|everybody)\s+)?"
    rf"(?:(?:who|which|that)\s+(?:are\s+)?)?"
    rf"(?:living|based|located|working)?\s*"
    rf"(?:in|from)\s+({_LOC_WORDS})\s+to\s+{_BOOL}\b",
    re.I,
)
_LOCATION_STATUS_RE2 = re.compile(
    rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+to\s+{_BOOL}\s+for\s+"
    rf"(?:(?:all\s+)?(?:the\s+)?(?:employees?|people|staff|everyone|everybody)\s+)?"
    rf"(?:(?:who|which|that)\s+(?:are\s+)?)?"
    rf"(?:living|based|located|working)?\s*"
    rf"(?:in|from)\s+({_LOC_WORDS})\b",
    re.I,
)

# Ordered: more specific patterns first.
_STATUS_CHANGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+(?:of|for)\s+{_EMAIL}\s+to\s+{_BOOL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+to\s+{_BOOL}\s+for\s+{_EMAIL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+(?:of|for)\s+{_UUID}\s+to\s+{_BOOL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+to\s+{_BOOL}\s+for\s+{_UUID}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+{_NAME}\s*'s\s+status\s+to\s+{_BOOL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+(?:of|for)\s+(?:the\s+)?{_NAME}\s+to\s+{_BOOL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+to\s+{_BOOL}\s+for\s+(?:the\s+)?{_NAME}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:set|change|update)\s+(?:the\s+)?{_NAME}\s+status\s+to\s+{_BOOL}\b",
        re.I,
    ),
    re.compile(
        rf"\bmark\s+(?:the\s+)?{_NAME}\s+(?:(?:status|flag)\s+)?(?:as|to)\s+{_BOOL}\b",
        re.I,
    ),
    re.compile(
        rf"\b(enable|disable)\s+(?:the\s+)?(?:status|flag)\s+(?:of|for)\s+(?:the\s+)?{_NAME}\b",
        re.I,
    ),
]

_STATUS_INTENT_RE = re.compile(
    r"\b(?:"
    r"(?:set|change|update|mark)\b.+\bstatus\b|"
    r"\bstatus\b.+\b(?:to|as|=)\b.+\b"
    r"(?:true|false|yes|no|on|off|active|inactive|enabled|disabled)\b|"
    r"\bmark\b.+\b(?:as|to)\b.+\b(?:active|inactive|true|false)\b|"
    r"(?:enable|disable|activate|deactivate)\s+(?:the\s+)?(?:status|flag)\b"
    r")",
    re.I,
)

# Pronoun subjects — person is bound later via SessionMemory / person_bindings.
_PRONOUN_STATUS_RE = re.compile(
    rf"\b(?:set|change|update)\s+(?:the\s+)?"
    rf"(?:her|his|their)\s+status\s+to\s+(?P<v1>{_BOOL})\b|"
    rf"\b(?:set|change|update)\s+(?:the\s+)?status\s+(?:of|for)\s+"
    rf"(?:her|him|them|she|he|they)\s+to\s+(?P<v2>{_BOOL})\b|"
    rf"\bmark\s+(?:her|him|them)\s+(?:(?:status|flag)\s+)?"
    rf"(?:as|to)\s+(?P<v3>{_BOOL})\b",
    re.I,
)

_EMAIL_RE = re.compile(_EMAIL, re.I)
_UUID_RE = re.compile(_UUID, re.I)


@dataclass
class StatusChangeRequest:
    matched: bool = False
    status_value: bool | None = None
    person_name: str | None = None
    email: str | None = None
    employee_id: str | None = None
    city: str | None = None
    country: str | None = None


def _parse_bool(raw: str) -> bool | None:
    key = (raw or "").strip().lower()
    if key in _TRUE_WORDS:
        return True
    if key in _FALSE_WORDS:
        return False
    return None


def _clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip(" .,?!")
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
    # "Alice Nguyen's status" salvage — drop trailing possessive.
    name = re.sub(r"'s$", "", name, flags=re.I).strip()
    if not name:
        return None
    if _DEICTIC_SUBJECT_RE.match(name):
        return None
    if name.lower() in _RESERVED_NAMES:
        return None
    # Reject multi-word if first token is reserved ("update status" salvage)
    first = name.split()[0].lower()
    if first in _RESERVED_NAMES:
        return None
    return name


def is_deictic_status_subject(question: str) -> bool:
    """True when the status subject is session deixis (current/this employee)."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(?:status)\s+(?:of|for)\s+(?:the\s+)?"
            r"(?:current|this|that|same)\s+(?:employee|person|one)\b|"
            r"\b(?:current|this|that)\s+(?:employee|person|one)\s+(?:status|flag)\b",
            q,
            re.I,
        )
    )


def _canon_location(raw: str) -> tuple[str | None, str | None]:
    """Return (city, country) from a location phrase."""
    return canon_place(raw)


def bind_status_subject_from_memory(
    *,
    person_name: str | None,
    email: str | None,
    employee_id: str | None,
    city: str | None,
    country: str | None,
    memory: SessionMemory | None,
) -> tuple[str | None, str | None]:
    """Fill (employee_id, display_name) from session focus when the write omits a subject.

    Prefer a single active referent, then he/she person_bindings. Does not invent a
    subject for multi-person cohorts or location-wide updates.
    """
    if person_name or email or employee_id or city or country:
        return employee_id, None
    if memory is None:
        return None, None

    if memory.active_referent and len(memory.active_referent.ids) == 1:
        eid = str(memory.active_referent.ids[0])
        return eid, _display_name_for_id(memory, eid)

    for key in ("she", "he", "her", "him", "they", "them"):
        eid = memory.person_bindings.get(key)
        if eid:
            return str(eid), _display_name_for_id(memory, str(eid))

    if memory.last_listed and len(memory.last_listed) == 1:
        eid = str(memory.last_listed[0].employee_id)
        return eid, memory.last_listed[0].display_name or _display_name_for_id(memory, eid)

    return None, None


def _display_name_for_id(memory: SessionMemory, employee_id: str) -> str | None:
    for ent in memory.last_listed:
        if str(ent.employee_id) == employee_id:
            return ent.display_name or None
    for ent in memory.entity_memory:
        if str(ent.employee_id) == employee_id:
            return ent.display_name or None
    if memory.active_referent and memory.active_referent.label:
        return memory.active_referent.label
    return None


def extract_status_change(question: str) -> StatusChangeRequest:
    """Detect agent-flag status updates (person, email, id, or location cohort)."""
    q = (question or "").strip()
    if not q:
        return StatusChangeRequest()

    # Pronoun subject — leave person_name unset; planner binds via memory.
    pron = _PRONOUN_STATUS_RE.search(q)
    if pron:
        raw_val = pron.group("v1") or pron.group("v2") or pron.group("v3")
        return StatusChangeRequest(matched=True, status_value=_parse_bool(raw_val or ""))

    # Location cohort first (employees living in Dubai, …)
    for pat in (_LOCATION_STATUS_RE, _LOCATION_STATUS_RE2):
        m = pat.search(q)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        if len(groups) < 2:
            continue
        # RE1: (location, bool)  RE2: (bool, location)
        if _parse_bool(groups[0]) is not None:
            value = _parse_bool(groups[0])
            loc = groups[1]
        else:
            loc = groups[0]
            value = _parse_bool(groups[1])
        city, country = _canon_location(loc)
        if city or country:
            return StatusChangeRequest(
                matched=True,
                status_value=value,
                city=city,
                country=country,
            )

    for pat in _STATUS_CHANGE_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        if not groups:
            continue

        if groups[0].lower() in {"enable", "disable"} and len(groups) >= 2:
            name = _clean_name(groups[1])
            return StatusChangeRequest(
                matched=True,
                person_name=name,
                status_value=groups[0].lower() == "enable",
            )

        subject = None
        value = None
        if len(groups) >= 2:
            if _parse_bool(groups[0]) is not None:
                value = _parse_bool(groups[0])
                subject = groups[1]
            else:
                subject = groups[0]
                value = _parse_bool(groups[1]) if len(groups) > 1 else None

        email = None
        employee_id = None
        name = None
        if subject:
            if _EMAIL_RE.fullmatch(subject.strip()):
                email = subject.strip()
            elif _UUID_RE.fullmatch(subject.strip()):
                employee_id = subject.strip().lower()
            else:
                name = _clean_name(subject)
                if name is None:
                    # Matched a reserved word as "name" — keep scanning
                    continue

        return StatusChangeRequest(
            matched=True,
            person_name=name,
            status_value=value,
            email=email,
            employee_id=employee_id,
        )

    if _STATUS_INTENT_RE.search(q):
        bool_m = re.search(rf"\b{_BOOL}\b", q, re.I)
        value = _parse_bool(bool_m.group(1)) if bool_m else None

        # Location salvage: "… in Dubai … true/false"
        loc_m = re.search(
            rf"\b(?:living|based|located|working)?\s*(?:in|from)\s+({_LOC_WORDS})\b",
            q,
            re.I,
        )
        if loc_m and value is not None:
            city, country = _canon_location(loc_m.group(1))
            if city or country:
                return StatusChangeRequest(
                    matched=True,
                    status_value=value,
                    city=city,
                    country=country,
                )

        email_m = _EMAIL_RE.search(q)
        if email_m:
            return StatusChangeRequest(
                matched=True, status_value=value, email=email_m.group(1)
            )
        uuid_m = _UUID_RE.search(q)
        if uuid_m:
            return StatusChangeRequest(
                matched=True,
                status_value=value,
                employee_id=uuid_m.group(1).lower(),
            )
        name_m = re.search(
            rf"\b(?:for|of)\s+(?:the\s+)?{_NAME}\b|"
            rf"\b{_NAME}\s+(?:status|flag)\b|"
            rf"\b{_NAME}\s*'s\s+status\b",
            q,
            re.I,
        )
        name = None
        if name_m:
            name = _clean_name(next((g for g in name_m.groups() if g), None))
        return StatusChangeRequest(matched=True, person_name=name, status_value=value)

    return StatusChangeRequest()


_CONFIRM_STATUS_RE = re.compile(
    r"^\s*(?:yes[,!]?\s+)?(?:please\s+)?confirm(?:\s+(?:the\s+)?status(?:\s+update)?)?\s*[.!?]?\s*$"
    r"|^\s*(?:yes|ok|okay|proceed|go ahead)\s*[.!?]?\s*$",
    re.I,
)


def is_confirm_status_update(question: str) -> bool:
    """True when the user is confirming a pending multi-person status write."""
    q = (question or "").strip()
    if not q or len(q) > 80:
        return False
    if extract_status_change(q).matched:
        return False
    # Prefer the explicit phrase the clarify copy asks for.
    if re.search(r"\bconfirm(?:\s+(?:the\s+)?status(?:\s+update)?)?\b", q, re.I):
        return True
    return bool(_CONFIRM_STATUS_RE.match(q))


def looks_like_bare_person_name(question: str) -> bool:
    """True when the utterance is essentially just a person name (clarify reply)."""
    q = (question or "").strip().strip(" .,?!")
    if not q or len(q) > 60:
        return False
    if extract_status_change(q).matched:
        return False
    if re.search(
        r"\b(how|what|who|where|when|why|list|show|find|count|set|change|update|"
        r"status|employees?|department|knows?|python|java)\b",
        q,
        re.I,
    ):
        return False
    if not re.fullmatch(
        r"[A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+){0,2}",
        q,
    ):
        return False
    tokens = [t.lower() for t in q.split()]
    blocked = _RESERVED_NAMES | {
        "engineering",
        "sales",
        "finance",
        "product",
        "operations",
        "people",
        "berlin",
        "dubai",
        "germany",
        "uae",
    }
    return not any(t in blocked for t in tokens)
