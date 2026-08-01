"""Per-turn context views — what residual LLM/slots may see of session memory.

Durable ``SessionMemory`` is unchanged; packing redacts by ``ContextNeed``.
"""
from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from app.application.planning.heuristic_planner import (
    _PRONOUN_ONLY_RE,
    refers_to_prior_set,
)
from app.application.understanding.list_referents import has_list_referent_phrase
from app.application.understanding.person_existence import (
    NON_PERSON,
    extract_person_existence_name,
)
from app.domain.session import SessionMemory

_LOOKUP_NAME_RE = re.compile(
    r"\b(?:"
    r"find|lookup|look\s+up|search\s+for|looking\s+for|"
    r"show\s+me|who\s+is|tell\s+me\s+about|what\s+about|profile\s+of|"
    r"named|called"
    r")\s+"
    r"(?:an?\s+)?"
    r"([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)",
    re.I,
)

_ELLIPTICAL_ATTR_RE = re.compile(
    r"\b("
    r"education|email|department|position|title|job\s*title|"
    r"manager|status|salary|hire\s*date|hired|"
    r"location|live[sd]?|based|from|city|country|"
    r"birthday|born|age|dob|phone"
    r")\b",
    re.I,
)

_MAX_ELLIPTICAL_TOKENS = 8


class ContextNeed(StrEnum):
    LIST_ORDINAL = "list_ordinal"
    ANAPHORA_NAMED = "anaphora_named"
    FRESH = "fresh"
    ANAPHORA = "anaphora"
    ELLIPTICAL_PERSON = "elliptical_person"


def _normalize_name(name: str) -> str:
    return " ".join((name or "").lower().split())


def _focal_aliases(memory: SessionMemory | None) -> set[str]:
    aliases: set[str] = set()
    if not memory:
        return aliases
    if memory.active_referent and memory.active_referent.label:
        aliases.add(_normalize_name(memory.active_referent.label))
    bound_ids = {
        str(v)
        for k, v in (memory.person_bindings or {}).items()
        if k in {"he", "she", "him", "her", "his", "hers"} and v
    }
    for ent in memory.entity_memory or []:
        eid = str(ent.employee_id)
        if bound_ids and eid not in bound_ids and len(bound_ids) == 1:
            # Still collect all entities for name matching.
            pass
        aliases.add(_normalize_name(ent.display_name))
        for a in ent.aliases or []:
            aliases.add(_normalize_name(a))
    if len(memory.last_employee_ids) == 1:
        eid = str(memory.last_employee_ids[0])
        for ent in memory.entity_memory or []:
            if str(ent.employee_id) == eid:
                aliases.add(_normalize_name(ent.display_name))
                for a in ent.aliases or []:
                    aliases.add(_normalize_name(a))
    return {a for a in aliases if a}


def _is_valid_person_token(name: str) -> bool:
    tokens = [t.lower() for t in re.split(r"[\s\-]+", name) if t]
    if not tokens or any(t in NON_PERSON for t in tokens):
        return False
    return True


def extract_lookup_name_candidate(question: str) -> str | None:
    """Name from existence asks or lookup verbs — not bare TitleCase."""
    existence = extract_person_existence_name(question)
    if existence:
        return existence
    q = (question or "").strip()
    m = _LOOKUP_NAME_RE.search(q)
    if not m:
        return None
    name = " ".join(m.group(1).split())
    if not _is_valid_person_token(name):
        return None
    return name


def _is_foreign_name(name: str | None, memory: SessionMemory | None) -> bool:
    if not name:
        return False
    norm = _normalize_name(name)
    if not norm:
        return False
    aliases = _focal_aliases(memory)
    if not aliases:
        return True
    if norm in aliases:
        return False
    # First-name overlap with focal (e.g. "Alice" vs "Alice Nguyen").
    for alias in aliases:
        if norm == alias.split()[0] or alias == norm.split()[0]:
            return False
        if norm in alias or alias in norm:
            return False
    return True


def _has_singular_binding(memory: SessionMemory | None) -> bool:
    if not memory or not memory.person_bindings:
        return False
    ids = {
        str(v)
        for k, v in memory.person_bindings.items()
        if k in {"he", "she", "him", "her", "his", "hers"} and v
    }
    if len(ids) == 1:
        return True
    if len(memory.last_employee_ids) == 1 and memory.entity_memory:
        return True
    return False


def _is_elliptical_person(question: str, memory: SessionMemory | None) -> bool:
    q = (question or "").strip()
    if not q or not _has_singular_binding(memory):
        return False
    tokens = q.split()
    if len(tokens) > _MAX_ELLIPTICAL_TOKENS:
        return False
    if extract_lookup_name_candidate(q):
        return False
    if refers_to_prior_set(q) and not _PRONOUN_ONLY_RE.search(q):
        # Cohort refine is anaphora, not elliptical person.
        return False
    if not _ELLIPTICAL_ATTR_RE.search(q):
        return False
    # Reject role/org phrasing that isn't a person attribute follow-up.
    if re.search(r"\b(requirements?|policy|package|benefits?|openings?)\b", q, re.I):
        return False
    return True


def classify_context_need(
    question: str,
    memory: SessionMemory | None,
) -> ContextNeed:
    """Classify what prior context the residual LLM may use this turn.

    Priority (first match wins): list_ordinal → anaphora_named → fresh (foreign
    name) → anaphora → elliptical_person → fresh.
    """
    q = (question or "").strip()
    refers = refers_to_prior_set(q)
    foreign = extract_lookup_name_candidate(q)
    is_foreign = _is_foreign_name(foreign, memory)

    if has_list_referent_phrase(q):
        return ContextNeed.LIST_ORDINAL

    if refers and is_foreign:
        return ContextNeed.ANAPHORA_NAMED

    if is_foreign:
        return ContextNeed.FRESH

    if refers or (_PRONOUN_ONLY_RE.search(q) and not is_foreign):
        # Pronoun person asks need bindings; cohort "them" needs ids.
        return ContextNeed.ANAPHORA

    if _is_elliptical_person(q, memory):
        return ContextNeed.ELLIPTICAL_PERSON

    return ContextNeed.FRESH


def redact_planner_packet(
    packet: dict[str, Any],
    need: ContextNeed,
    *,
    memory: SessionMemory | None = None,
) -> dict[str, Any]:
    """Apply context_need redaction to a fully built planner packet."""
    del memory  # reserved if we need focal entity injection later
    out = dict(packet)
    out["context_need"] = need.value

    if need == ContextNeed.FRESH:
        out["recent_messages"] = []
        out["last_employee_ids"] = []
        out["last_employee_ids_truncated"] = False
        out["last_listed"] = []
        out["active_referent"] = None
        out["last_focus"] = None
        out["constraints"] = []
        out["entities"] = []
        out["named_sets"] = {}
        out["person_bindings"] = {}
        out["tool_facts"] = []
        out["summary"] = ""
        return out

    if need == ContextNeed.LIST_ORDINAL:
        out["last_employee_ids"] = []
        out["last_employee_ids_truncated"] = False
        out["active_referent"] = None
        out["constraints"] = []
        out["named_sets"] = {}
        out["entities"] = []
        out["tool_facts"] = []
        out["summary"] = ""
        # Keep last_listed + person_bindings + last_focus + short recent.
        return out

    if need == ContextNeed.ELLIPTICAL_PERSON:
        out["last_employee_ids"] = []
        out["last_employee_ids_truncated"] = False
        out["last_listed"] = []
        out["active_referent"] = None
        out["constraints"] = []
        out["named_sets"] = {}
        out["tool_facts"] = []
        out["summary"] = ""
        out["recent_messages"] = []
        # Keep person_bindings + last_focus; trim entities to focal if possible.
        bindings = out.get("person_bindings") or {}
        focal_ids = {
            str(v)
            for k, v in bindings.items()
            if k in {"he", "she", "him", "her", "his", "hers"} and v
        }
        if focal_ids and isinstance(out.get("entities"), list):
            out["entities"] = [
                e
                for e in out["entities"]
                if isinstance(e, dict) and str(e.get("employee_id")) in focal_ids
            ][:1]
        return out

    # anaphora / anaphora_named — keep current rich packet
    return out


def redact_turn_digest(
    digest: dict[str, Any],
    need: ContextNeed,
) -> dict[str, Any]:
    """Apply context_need redaction to a slot-extractor turn digest."""
    out = dict(digest)
    out["context_need"] = need.value

    if need == ContextNeed.FRESH:
        out["recent_messages"] = []
        out["last_listed"] = []
        out["person_bindings"] = {}
        out["last_focus"] = None
        out["cohort_size"] = 0
        out["has_prior_cohort"] = False
        return out

    if need == ContextNeed.LIST_ORDINAL:
        out["cohort_size"] = 0
        out["has_prior_cohort"] = False
        return out

    if need == ContextNeed.ELLIPTICAL_PERSON:
        out["recent_messages"] = []
        out["last_listed"] = []
        out["cohort_size"] = 0
        out["has_prior_cohort"] = False
        return out

    return out


def lint_fresh_scope(
    plan: Any,
    need: ContextNeed,
    question: str,
) -> str | None:
    """Return an error string if a fresh plan wrongly scopes to a prior cohort.

    Only applies to ``fresh`` turns. Ordinal/anaphora plans may legally carry
    resolved employee_ids from last_listed / prior cohort.
    """
    if need != ContextNeed.FRESH:
        return None
    nodes = getattr(plan, "nodes", None) or []
    if not nodes:
        return None
    lookup = extract_lookup_name_candidate(question)
    lookup_l = (lookup or "").lower()

    for node in nodes:
        name = getattr(node, "name", "") or ""
        params = getattr(node, "params", None) or {}
        bindings = getattr(node, "input_bindings", None) or {}
        if name not in {"sql", "resume_search", "employee"}:
            continue
        if params.get("use_session_cohort") or bindings.get("employee_ids"):
            return f"{name}: session cohort binding on fresh turn"
        filters = params.get("filters") or {}
        eids = params.get("employee_ids") or filters.get("employee_ids")
        if eids:
            person_name = str(params.get("name") or "").lower()
            if name == "employee" and lookup_l and lookup_l in person_name:
                continue
            if name == "employee" and params.get("action") == "by_name" and person_name:
                continue
            # resume_search with explicit name + optional id scope is OK
            if name == "resume_search" and person_name and (
                not lookup_l or lookup_l in person_name
            ):
                continue
            return f"{name}: employee_ids scope without fresh by-name target"
        if name == "employee" and params.get("action") == "by_id":
            if not lookup_l:
                return "employee: by_id on fresh turn without lookup name"
    return None


def minimal_planner_packet(
    *,
    question: str,
    role: str,
    tools: list[dict],
    schema_catalog: dict,
    query_state: dict | None,
    context_need: str,
) -> dict[str, Any]:
    """Stripped packet for LLM retry after poison / lint failure."""
    from app.application.memory.context_budget import estimate_tokens

    packet = {
        "question": question,
        "role": role,
        "tools": tools,
        "schema_catalog": schema_catalog,
        "query_state": query_state,
        "context_need": context_need,
        "recent_messages": [],
        "last_employee_ids": [],
        "last_employee_ids_truncated": False,
        "last_listed": [],
        "active_referent": None,
        "last_focus": None,
        "constraints": [],
        "entities": [],
        "named_sets": {},
        "person_bindings": {},
        "tool_facts": [],
        "summary": "",
    }
    packet["approx_tokens"] = estimate_tokens(str(packet))
    return packet
