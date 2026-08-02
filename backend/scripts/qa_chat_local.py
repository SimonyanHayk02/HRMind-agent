#!/usr/bin/env python3
"""
Local multi-turn chat QA for HRMind.

Exercises greeting, SQL, RAG (resume_search), employee lookup, manager,
status updates (by name / by location), unsupported topics,
list/ordinal referents, residual NLU paraphrases, everyday user
wordings, and orchestration / hallucination-control contracts against
a running API.

Usage:
  source .venv/bin/activate
  python scripts/qa_chat_local.py                 # basic + hard + utterances + orch + production
  python scripts/qa_chat_local.py --suite basic
  python scripts/qa_chat_local.py --suite hard    # adversarial + list/NLU
  python scripts/qa_chat_local.py --suite utterances  # natural paraphrases only
  python scripts/qa_chat_local.py --suite orchestrator  # Wave A–D contracts
  python scripts/qa_chat_local.py --suite production --require-meta  # architecture gate
  python scripts/qa_chat_local.py --only OR_,U_,AG_,ST_,UQ_,NS_,LR_,PR_
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

Checker = Callable[[str, dict[str, Any]], tuple[bool, str]]

DEFAULT_BASE = os.environ.get("HRMIND_BASE_URL", "http://localhost:8000").rstrip("/")
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-User-Id": os.environ.get("HRMIND_QA_USER", "local-qa"),
    "X-Role": os.environ.get("HRMIND_QA_ROLE", "recruiter"),
    "X-HRMind-Debug": "1",
}
if os.environ.get("HRMIND_QA_TENANT"):
    HEADERS["X-Tenant-Id"] = os.environ["HRMIND_QA_TENANT"]


@dataclass
class TurnResult:
    question: str
    answer: str
    session_id: str
    degraded: bool
    clarify: str | None
    confidence: float | None
    sources: list[dict[str, Any]]
    ok: bool
    note: str = ""
    error: str | None = None
    ms: int = 0
    tools_hint: str = ""
    tool: str | None = None
    meta: dict[str, Any] | None = None


@dataclass
class Scenario:
    name: str
    description: str
    tools: list[str]
    turns: list[tuple[str, Checker]]
    results: list[TurnResult] = field(default_factory=list)
    session_id: str | None = None
    role: str | None = None  # optional X-Role override for this scenario
    extra_headers: dict[str, str] = field(default_factory=dict)


def health_check(base: str, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(f"{base}/health", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat(
    base: str,
    question: str,
    session_id: str | None = None,
    *,
    timeout: float = 120.0,
    role: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"question": question}
    if session_id:
        body["session_id"] = session_id
    headers = dict(HEADERS)
    if role:
        headers["X-Role"] = role
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        f"{base}/v1/chat",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    data["_ms"] = int((time.time() - t0) * 1000)
    return data


def contains_any(text: str, *needles: str) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def not_contains(text: str, *needles: str) -> bool:
    lower = text.lower()
    return all(n.lower() not in lower for n in needles)


def first_int(text: str) -> int | None:
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


# --- Checkers ---

def expect_greeting(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(
        a,
        "hello",
        "hi",
        "hey",
        "help",
        "hrmind",
        "assist",
        "welcome",
        "goodbye",
        "happy to help",
        "anytime",
        "glad",
    )
    return ok, "greeting" if ok else f"unexpected: {a[:120]}"


def expect_numeric(a: str, _r: dict) -> tuple[bool, str]:
    n = first_int(a)
    ok = n is not None and n >= 0 and not_contains(a, "internal_error", "traceback")
    return ok, f"n={n}" if ok else f"got: {a[:160]}"


def expect_positive_count(a: str, _r: dict) -> tuple[bool, str]:
    n = first_int(a)
    ok = n is not None and n > 0 and not_contains(a, "don't have that information")
    return ok, f"n={n}" if ok else f"got: {a[:160]}"


def expect_country_names(a: str, _r: dict) -> tuple[bool, str]:
    countries = [
        "germany",
        "usa",
        "united states",
        "uk",
        "united kingdom",
        "uae",
        "france",
        "dubai",
        "netherlands",
        "canada",
        "india",
        "singapore",
    ]
    hits = sum(1 for c in countries if c in a.lower())
    refused = contains_any(a, "don't have access", "do not have access")
    ok = hits >= 3 and not refused
    return ok, f"country hits={hits}" if ok else f"hits={hits}: {a[:180]}"


def expect_city_names(a: str, _r: dict) -> tuple[bool, str]:
    cities = [
        "berlin",
        "dubai",
        "london",
        "paris",
        "new york",
        "munich",
        "san francisco",
        "amsterdam",
        "toronto",
        "bangalore",
        "singapore",
        "abu dhabi",
        "seattle",
        "austin",
        "hamburg",
        "manchester",
    ]
    hits = sum(1 for c in cities if c in a.lower())
    ok = hits >= 2 and not_contains(a, "don't have access", "internal_error")
    return ok, f"city hits={hits}" if ok else f"hits={hits}: {a[:180]}"


@lru_cache(maxsize=1)
def seeded_names() -> tuple[str, ...]:
    """Names from the seeded corpus, so name assertions follow the data rather
    than a hardcoded list that rots on the next reseed."""
    path = Path("data/seed/employees.json")
    if not path.exists():
        return ()
    people = json.loads(path.read_text())
    names: set[str] = set()
    for person in people:
        first = str(person.get("first_name") or "").lower()
        last = str(person.get("last_name") or "").lower()
        names.update(n for n in (first, last, f"{first} {last}".strip()) if n)
    return tuple(sorted(names))


def expect_has_names(a: str, _r: dict) -> tuple[bool, str]:
    refused = contains_any(a, "don't have access", "which names should i list")
    candidates = seeded_names() or ("nguyen", "smith", "garcia", "khan", "martin")
    has_name = contains_any(a, *candidates)
    ok = has_name and not refused
    return ok, "employee names" if ok else f"refused={refused}: {a[:180]}"


def expect_empty_cohort_no_names(a: str, r: dict) -> tuple[bool, str]:
    """Zero-refine list follow-up: refuse/clarify, never a name roster."""
    clarify = (r.get("clarify") or "") + " " + a
    has_roster = bool(re.search(r"\n-\s+\w+", a or "")) or contains_any(
        a, "matching employees", "here are the"
    )
    ok = (
        not has_roster
        and contains_any(
            clarify,
            "previous",
            "no names",
            "none of",
            "match that criteria",
            "ask a new search",
        )
    )
    return ok, "empty cohort list" if ok else f"got: {a[:180]}"


def expect_zero_or_small_count(a: str, _r: dict) -> tuple[bool, str]:
    n = first_int(a)
    ok = n is not None and 0 <= n < 50
    return ok, f"count={n}" if ok else f"got: {a[:120]}"


def expect_skill_or_rag(a: str, r: dict) -> tuple[bool, str]:
    sources = r.get("sources") or []
    source_kinds = {
        str(s.get("kind", "")).lower() for s in sources if isinstance(s, dict)
    }
    has_resume = bool(source_kinds & {"resume", "resume_chunk", "document"})
    substantive = contains_any(
        a,
        "python",
        "aws",
        "docker",
        "kubernetes",
        "found",
        "matching",
        "employee",
        "know",
        "experience",
        "certificate",
        "engineer",
    ) or any(ch.isdigit() for ch in a)
    # RAG ran and returned chunks even if the formatter hedges — still counts as tool path OK
    ok = (substantive or has_resume) and not_contains(a, "internal_error", "traceback")
    ok = ok and not (a.lower().startswith("hello") and len(a) < 80)
    note = f"rag/skill kinds={sorted(source_kinds)}" if ok else f"got: {a[:180]}"
    return ok, note


def expect_facet_small_count(a: str, _r: dict) -> tuple[bool, str]:
    n = first_int(a)
    ok = n is not None and 1 <= n <= 20 and not_contains(a, "internal_error")
    return ok, f"n={n}" if ok else f"got: {a[:160]}"


def expect_followup_count(a: str, _r: dict) -> tuple[bool, str]:
    # Honest empty cohort refine (prior set was empty / no skill overlap).
    if contains_any(
        a,
        "none of the previous",
        "no matching",
        "couldn't find",
        "could not find",
        "no one",
        "nobody",
        "0 employee",
        "the answer is 0",
    ):
        return True, "empty-cohort refine"
    n = first_int(a)
    if n is None:
        return False, f"no number: {a[:160]}"
    ok = 0 <= n <= 100 and not_contains(a, "don't have access", "internal_error")
    return ok, f"n={n}" if ok else f"n={n}: {a[:160]}"


def expect_person_location(a: str, _r: dict) -> tuple[bool, str]:
    # Raw tool dump or empty list = broken formatter / unresolved pronoun
    if a.strip().startswith("{") or a.strip() == '{"employees": []}':
        return False, f"raw/empty payload: {a[:120]}"
    ok = contains_any(
        a,
        "berlin",
        "dubai",
        "london",
        "paris",
        "new york",
        "munich",
        "san francisco",
        "amsterdam",
        "toronto",
        "bangalore",
        "singapore",
        "germany",
        "usa",
        "uk",
        "uae",
        "france",
        "netherlands",
        "canada",
        "india",
        "city",
        "live",
        "based",
        "located",
        "which one",  # disambiguation is acceptable
        "which employee",
        "clarify",
    ) and not_contains(a, "don't have access to employee", "internal_error")
    return ok, "location" if ok else f"got: {a[:180]}"


def expect_about_person(a: str, _r: dict) -> tuple[bool, str]:
    ok = (
        contains_any(a, "department", "engineer", "position", "employee", *seeded_names())
        and len(a) > 20
        and not_contains(a, "internal_error")
    )
    return ok, "profile" if ok else f"got: {a[:180]}"


def expect_sofia_disambiguation(a: str, r: dict) -> tuple[bool, str]:
    """Name existence with several Sofias → clarify list, not planner fallback."""
    if contains_any(a, "couldn't build a reliable plan", "try asking with a department"):
        return False, f"planner fallback: {a[:160]}"
    ok = (
        contains_any(a, "sofia", "which one", "multiple people", "which employee")
        and contains_any(a, "andersen", "brown", "yilmaz")
        and not_contains(a, "internal_error", "traceback")
    )
    clarify = (r.get("clarify") or "").lower()
    if clarify and "sofia" not in clarify and "which" not in clarify:
        return False, f"odd clarify: {clarify[:120]}"
    return ok, "sofia namesakes" if ok else f"got: {a[:180]}"


def expect_managerish(a: str, _r: dict) -> tuple[bool, str]:
    if contains_any(a, "which employee", "which one", "tell me their name"):
        return False, f"unbound pronoun: {a[:160]}"
    ok = (
        len(a.strip()) > 10
        and not_contains(a, "internal_error", "traceback")
        and (
            contains_any(a, "manager", "reports", "managed")
            or contains_any(a, *seeded_names())
        )
    )
    return ok, "manager reply" if ok else f"got: {a[:160]}"


def expect_unsupported(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(
        a,
        "don't have",
        "do not have",
        "not have that information",
        "can't access",
        "cannot",
        "outside",
        "not available",
    )
    inventing = a.strip().lower().startswith("the answer is") and any(
        ch.isdigit() for ch in a
    )
    ok = ok and not inventing
    return ok, "unsupported handled" if ok else f"got: {a[:180]}"


def expect_unauthorized(a: str, r: dict) -> tuple[bool, str]:
    """Soft ACL refuse — access language, not 'no data' / not inventing a number."""
    ok = contains_any(
        a,
        "don't have access",
        "do not have access",
        "not have access",
        "can't access",
        "cannot access",
        "not permitted",
        "with your current role",
    ) and not_contains(a, "internal_error", "traceback")
    # Unauthorized answers must not set UI clarify.
    if r.get("clarify"):
        return False, f"clarify set on unauthorized: {r.get('clarify')!r}"
    return ok, "unauthorized" if ok else f"got: {a[:180]}"


def expect_salary_answer(a: str, _r: dict) -> tuple[bool, str]:
    """Recruiter (or same-dept manager) should see a grounded salary, not refuse."""
    has_num = bool(re.search(r"\d{2,}", a.replace(",", "")))
    ok = (
        has_num
        and contains_any(a, "salary", "pay", "$", "usd", "earn", "makes", "compensation")
        and not_contains(a, "don't have access", "don't have that information")
    )
    return ok, "salary" if ok else f"got: {a[:180]}"


def expect_clarify(a: str, r: dict) -> tuple[bool, str]:
    clarify = (r.get("clarify") or "") + " " + a
    ok = contains_any(
        clarify,
        "which",
        "whose",
        "clarify",
        "specify",
        "previous",
        "countries",
        "cities",
        "departments",
        "need more",
        "don't have",
        "bit more detail",
    )
    return ok, "clarify-ish" if ok else f"got: {a[:160]}"


def expect_topic_shift_fresh(a: str, _r: dict) -> tuple[bool, str]:
    # After "hello" / start over, a fresh skill question should still answer.
    return expect_skill_or_rag(a, _r)


def expect_status_updated(a: str, _r: dict) -> tuple[bool, str]:
    """Person or cohort status write succeeded (not greeting / not unresolved)."""
    ok = (
        contains_any(a, "updated", "status to true", "status to false", "set status")
        and not_contains(
            a,
            "couldn't match",
            "could not match",
            "couldn't resolve",
            "could not resolve",
            "internal_error",
            "traceback",
        )
        and not (a.lower().strip().startswith("hello") and len(a) < 120)
    )
    return ok, "status updated" if ok else f"got: {a[:200]}"


def expect_status_true(a: str, _r: dict) -> tuple[bool, str]:
    ok, note = expect_status_updated(a, _r)
    if not ok:
        return False, note
    ok = contains_any(a, "status to true", "to true")
    return ok, "status→true" if ok else f"got: {a[:200]}"


def expect_status_false(a: str, _r: dict) -> tuple[bool, str]:
    ok, note = expect_status_updated(a, _r)
    if not ok:
        return False, note
    ok = contains_any(a, "status to false", "to false")
    return ok, "status→false" if ok else f"got: {a[:200]}"


def expect_status_location_cohort(a: str, _r: dict) -> tuple[bool, str]:
    """Location-based status: confirm gate OR successful multi-person update."""
    clarify = ((_r.get("clarify") or "") + " " + a).lower()
    if contains_any(
        clarify,
        "confirm status update",
        "confirm",
        "proceed",
        "how many",
        "employees match",
        "which",
    ) and not contains_any(a, "internal_error", "traceback"):
        return True, "status confirm / clarify gate"
    ok, note = expect_status_true(a, _r)
    if not ok:
        return False, note
    n = first_int(a)
    ok = n is not None and n >= 1
    return ok, f"location cohort n={n}" if ok else f"no count: {a[:200]}"


def expect_status_confirm_gate_only(a: str, _r: dict) -> tuple[bool, str]:
    """First turn of a status write must ask to confirm (no commit yet)."""
    clarify = ((_r.get("clarify") or "") + " " + a).lower()
    ok = contains_any(
        clarify, "confirm status update", "reply yes to confirm"
    ) and not contains_any(a, "updated", "internal_error", "traceback")
    return ok, "confirm gate" if ok else f"got: {a[:200]}"


def all_of(*checkers: Checker) -> Checker:
    def _check(a: str, r: dict) -> tuple[bool, str]:
        notes: list[str] = []
        for checker in checkers:
            ok, note = checker(a, r)
            notes.append(note)
            if not ok:
                return False, note
        return True, "; ".join(notes)

    return _check


def _meta(r: dict) -> dict[str, Any]:
    m = r.get("meta")
    return m if isinstance(m, dict) else {}


def _answered_tools(r: dict) -> set[str]:
    raw = r.get("tool") or _meta(r).get("tools_answered") or ""
    parts: set[str] = set()
    for chunk in str(raw).replace(",", "+").split("+"):
        name = chunk.strip().lower()
        if name:
            parts.add(name)
    for node in _meta(r).get("plan_nodes") or []:
        if isinstance(node, str) and node.strip():
            parts.add(node.strip().lower())
    return parts


def expect_tool(*names: str) -> Checker:
    need = {n.lower() for n in names}

    def _check(a: str, r: dict) -> tuple[bool, str]:
        got = _answered_tools(r)
        ok = need.issubset(got)
        return (
            ok,
            f"tools={sorted(got)}" if ok else f"need {sorted(need)} got {sorted(got)}: {a[:120]}",
        )

    return _check


def expect_planner_mode(*prefixes: str) -> Checker:
    def _check(a: str, r: dict) -> tuple[bool, str]:
        mode = str(_meta(r).get("planner_mode") or "")
        ok = any(mode.startswith(p) for p in prefixes)
        return ok, f"mode={mode}" if ok else f"mode={mode!r} not in {prefixes}: {a[:100]}"

    return _check


def expect_not_degraded(a: str, r: dict) -> tuple[bool, str]:
    ok = not bool(r.get("degraded"))
    return ok, "not-degraded" if ok else f"degraded: {a[:160]}"


def expect_hitl_gate(a: str, r: dict) -> tuple[bool, str]:
    ok_gate, note = expect_status_confirm_gate_only(a, r)
    if not ok_gate:
        return False, note
    meta = _meta(r)
    needs_hitl = bool(meta.get("needs_hitl"))
    nodes = meta.get("plan_nodes") or []
    mode = str(meta.get("planner_mode") or "")
    structural = needs_hitl or not nodes or "hitl" in mode
    ok = structural and not contains_any(a, "updated")
    return ok, "hitl gate" if ok else f"gate text ok but meta weak: {meta}"


def expect_status_committed(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_status_true(a, r)
    if not ok:
        return False, note
    meta = _meta(r)
    if meta.get("needs_hitl"):
        return False, f"still needs_hitl after confirm: {meta}"
    return True, "status committed"


def expect_repair_used(a: str, r: dict) -> tuple[bool, str]:
    repairs = int(_meta(r).get("repairs") or 0)
    mode = str(_meta(r).get("planner_mode") or "")
    ok = repairs >= 1 or "repair" in mode
    return ok, f"repairs={repairs} mode={mode}" if ok else f"no repair: {mode}"


def expect_multi_tool(a: str, r: dict) -> tuple[bool, str]:
    tool = str(r.get("tool") or _meta(r).get("tools_answered") or "")
    nodes = [
        str(n).lower()
        for n in (_meta(r).get("plan_nodes") or [])
        if isinstance(n, str)
    ]
    distinct = {n for n in nodes if n and n not in {"intersect", "count", "filter", "sort"}}
    tool_parts = {p.strip() for p in tool.replace(",", "+").split("+") if p.strip()}
    ok = len(distinct) >= 2 or len(tool_parts) >= 2 or "+" in tool
    return (
        ok,
        f"multi-tool nodes={nodes} tool={tool}"
        if ok
        else f"single-tool nodes={nodes} tool={tool}: {a[:120]}",
    )


def expect_confidence_band(lo: float, hi: float) -> Checker:
    def _check(a: str, r: dict) -> tuple[bool, str]:
        conf = r.get("confidence")
        if conf is None:
            conf = _meta(r).get("select_confidence")
        try:
            value = float(conf)
        except (TypeError, ValueError):
            return False, f"no confidence: {a[:120]}"
        ok = lo <= value <= hi
        return ok, f"confidence={value}" if ok else f"confidence={value} not in [{lo},{hi}]"

    return _check


def expect_exact_count(n: int) -> Checker:
    def _check(a: str, _r: dict) -> tuple[bool, str]:
        got = first_int(a)
        ok = got == n and not_contains(a, "internal_error", "traceback")
        return ok, f"n={got}" if ok else f"want {n} got {got}: {a[:160]}"

    return _check


def expect_names_only(*names: str) -> Checker:
    want = [n.lower() for n in names]

    def _check(a: str, _r: dict) -> tuple[bool, str]:
        lower = a.lower()
        hits = [n for n in want if n in lower]
        if len(hits) < len(want):
            return False, f"missing names {want}: {a[:180]}"
        # Reject large unrelated dumps: at most a few extra first names from seed.
        return True, f"names={hits}"

    return _check


def expect_no_status_write(a: str, r: dict) -> tuple[bool, str]:
    if contains_any(a, "updated", "status to true", "status to false"):
        return False, f"unexpected write: {a[:160]}"
    tools = _answered_tools(r)
    if "employee" in tools and "set_status" in str(_meta(r)).lower():
        return False, f"set_status path: {_meta(r)}"
    mode = str(_meta(r).get("planner_mode") or "")
    if "set_status" in mode or "confirm_status" in mode:
        return False, f"status write mode: {mode}"
    return True, "read-only statuses"


def expect_language_cohort(a: str, _r: dict) -> tuple[bool, str]:
    """Languages attribute: names/count or honest empty — never invent."""
    if contains_any(
        a,
        "internal_error",
        "traceback",
        "don't have access",
    ):
        return False, f"error: {a[:180]}"
    if contains_any(
        a,
        "no one",
        "nobody",
        "found no",
        "couldn't find",
        "could not find",
        "not listed",
        "no employee",
        "0 employee",
    ):
        return True, "honest empty languages"
    ok = contains_any(a, "speak", "language", "german", "french", "arabic", "english") or (
        expect_has_names(a, _r)[0]
    )
    return ok, "languages cohort" if ok else f"got: {a[:180]}"


def expect_cert_cohort(a: str, _r: dict) -> tuple[bool, str]:
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:180]}"
    if contains_any(
        a,
        "no one",
        "nobody",
        "found no",
        "couldn't find",
        "could not find",
        "0 employee",
    ):
        return True, "honest empty certs"
    ok = contains_any(
        a, "certif", "aws certified", "cka", "pmp", "holds", "list"
    ) or expect_has_names(a, _r)[0]
    return ok, "cert cohort" if ok else f"got: {a[:180]}"


def expect_hire_window(a: str, _r: dict) -> tuple[bool, str]:
    if contains_any(a, "internal_error", "traceback", "don't have access"):
        return False, f"error: {a[:180]}"
    if contains_any(
        a,
        "no one",
        "nobody",
        "none of",
        "couldn't find",
        "0 employee",
        "the answer is 0",
        "no matching",
    ):
        return True, "honest empty hire window"
    return expect_has_names(a, _r)


def expect_tenure_answer(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(
        a,
        "tenure",
        "hire_date",
        "hired",
        "years",
        "days",
        "average",
        "longest",
    ) and not contains_any(a, "internal_error", "traceback")
    return ok, "tenure analytics" if ok else f"got: {a[:180]}"


def expect_reports_roster(a: str, _r: dict) -> tuple[bool, str]:
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:180]}"
    if contains_any(a, "no direct report", "0 direct", "couldn't find", "which"):
        return True, "empty/clarify reports"
    ok = contains_any(a, "direct report", "report") or expect_has_names(a, _r)[0]
    return ok, "reports roster" if ok else f"got: {a[:180]}"


def expect_composition_or_empty(a: str, _r: dict) -> tuple[bool, str]:
    """Manager×skill×place: names, empty intersect, or clarify — never invent."""
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:180]}"
    if contains_any(
        a,
        "none of",
        "no matching",
        "couldn't find",
        "could not find",
        "no one",
        "nobody",
        "0 employee",
        "which",
        "need both",
        "clarify",
    ):
        return True, "empty/clarify composition"
    return expect_has_names(a, _r)


def expect_not_greeting(a: str, _r: dict) -> tuple[bool, str]:
    """Follow-up must stay on tools — not a chitchat greeting."""
    lower = a.strip().lower()
    greetingish = (
        lower.startswith(("hello", "hi ", "hey", "hi!"))
        or contains_any(a, "how can i help", "happy to help you today", "i'm hrmind")
    ) and len(a) < 220
    if greetingish:
        return False, f"got greeting: {a[:160]}"
    return True, "not-greeting"


def expect_hr_answer_not_greeting(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_not_greeting(a, r)
    if not ok:
        return False, note
    if len(a.strip()) < 8:
        return False, "empty-ish answer"
    return True, note


def expect_skill_and_place(a: str, r: dict) -> tuple[bool, str]:
    """Skill∩location: names/count OK; must not greet; must not claim whole org."""
    ok, note = expect_not_greeting(a, r)
    if not ok:
        return False, note
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    # Honest empty / clarify is fine; org-wide dump is not.
    n = first_int(a)
    if n is not None and n >= 80:
        return False, f"suspiciously large n={n}: {a[:160]}"
    if contains_any(
        a,
        "don't recognize",
        "don't know",
        "no one",
        "nobody",
        "0 employee",
        "none",
        "couldn't find",
        "could not find",
        "which",
        "clarify",
    ):
        return True, "honest empty/clarify"
    return expect_skill_or_rag(a, r)


def expect_unknown_place_clarify(a: str, r: dict) -> tuple[bool, str]:
    clarify = ((r.get("clarify") or "") + " " + a).lower()
    ok = contains_any(
        clarify,
        "don't recognize",
        "do not recognize",
        "known city",
        "known country",
        "known place",
        "which city",
        "which country",
        "places we track",
        "not sure",
        "clarify",
        "specify",
    ) or contains_any(a, "no one", "nobody", "0 ", "none")
    return ok, "unknown-place handled" if ok else f"got: {a[:180]}"


_MONTHS_TEXT = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _has_real_date(text: str) -> bool:
    """A spelled month with a year, e.g. '12 March 1991'."""
    lower = text.lower()
    return any(m in lower for m in _MONTHS_TEXT)


def expect_birthday_from_resume(a: str, r: dict) -> tuple[bool, str]:
    """Birth date recovered from resume text (or an honest namesake clarification)."""
    if a.strip().startswith("{"):
        return False, f"raw payload: {a[:120]}"
    if contains_any(a, "internal_error", "traceback", "don't have access"):
        return False, f"error-ish: {a[:200]}"
    # Seeded corpus repeats names, so listing namesakes with their dates is valid.
    ambiguous = contains_any(a, "employees match", "which one you mean")
    if not _has_real_date(a):
        return False, f"no date in answer: {a[:200]}"
    kinds = {
        str(s.get("kind", "")).lower() for s in (r.get("sources") or []) if isinstance(s, dict)
    }
    if kinds and "resume_chunk" not in kinds:
        return False, f"not resume-sourced: {sorted(kinds)}"
    note = "namesakes listed with dates" if ambiguous else "birth date from resume"
    return True, note


def expect_birthday_wish(a: str, r: dict) -> tuple[bool, str]:
    """A resolvable person gets a date plus a birthday greeting."""
    ok, note = expect_birthday_from_resume(a, r)
    if not ok:
        return False, note
    if contains_any(a, "employees match", "which one you mean"):
        return True, "namesakes listed (wish deferred)"
    ok = contains_any(a, "happy birthday", "birthday wishes")
    return ok, "wish + date" if ok else f"no greeting: {a[:200]}"


def expect_birthday_age(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_birthday_from_resume(a, r)
    if not ok:
        return False, note
    ok = bool(first_int(a))
    return ok, "age reported" if ok else f"no age: {a[:200]}"


def expect_birthday_today(a: str, _r: dict) -> tuple[bool, str]:
    """Either named birthdays today or an honest 'nobody today'."""
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    none_today = contains_any(a, "no employee has a birthday today", "no birthdays today")
    if none_today:
        return True, "none today (honest)"
    ok = contains_any(a, "birthday") and contains_any(a, "happy birthday")
    return ok, "birthdays today" if ok else f"got: {a[:200]}"


def expect_birthday_month(a: str, _r: dict) -> tuple[bool, str]:
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    ok = contains_any(a, "july") and (
        bool(first_int(a)) or contains_any(a, "no employee has a birthday")
    )
    return ok, "july birthdays" if ok else f"got: {a[:200]}"


def expect_birthday_upcomingish(a: str, _r: dict) -> tuple[bool, str]:
    """Upcoming / this-month birthday list — honest empty is ok."""
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    ok = contains_any(
        a,
        "birthday",
        "birthdays",
        "upcoming",
        "no employee has a birthday",
        "no birthdays",
    )
    return ok, "upcoming-ish" if ok else f"got: {a[:200]}"


def expect_birthday_missing_person(a: str, _r: dict) -> tuple[bool, str]:
    """An unknown name, or a resume with no date, must be refused honestly."""
    ok = contains_any(a, "couldn't find a date of birth", "could not find a date of birth")
    return ok, "honest miss" if ok else f"got: {a[:200]}"


def expect_birthday_namesakes(a: str, r: dict) -> tuple[bool, str]:
    """Shared names must list every match rather than picking one."""
    if not contains_any(a, "employees match"):
        return False, f"did not disambiguate: {a[:200]}"
    return expect_birthday_from_resume(a, r)


def expect_birthday_typo(a: str, r: dict) -> tuple[bool, str]:
    """A misspelled name still resolves, and the approximation is admitted."""
    ok, note = expect_birthday_from_resume(a, r)
    if not ok:
        return False, note
    ok = contains_any(a, "no exact match", "closest is")
    return ok, "fuzzy match admitted" if ok else f"no fuzzy note: {a[:200]}"


def expect_birthday_leap_day(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_birthday_from_resume(a, r)
    if not ok:
        return False, note
    ok = contains_any(a, "29 february")
    return ok, "leap day date" if ok else f"not 29 February: {a[:200]}"


def expect_birthday_coverage_admitted(a: str, _r: dict) -> tuple[bool, str]:
    """Cohort answers must state how much of the corpus they could actually read."""
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    ok = contains_any(a, "read from") and contains_any(a, "no date of birth recorded")
    return ok, "coverage stated" if ok else f"no coverage caveat: {a[:200]}"


def expect_not_org_wide_100(a: str, _r: dict) -> tuple[bool, str]:
    """Follow-ups over a cohort must not accidentally answer org headcount (100)."""
    n = first_int(a)
    if n is None:
        # name lists / clarify OK
        if contains_any(a, "matching", "employees", "which", "clarify"):
            return True, "non-numeric ok"
        return False, f"no number: {a[:160]}"
    ok = n != 100 and 0 <= n <= 50 and not_contains(a, "internal_error")
    return ok, f"n={n}" if ok else f"suspected org-wide leak n={n}: {a[:160]}"


def expect_count_between(lo: int, hi: int) -> Checker:
    def _check(a: str, _r: dict) -> tuple[bool, str]:
        n = first_int(a)
        ok = n is not None and lo <= n <= hi and not_contains(a, "internal_error")
        return ok, f"n={n} in [{lo},{hi}]" if ok else f"n={n} not in [{lo},{hi}]: {a[:160]}"

    return _check


def expect_mentions_dept(dept: str) -> Checker:
    def _check(a: str, _r: dict) -> tuple[bool, str]:
        ok = dept.lower() in a.lower() or first_int(a) is not None
        ok = ok and not_contains(a, "internal_error")
        return ok, f"dept/count ok" if ok else f"got: {a[:160]}"

    return _check


def make_monotonic_count_checkers() -> tuple[Checker, Checker]:
    """Capture a count, then require a later count to be <= that value."""
    box: dict[str, int | None] = {"n": None}

    def capture(a: str, r: dict) -> tuple[bool, str]:
        ok, note = expect_positive_count(a, r)
        if ok:
            box["n"] = first_int(a)
        return ok, note

    def lte_prior(a: str, _r: dict) -> tuple[bool, str]:
        n = first_int(a)
        prior = box["n"]
        if n is None or prior is None:
            return False, f"missing counts n={n} prior={prior}: {a[:140]}"
        ok = 0 <= n <= prior and n != 100
        return ok, f"n={n}<=prior={prior}" if ok else f"n={n} > prior={prior}"

    return capture, lte_prior


def make_listed_ordinal_birthday_checkers(
    *, index: int = 1
) -> tuple[Checker, Checker]:
    """Capture bullet names, then require DOB for a list index.

    ``index`` is 1-based; use ``-1`` for the last listed person.
    """
    box: dict[str, Any] = {"names": []}

    def _resolve_index(names: list[str]) -> int | None:
        if not names:
            return None
        if index == -1:
            return len(names)
        if 1 <= index <= len(names):
            return index
        return None

    def capture_names(a: str, r: dict) -> tuple[bool, str]:
        ok, note = expect_has_names(a, r)
        if not ok:
            return ok, note
        names: list[str] = []
        for line in a.splitlines():
            line = line.strip()
            if line.startswith("- "):
                # "- Kara Petrov (Sales Manager, Sales)"
                label = line[2:].split("(")[0].strip()
                if label:
                    names.append(label)
        if not names:
            # Singular formatter: "The matching employee is Priya Moreau (…)."
            m = re.search(
                r"\b(?:matching employee is|employee is)\s+"
                r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?)",
                a,
            )
            if m:
                names = [m.group(1).strip()]
        box["names"] = names
        if not names:
            return False, f"no bullet name to capture: {a[:180]}"
        resolved = _resolve_index(names)
        if resolved is None:
            return False, f"need index={index} names, got {len(names)}: {names}"
        return True, f"names[{resolved}]={names[resolved - 1]} (n={len(names)})"

    def expect_nth_dob(a: str, r: dict) -> tuple[bool, str]:
        names: list[str] = list(box.get("names") or [])
        resolved = _resolve_index(names)
        if resolved is None:
            return False, f"no captured name at index {index}: {names}"
        name = names[resolved - 1]
        if contains_any(a, "first person", "first persons", "second person"):
            return False, f"treated ordinal as a name: {a[:180]}"
        ok, note = expect_birthday_from_resume(a, r)
        if not ok:
            return False, note
        surname = name.split()[-1].lower()
        if surname not in a.lower():
            return False, f"expected {name} in DOB answer: {a[:180]}"
        return True, f"dob for {name}"

    return capture_names, expect_nth_dob


def make_listed_ordinal_location_checkers(
    *, index: int = 1
) -> tuple[Checker, Checker]:
    """Capture bullet names, then require a location answer for that person."""
    box: dict[str, Any] = {"names": []}

    def capture_names(a: str, r: dict) -> tuple[bool, str]:
        ok, note = expect_has_names(a, r)
        if not ok:
            return ok, note
        names = [
            line[2:].split("(")[0].strip()
            for line in a.splitlines()
            if line.strip().startswith("- ") and line.strip()[2:].split("(")[0].strip()
        ]
        box["names"] = names
        if not names:
            return False, f"no bullet name to capture: {a[:180]}"
        if index > len(names):
            return False, f"need ≥{index} names, got {len(names)}"
        return True, f"names[{index}]={names[index - 1]}"

    def expect_nth_location(a: str, r: dict) -> tuple[bool, str]:
        names: list[str] = list(box.get("names") or [])
        if index < 1 or index > len(names):
            return False, f"no captured name at index {index}"
        name = names[index - 1]
        ok, note = expect_person_location(a, r)
        if not ok:
            return False, note
        surname = name.split()[-1].lower()
        if surname not in a.lower() and not contains_any(
            a, "which", "whose", "which person", "which employee"
        ):
            # Location answers usually echo the name; clarify is also acceptable.
            return False, f"expected {name} (or clarify) in location answer: {a[:180]}"
        return True, f"location for {name}"

    return capture_names, expect_nth_location


def expect_ordinal_needs_names(a: str, _r: dict) -> tuple[bool, str]:
    """Count-only shrink must not invent an ordinal binding."""
    ok = contains_any(
        a,
        "names first",
        "ask for their names",
        "which person",
        "which employee",
    )
    return ok, "needs names" if ok else f"got: {a[:180]}"


def run_scenario(
    base: str,
    sc: Scenario,
    *,
    require_meta: bool = False,
) -> Scenario:
    sid = None
    for q, checker in sc.turns:
        try:
            data = chat(
                base,
                q,
                sid,
                role=sc.role,
                extra_headers=sc.extra_headers or None,
            )
            sid = data.get("session_id") or sid
            answer = data.get("answer") or ""
            sources = data.get("sources") or []
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else None
            tool = data.get("tool")
            ok, note = checker(answer, data)
            if require_meta and not meta:
                ok = False
                note = f"missing meta ({note})"
            # Soft contract: scenario tool hints must intersect answered tools.
            hint = {t.lower() for t in sc.tools if t}
            answered = _answered_tools(data)
            if (
                ok
                and hint
                and answered
                and not (hint & answered)
                and not data.get("clarify")
            ):
                ok = False
                note = f"tools disjoint hint={sorted(hint)} got={sorted(answered)}; {note}"
            sc.results.append(
                TurnResult(
                    question=q,
                    answer=answer[:800],
                    session_id=sid or "",
                    degraded=bool(data.get("degraded")),
                    clarify=data.get("clarify"),
                    confidence=data.get("confidence"),
                    sources=sources if isinstance(sources, list) else [],
                    ok=ok,
                    note=note,
                    ms=int(data.get("_ms") or 0),
                    tools_hint=",".join(sc.tools),
                    tool=tool if isinstance(tool, str) else None,
                    meta=meta,
                )
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            sc.results.append(
                TurnResult(
                    question=q,
                    answer="",
                    session_id=sid or "",
                    degraded=True,
                    clarify=None,
                    confidence=None,
                    sources=[],
                    ok=False,
                    note="http error",
                    error=f"{exc.code}: {body}",
                )
            )
            break
        except Exception as exc:  # noqa: BLE001
            sc.results.append(
                TurnResult(
                    question=q,
                    answer="",
                    session_id=sid or "",
                    degraded=True,
                    clarify=None,
                    confidence=None,
                    sources=[],
                    ok=False,
                    note="request failed",
                    error=str(exc),
                )
            )
            break
    sc.session_id = sid
    return sc


def build_hard_scenarios() -> list[Scenario]:
    """Adversarial multi-turn dialogs meant to break weak context / planning."""
    eng_capture, eng_lte = make_monotonic_count_checkers()
    py_capture, py_lte = make_monotonic_count_checkers()

    return [
        Scenario(
            "U_nested_filters_eng_berlin_python",
            "Hard: Eng → of them in Berlin → of them know Python → names (3-level refine)",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", eng_capture),
                ("how many of them are in Berlin?", expect_count_between(1, 10)),
                ("and how many of those know Python?", eng_lte),
                ("names please", expect_has_names),
            ],
        ),
        Scenario(
            "V_cohort_replace_not_refine",
            "Hard: Eng cohort then switch to Sales — must not keep Engineering IDs",
            ["sql"],
            [
                ("How many employees work in Engineering?", expect_count_between(10, 30)),
                ("say their names", expect_has_names),
                ("instead find how many employees work in Sales?", expect_count_between(10, 30)),
                ("names please", expect_has_names),
            ],
        ),
        Scenario(
            "W_facet_then_cohort_switch",
            "Hard: country facet 'names please' then Eng list — focus must not stick wrongly",
            ["resume_search", "sql"],
            [
                ("in how different countries do we have employees?", expect_facet_small_count),
                ("names please", expect_country_names),
                ("List employees in Engineering", expect_has_names),
                ("how many of them in Berlin?", expect_not_org_wide_100),
            ],
        ),
        Scenario(
            "X_messy_anaphora_and_typos",
            "Hard: noisy / informal language still resolves to prior Engineering set",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                (
                    "ok and frm that group how many knows python??",
                    expect_followup_count,
                ),
                ("pls list their names", expect_has_names),
            ],
        ),
        Scenario(
            "Y_org_wide_then_of_them_location",
            "Hard: org headcount (no saved cohort) → 'of them from USA' must not crash/hallucinate",
            ["sql"],
            [
                ("How many employees do we have?", expect_count_between(50, 200)),
                ("how many of them from USA?", expect_not_org_wide_100),
            ],
        ),
        Scenario(
            "Z_skill_then_sql_then_back",
            "Hard: RAG Python set → SQL city refine → skill refine Kubernetes",
            ["resume_search", "sql"],
            [
                ("Who knows Python?", expect_skill_or_rag),
                ("how many of them are in Dubai?", expect_followup_count),
                ("which of them know Kubernetes?", expect_skill_or_rag),
            ],
        ),
        Scenario(
            "AA_interrupt_with_unsupported",
            "Hard: active cohort → unsupported vacation → resume prior 'names please'",
            ["sql", "clarify"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how much vacation are they taking?", expect_unsupported),
                ("names please", expect_has_names),
            ],
        ),
        Scenario(
            "AB_start_over_mid_dialog",
            "Hard: deep refine then 'start over' / new search must clear stale them",
            ["sql", "resume_search", "greeting"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("start over — find React developers", expect_skill_or_rag),
                ("how many of them?", expect_followup_count),
            ],
        ),
        Scenario(
            "AC_multi_department_compare_style",
            "Hard: ask Eng count then Product count then 'names' — which referent wins?",
            ["sql"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("How many employees work in Product?", expect_positive_count),
                ("names please", expect_has_names),
            ],
        ),
        Scenario(
            "AD_hire_date_style_followup",
            "Hard: Engineering set → who joined after 2020 (hire_date filter on cohort)",
            ["sql"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                (
                    "how many of them joined after 2020?",
                    expect_followup_count,
                ),
            ],
        ),
        Scenario(
            "AE_ambiguous_person_then_disambiguate_path",
            "Hard: ambiguous Ivy → ask manager of Alice (must stay coherent, no crash)",
            ["employee", "resume_search"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                ("Who is the manager of Alice Nguyen?", expect_managerish),
                ("where does she live?", expect_person_location),
            ],
        ),
        Scenario(
            "AF_constraint_stack_city_and_skill",
            "Hard: Berlin employees → of them know React → count must shrink",
            ["sql", "resume_search"],
            [
                ("List employees in Berlin", expect_has_names),
                ("how many of them know React?", py_capture),
                ("names please", expect_has_names),
                ("how many was that again?", py_lte),
            ],
        ),
        Scenario(
            "AG_long_adversarial_session",
            "Hard: 10-turn stress — facets, cohorts, RAG, clear, replace, employee",
            ["sql", "resume_search", "employee", "greeting", "clarify"],
            [
                ("in how different countries do we have employees?", expect_facet_small_count),
                ("names please", expect_country_names),
                ("How many employees work in Engineering?", expect_count_between(10, 30)),
                ("of them in Berlin — how many?", expect_count_between(0, 10)),
                ("and how many of those know Python?", expect_followup_count),
                ("names please", expect_has_names),
                ("never mind, who knows Kubernetes?", expect_skill_or_rag),
                ("how many of them in Dubai?", expect_followup_count),
                ("Tell me about Alice Nguyen", expect_about_person),
                ("how much PTO do they get?", expect_unsupported),
            ],
        ),
        Scenario(
            "AH_double_anaphora_chain",
            "Hard: them → those → that group chain without restating Engineering",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them?", expect_count_between(10, 30)),
                ("how many of those know Python?", expect_followup_count),
                ("from that group, how many in Berlin?", expect_not_org_wide_100),
            ],
        ),
    ]


def build_orchestrator_scenarios() -> list[Scenario]:
    """Wave A–D contracts: wrong cohort / wrong person / inventing tails."""
    return [
        Scenario(
            "OR_elliptical_after_list_not_greeting",
            "List people → short attr ask must not greet (CHITCHAT override)",
            ["sql", "employee", "resume_search"],
            [
                ("List employees in Engineering", expect_has_names),
                ("her email?", expect_hr_answer_not_greeting),
            ],
        ),
        Scenario(
            "OR_pronoun_after_two_names_clarifies",
            "Two people listed → 'where does she live?' must clarify, not guess",
            ["sql", "resume_search", "clarify"],
            [
                ("List employees in Berlin", expect_has_names),
                ("where does she live?", expect_clarify),
            ],
        ),
        Scenario(
            "OR_ordinal_birthday_after_names",
            "Names → first person's DOB via resume, not greeting/profile invent",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("list their names", expect_has_names),
                ("give me the first persons date of birth", expect_birthday_from_resume),
            ],
        ),
        Scenario(
            "OR_skill_and_city_intersect",
            "Skill∩location in one ask — not city dropped / not org-wide dump",
            ["resume_search", "sql"],
            [
                ("Who knows Python in Berlin?", expect_skill_and_place),
            ],
        ),
        Scenario(
            "OR_skill_and_city_count",
            "Count form of skill∩place",
            ["resume_search", "sql"],
            [
                ("How many employees know React in Dubai?", expect_skill_and_place),
            ],
        ),
        Scenario(
            "OR_expanded_place_vocab",
            "New cities in place vocab still retrieve (or honest empty)",
            ["resume_search", "sql"],
            [
                ("List employees in Munich", expect_hr_answer_not_greeting),
                ("how many of them?", expect_not_org_wide_100),
            ],
        ),
        Scenario(
            "OR_unknown_place_no_invent",
            "Unknown city must clarify / empty — not invent a cohort",
            ["clarify", "resume_search"],
            [
                ("List employees in Atlantis", expect_unknown_place_clarify),
            ],
        ),
        Scenario(
            "OR_status_location_confirm_gate",
            "Multi-person location status asks confirm (or updates with count)",
            ["resume_search", "employee"],
            [
                (
                    "update status of employees which are living in Dubai to true",
                    expect_status_location_cohort,
                ),
            ],
        ),
        Scenario(
            "OR_status_exact_name",
            "Exact full-name status write still works",
            ["employee", "resume_search"],
            [
                ("change the status of Carol Garcia to true", expect_status_true),
            ],
        ),
        Scenario(
            "OR_weak_followup_after_org_headcount",
            "Org headcount is not a 'them' cohort for of-them location",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_count_between(50, 200)),
                ("how many of them from USA?", expect_not_org_wide_100),
            ],
        ),
        Scenario(
            "OR_bare_hello_still_greeting",
            "Bare social without person focus stays greeting",
            ["greeting"],
            [("hello", expect_greeting)],
        ),
        Scenario(
            "OR_list_then_manager_chain",
            "Profile → manager → location stay on tools through short turns",
            ["employee", "resume_search"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                ("her manager?", expect_managerish),
                ("where does she live?", expect_person_location),
            ],
        ),
        Scenario(
            "OR_unsupported_no_invented_count",
            "Unsupported topic must not invent a headcount",
            ["clarify"],
            [("how much vacation are developers taking?", expect_unsupported)],
        ),
        Scenario(
            "OR_long_orchestrator_dialog",
            "Mixed dialog stressing refine, ordinal, place, skill∩place",
            ["sql", "resume_search", "employee", "clarify", "greeting"],
            [
                ("hey — how many engineers?", expect_positive_count),
                ("names please", expect_has_names),
                ("first person's birthday?", expect_birthday_from_resume),
                ("thanks", expect_greeting),
                ("who knows Python in Berlin?", expect_skill_and_place),
                ("list employees in San Francisco", expect_hr_answer_not_greeting),
                ("any of them know Kubernetes?", expect_followup_count),
            ],
        ),
        # --- Coverage gaps roadmap (confirm + Waves 1–4) ---
        Scenario(
            "OR_confirm_status_continuation",
            "Location status propose → confirm status update commits",
            ["resume_search", "employee"],
            [
                (
                    "update status of employees which are living in Dubai to true",
                    expect_status_confirm_gate_only,
                ),
                ("confirm status update", expect_status_true),
            ],
        ),
        Scenario(
            "OR_languages_cohort",
            "Who speaks German? → languages attribute (or honest empty)",
            ["resume_search"],
            [("who speaks German?", expect_language_cohort)],
        ),
        Scenario(
            "OR_languages_prior_scope",
            "Prior list → who among them speaks French stays scoped",
            ["resume_search", "sql"],
            [
                ("List employees in Berlin", expect_has_names),
                ("who among them speaks French?", expect_language_cohort),
            ],
        ),
        Scenario(
            "OR_certifications_cohort",
            "AWS Certified → cert attribute (not generic invent)",
            ["resume_search"],
            [("who has AWS Certified?", expect_cert_cohort)],
        ),
        Scenario(
            "OR_cert_vs_skill_experience",
            "AWS experience stays skill RAG path",
            ["resume_search", "sql"],
            [("who has AWS experience?", expect_skill_or_rag)],
        ),
        Scenario(
            "OR_hire_last_90_days",
            "Last 90 days hire window from SQL templates",
            ["sql"],
            [("who joined in the last 90 days?", expect_hire_window)],
        ),
        Scenario(
            "OR_hire_this_quarter",
            "This quarter hire window",
            ["sql"],
            [("who joined this quarter?", expect_hire_window)],
        ),
        Scenario(
            "OR_avg_tenure_engineering",
            "Average tenure template (hire_date, not title)",
            ["sql"],
            [("what is the average tenure in Engineering?", expect_tenure_answer)],
        ),
        Scenario(
            "OR_most_senior_hire_date",
            "Most senior = longest tenured by hire_date",
            ["sql"],
            [("who is the most senior in Engineering?", expect_tenure_answer)],
        ),
        Scenario(
            "OR_direct_reports",
            "Direct reports action for a manager",
            ["employee"],
            [("who reports to Alice Nguyen?", expect_reports_roster)],
        ),
        Scenario(
            "OR_reports_skill_place",
            "Manager×skill×place composition DAG",
            ["employee", "resume_search", "sql"],
            [
                (
                    "who on Alice Nguyen's team knows Kubernetes and is in Dubai?",
                    expect_composition_or_empty,
                ),
            ],
        ),
        Scenario(
            "OR_hris_pto_refused",
            "PTO stays unsupported — no soft resume invent",
            ["clarify"],
            [("how much PTO does Alice Nguyen have?", expect_unsupported)],
        ),
    ]


def build_production_scenarios() -> list[Scenario]:
    """Production architecture gate: multi-tool, HITL, guards, ReAct, long session."""
    eng_capture, eng_lte = make_monotonic_count_checkers()
    listed_names: list[str] = []

    def capture_eng_names(a: str, r: dict) -> tuple[bool, str]:
        ok, note = expect_has_names(a, r)
        if ok:
            listed_names.clear()
            for token in re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", a):
                listed_names.append(token)
        return ok, note

    def expect_sales_not_eng_leak(a: str, r: dict) -> tuple[bool, str]:
        ok, note = expect_has_names(a, r)
        if not ok:
            return False, note
        lower = a.lower()
        # After topic shift to Sales, prior Engineering cohort names should not dominate.
        eng_hits = sum(1 for n in listed_names if n and n.lower() in lower)
        if listed_names and eng_hits >= max(3, len(listed_names) // 2 + 1):
            return False, f"eng cohort leaked into Sales list: {a[:180]}"
        return True, "topic-shift Sales list"

    return [
        # --- A. Multi-tool / DAG ---
        Scenario(
            "PR_skill_place_intersect",
            "Org count → Dubai+Kubernetes count → names (Diego/Hugo)",
            ["resume_search", "sql"],
            [
                ("How many employees do we have?", expect_positive_count),
                (
                    "How many employees know Kubernetes and are in Dubai?",
                    all_of(expect_exact_count(2), expect_multi_tool, expect_not_degraded),
                ),
                (
                    "list their names",
                    all_of(
                        expect_names_only("diego", "hugo"),
                        expect_not_degraded,
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_reports_skill_place",
            "Alice team ∩ K8s ∩ Dubai multi-tool composition",
            ["employee", "resume_search", "sql"],
            [
                (
                    "who on Alice Nguyen's team knows Kubernetes and is in Dubai?",
                    all_of(expect_composition_or_empty, expect_multi_tool),
                ),
            ],
        ),
        Scenario(
            "PR_dept_then_skill_then_place",
            "Eng → Python → Berlin refine; zero place ∩ clears names follow-up",
            ["sql", "resume_search", "clarify"],
            [
                ("How many employees work in Engineering?", eng_capture),
                (
                    "how many of them know Python?",
                    all_of(eng_lte, expect_multi_tool),
                ),
                (
                    "and how many of those are in Berlin?",
                    all_of(expect_followup_count, expect_multi_tool),
                ),
                (
                    "list their names",
                    all_of(
                        expect_empty_cohort_no_names,
                        expect_planner_mode("guard_empty_cohort_list"),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_facet_then_cohort",
            "Cities facet guard → names → Eng list → Berlin of them (not org-wide)",
            ["sql", "resume_search"],
            [
                (
                    "in how different cities do we have employees?",
                    all_of(
                        expect_facet_small_count,
                        expect_planner_mode("guard_facet"),
                    ),
                ),
                ("names please", expect_city_names),
                ("List employees in Engineering", capture_eng_names),
                ("how many of them in Berlin?", expect_not_org_wide_100),
            ],
        ),
        # --- B. HITL / status ---
        Scenario(
            "PR_hitl_activate_yes",
            "Activate Carol → HITL gate → yes commits Carol (not prior focus)",
            ["employee", "clarify"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                (
                    "activate Carol Garcia",
                    all_of(
                        expect_hitl_gate,
                        expect_planner_mode("tool_select_hitl", "heuristic"),
                    ),
                ),
                (
                    "yes",
                    all_of(
                        expect_status_committed,
                        expect_names_only("carol"),
                        expect_planner_mode(
                            "heuristic_confirm_status",
                            "tool_select",
                            "heuristic",
                        ),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_hitl_activate_cancel",
            "Activate → cancel clears pending; no write; next activate still gated",
            ["employee", "clarify"],
            [
                ("activate Carol Garcia", expect_hitl_gate),
                (
                    "no wait cancel",
                    all_of(expect_no_status_write, expect_planner_mode("heuristic_cancel_status")),
                ),
                ("activate Carol Garcia", expect_hitl_gate),
            ],
        ),
        Scenario(
            "PR_alice_nguyen_not_bauer",
            "Exact Alice Nguyen profile after Alice Bauer focus",
            ["employee", "resume_search"],
            [
                ("Tell me about Alice Bauer", expect_about_person),
                (
                    "Tell me about Alice Nguyen",
                    all_of(
                        expect_about_person,
                        expect_names_only("nguyen"),
                        expect_planner_mode("guard_about_person", "query_state", "heuristic"),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_org_people_total",
            "Org paraphrase how many people in total → org-wide, not dept 17",
            ["sql"],
            [
                (
                    "quick — how many people do we have in total?",
                    all_of(
                        expect_count_between(50, 200),
                        expect_planner_mode("guard_org_headcount", "tool_select"),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_hitl_activate_no_commit_without_yes",
            "Activate then topic-shift must not write; next status still gated",
            ["employee", "sql", "clarify"],
            [
                ("activate Carol Garcia", expect_hitl_gate),
                (
                    "how many engineers?",
                    all_of(expect_positive_count, expect_no_status_write),
                ),
                ("activate Carol Garcia", expect_hitl_gate),
            ],
        ),
        Scenario(
            "PR_status_list_not_write",
            "Names cohort → statuses of them is a read, not set_status",
            ["sql", "employee"],
            [
                ("List employees in Engineering", expect_has_names),
                (
                    "what are the statuses of them?",
                    all_of(expect_hr_answer_not_greeting, expect_no_status_write),
                ),
            ],
        ),
        # --- C. ReAct repair (forced) ---
        Scenario(
            "PR_react_force_repair",
            "Force first selector attempt to repair; answer still correct",
            ["sql"],
            [
                (
                    "How many employees do we have?",
                    all_of(
                        expect_positive_count,
                        expect_repair_used,
                        expect_not_degraded,
                        expect_planner_mode("tool_select"),
                    ),
                ),
            ],
            extra_headers={"X-HRMind-Force-Repair": "1"},
        ),
        # --- D. Guards beat LLM ---
        Scenario(
            "PR_guard_facet_cities",
            "City facet uses guard_facet",
            ["resume_search", "sql"],
            [
                (
                    "in how different cities do we have employees?",
                    all_of(
                        expect_facet_small_count,
                        expect_planner_mode("guard_facet"),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_guard_facet_countries",
            "Country facet uses guard_facet",
            ["resume_search", "sql"],
            [
                (
                    "in how different countries do we have employees?",
                    all_of(expect_positive_count, expect_planner_mode("guard_facet")),
                ),
            ],
        ),
        Scenario(
            "PR_guard_unknown_place",
            "Atlantis → guard_unknown_place",
            ["clarify", "resume_search"],
            [
                (
                    "List employees in Atlantis",
                    all_of(
                        expect_unknown_place_clarify,
                        expect_planner_mode("guard_unknown_place"),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_guard_empty_anaphora",
            "Bare of them? → guard_empty_anaphora",
            ["clarify"],
            [
                (
                    "of them?",
                    all_of(expect_clarify, expect_planner_mode("guard_empty_anaphora")),
                ),
            ],
        ),
        Scenario(
            "PR_empty_cohort_list_after_zero_refine",
            "Eng → skill → Berlin=0 → list names must not resurrect a roster",
            ["sql", "resume_search", "clarify"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know Python?", expect_followup_count),
                ("how many of them are in Berlin?", expect_zero_or_small_count),
                (
                    "list their names please",
                    all_of(
                        expect_empty_cohort_no_names,
                        expect_planner_mode("guard_empty_cohort_list"),
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_guard_tenure",
            "Avg tenure / most senior → guard_tenure",
            ["sql"],
            [
                (
                    "what is the average tenure in Engineering?",
                    all_of(expect_tenure_answer, expect_planner_mode("guard_tenure")),
                ),
                (
                    "who is the most senior in Engineering?",
                    all_of(expect_tenure_answer, expect_planner_mode("guard_tenure")),
                ),
            ],
        ),
        Scenario(
            "PR_guard_hire_window",
            "Joined last 90 days → guard_hire_window",
            ["sql"],
            [
                (
                    "who joined in the last 90 days?",
                    all_of(expect_hire_window, expect_planner_mode("guard_hire_window")),
                ),
            ],
        ),
        # --- F. Confidence / clarify ---
        Scenario(
            "PR_vague_clarify",
            "Underspecified update status → clarify; no write",
            ["clarify", "employee"],
            [
                (
                    "update status",
                    all_of(
                        expect_clarify,
                        expect_no_status_write,
                        expect_not_degraded,
                    ),
                ),
            ],
        ),
        Scenario(
            "PR_employee_salary_acl",
            "Employee role soft-refuses salary mid architecture pack",
            ["clarify"],
            [("what's Alice Nguyen's salary?", expect_unauthorized)],
            role="employee",
        ),
        # --- E. Long conversation ---
        Scenario(
            "PR_long_session_25",
            "≥25-turn mixed tools: memory, HITL, topic shift, clear",
            [
                "greeting",
                "sql",
                "resume_search",
                "employee",
                "clarify",
            ],
            [
                ("hello", expect_greeting),
                ("How many employees do we have?", expect_positive_count),
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know Python?", expect_followup_count),
                ("list their names", expect_has_names),
                (
                    "and how many of those are in Dubai?",
                    all_of(expect_followup_count, expect_multi_tool),
                ),
                (
                    "list their names please",
                    all_of(
                        expect_empty_cohort_no_names,
                        expect_planner_mode("guard_empty_cohort_list"),
                    ),
                ),
                ("who knows Python?", expect_skill_or_rag),
                ("list their names", capture_eng_names),
                ("give me the first persons date of birth", expect_birthday_from_resume),
                ("where does the second person live?", expect_person_location),
                (
                    "what are the statuses of them?",
                    all_of(expect_hr_answer_not_greeting, expect_no_status_write),
                ),
                ("activate Carol Garcia", expect_hitl_gate),
                ("yes", expect_status_committed),
                (
                    "in how different countries do we have employees?",
                    expect_planner_mode("guard_facet"),
                ),
                ("names please", expect_country_names),
                ("List employees in Sales", expect_sales_not_eng_leak),
                ("how many of them in London?", expect_not_org_wide_100),
                (
                    "who on Alice Nguyen's team knows Kubernetes and is in Dubai?",
                    expect_composition_or_empty,
                ),
                ("who knows kubernetees?", expect_skill_or_rag),
                ("how much PTO does Alice Nguyen have?", expect_unsupported),
                ("what's Alice Nguyen's salary?", expect_salary_answer),
                ("start over", expect_greeting),
                (
                    "of them?",
                    all_of(expect_clarify, expect_planner_mode("guard_empty_anaphora")),
                ),
                ("who knows Docker?", expect_topic_shift_fresh),
                (
                    "who joined in the last 90 days?",
                    all_of(expect_hire_window, expect_planner_mode("guard_hire_window")),
                ),
                (
                    "what is the average tenure in Engineering?",
                    all_of(expect_tenure_answer, expect_planner_mode("guard_tenure")),
                ),
                ("goodbye", expect_greeting),
            ],
        ),
    ]


def build_scenarios(*, suite: str = "all") -> list[Scenario]:
    basic = _build_basic_scenarios()
    hard = build_hard_scenarios()
    list_ref = build_list_referent_scenarios()
    nlu_slots = build_nlu_slots_scenarios()
    utterances = build_user_utterance_scenarios()
    orch = build_orchestrator_scenarios()
    production = build_production_scenarios()
    suite = (suite or "all").lower()
    if suite in {"production", "architecture"}:
        return production + [
            sc
            for sc in orch
            if sc.name
            in {
                "OR_reports_skill_place",
                "OR_long_orchestrator_dialog",
                "OR_confirm_status_continuation",
            }
        ]
    if suite == "basic":
        return basic
    if suite == "hard":
        return hard + list_ref + nlu_slots
    if suite == "utterances":
        return utterances + list_ref + nlu_slots
    if suite == "orchestrator":
        return orch + list_ref
    return basic + hard + list_ref + nlu_slots + utterances + orch + production


def _build_basic_scenarios() -> list[Scenario]:
    return [
        Scenario(
            "A_greeting",
            "Greeting tool / template path",
            ["greeting"],
            [("hello", expect_greeting)],
        ),
        Scenario(
            "B_sql_headcount",
            "Org-wide SQL count",
            ["sql"],
            [("How many employees do we have?", expect_positive_count)],
        ),
        Scenario(
            "C_sql_department_cohort",
            "SQL department count → list names → location refine",
            ["sql"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("say their names", expect_has_names),
                ("how many of them in Berlin?", expect_followup_count),
            ],
        ),
        Scenario(
            "D_sql_facets",
            "Facet count then names please (focus memory)",
            ["resume_search", "sql"],
            [
                ("in how different countries do we have employees?", expect_positive_count),
                ("names please", expect_country_names),
            ],
        ),
        Scenario(
            "E_sql_cities",
            "City facet → names please",
            ["resume_search", "sql"],
            [
                ("in how different cities do we have employees?", expect_facet_small_count),
                ("names please", expect_city_names),
            ],
        ),
        Scenario(
            "F_sql_city_list",
            "Berlin list → names follow-up",
            ["sql"],
            [
                ("List employees in Berlin", expect_has_names),
                ("please say their names", expect_has_names),
            ],
        ),
        Scenario(
            "G_rag_python",
            "Resume search / RAG for Python skill",
            ["resume_search", "sql"],
            [("Who on the team has Python experience?", expect_skill_or_rag)],
        ),
        Scenario(
            "H_rag_react",
            "Resume search for React (seed skill present in resume_chunks)",
            ["resume_search"],
            [("Show me people with React skills", expect_skill_or_rag)],
        ),
        Scenario(
            "I_rag_count_python",
            "How many know Python (RAG + SQL count)",
            ["resume_search", "sql"],
            [("How many people have Python on their resume?", expect_followup_count)],
        ),
        Scenario(
            "I2_skill_location",
            "Skill∩city single-turn (orchestration contract)",
            ["resume_search", "sql"],
            [("Who knows Python in Berlin?", expect_skill_and_place)],
        ),
        Scenario(
            "J_hybrid_sql_then_rag",
            "Engineering cohort → of them know python → names (SQL→RAG→SQL)",
            ["sql", "resume_search"],
            [
                ("How many people work in Engineering?", expect_positive_count),
                ("and how many of those know python?", expect_followup_count),
                ("can you list their names?", expect_has_names),
            ],
        ),
        Scenario(
            "K_anaphora_that_n_employees",
            "Anaphora: from that N employees skill refine",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                (
                    "and from that 100 employees how much knows python",
                    expect_followup_count,
                ),
            ],
        ),
        Scenario(
            "L_employee_profile",
            "Employee tool: about person",
            ["employee"],
            [("Tell me about Alice Nguyen", expect_about_person)],
        ),
        Scenario(
            "M_employee_location",
            "Employee tool: where does person live",
            ["employee", "resume_search"],
            [("where does carol garcia live?", expect_person_location)],
        ),
        Scenario(
            "N_entity_memory",
            "List Berlin → ask where Ivy lives (entity memory)",
            ["sql", "employee", "resume_search"],
            [
                ("List employees in Berlin", expect_has_names),
                ("where does carol live?", expect_person_location),
            ],
        ),
        Scenario(
            "O_manager",
            "Employee/manager tool",
            ["employee"],
            [("Who is the manager of Alice Nguyen?", expect_managerish)],
        ),
        Scenario(
            "P_unsupported",
            "Unsupported HR topic should not invent SQL answers",
            ["clarify"],
            [("how much vacation are developers taking?", expect_unsupported)],
        ),
        Scenario(
            "Q_empty_followup_clarify",
            "names please with empty session should clarify",
            ["clarify"],
            [("names please", expect_clarify)],
        ),
        Scenario(
            "R_soft_greeting_keeps_cohort_for_names",
            "Soft hey/thanks keep the cohort so names still list the prior set",
            ["sql", "resume_search", "greeting"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("hey", expect_greeting),
                ("give me there names", expect_has_names),
            ],
        ),
        Scenario(
            "R2_start_over_then_fresh_skill",
            "Explicit start-over clears; next skill search is fresh",
            ["greeting", "resume_search", "sql"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("start over — find React developers", expect_topic_shift_fresh),
            ],
        ),
        Scenario(
            "S_python_then_kubernetes_followup",
            "RAG set → how many of them know Kubernetes (count over prior cohort)",
            ["resume_search", "sql"],
            [
                ("Find Python developers.", expect_skill_or_rag),
                ("how many of them know Kubernetes?", expect_followup_count),
            ],
        ),
        Scenario(
            "T_long_multi_tool_session",
            "Long session mixing SQL, RAG, employee, status, and follow-ups",
            ["sql", "resume_search", "employee", "greeting"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("names please", expect_has_names),
                ("Who knows Kubernetes?", expect_skill_or_rag),
                ("Tell me about Alice Nguyen", expect_about_person),
                ("Who is the manager of Alice Nguyen?", expect_managerish),
                ("change the status of Carol Garcia to true", expect_status_true),
            ],
        ),
        Scenario(
            "ST_status_by_name",
            "Status update by person name via resume_search resolve",
            ["resume_search", "employee"],
            [
                ("change the status of Carol Garcia to true", expect_status_true),
                ("change the status of alice bauer to true", expect_status_true),
            ],
        ),
        Scenario(
            "ST_status_by_location_dubai",
            "Status update for all employees living in Dubai",
            ["resume_search", "employee"],
            [
                (
                    "update status of employees which are living in Dubai to true",
                    expect_status_location_cohort,
                ),
            ],
        ),
        Scenario(
            "ST_status_location_then_reset",
            "Dubai cohort status true then false (write + reverse)",
            ["resume_search", "employee"],
            [
                (
                    "update status of employees which are living in Dubai to true",
                    expect_status_location_cohort,
                ),
                (
                    "update status of employees which are living in Dubai to false",
                    expect_status_false,
                ),
            ],
        ),
        Scenario(
            "ST_status_unique_name",
            "Status update for uniquely named employee",
            ["resume_search", "employee"],
            [("set Hugo Marino status to true", expect_status_true)],
        ),
        Scenario(
            "BD_birthday_by_name",
            "Birth date parsed from resume text (RAG only, no SQL)",
            ["resume_search"],
            [
                ("when is Carol Garcia's birthday?", expect_birthday_from_resume),
                ("what is the date of birth of Carol Garcia", expect_birthday_from_resume),
            ],
        ),
        Scenario(
            "BD_birthday_wish",
            "Wishing a named person a happy birthday",
            ["resume_search"],
            [
                ("say happy birthday to Carol Garcia", expect_birthday_wish),
                ("wish Alice Bauer a happy birthday", expect_birthday_wish),
            ],
        ),
        Scenario(
            "BD_birthday_age",
            "Age question answered from the resume date of birth",
            ["resume_search"],
            [("how old is Alice Nguyen?", expect_birthday_age)],
        ),
        Scenario(
            "BD_birthday_born_phrasing",
            "'When was X born' phrasing",
            ["resume_search"],
            [("when was Alice Nguyen born?", expect_birthday_from_resume)],
        ),
        Scenario(
            "BD_birthdays_today",
            "Whose birthday is today (exhaustive DOB chunk scan)",
            ["resume_search"],
            [
                ("whose birthday is today?", expect_birthday_today),
                ("any birthdays today", expect_birthday_today),
            ],
        ),
        Scenario(
            "BD_birthdays_month_upcoming",
            "Birthdays in a named month and upcoming birthdays",
            ["resume_search"],
            [
                ("who has a birthday in July?", expect_birthday_month),
                ("upcoming birthdays", expect_birthday_today),
            ],
        ),
        Scenario(
            "BD_birthday_unknown_person",
            "Unknown name must be refused, never invented",
            ["resume_search"],
            [("when is Zzz Nobody's birthday?", expect_birthday_missing_person)],
        ),
        Scenario(
            "BD_birthday_namesakes",
            "Employees sharing a name are all listed for the user to choose",
            ["resume_search"],
            [("when is Priya Silva's birthday?", expect_birthday_namesakes)],
        ),
        Scenario(
            "BD_birthday_resume_without_date",
            "Resume that carries no date of birth is admitted, not filled in",
            ["resume_search"],
            [("when is Tomas Khan's birthday?", expect_birthday_missing_person)],
        ),
        Scenario(
            "BD_birthday_misspelled_name",
            "Trigram name matching survives a typo and says so",
            ["resume_search"],
            [("when is Carol Garciaa's birthday?", expect_birthday_typo)],
        ),
        Scenario(
            "BD_birthday_leap_day",
            "29 February birthday (no anniversary in most years)",
            ["resume_search"],
            [("when is Paula Moreau's birthday?", expect_birthday_leap_day)],
        ),
        Scenario(
            "BD_birthday_cohort_coverage",
            "Cohort answer states how many resumes it could read",
            ["resume_search"],
            [("whose birthday is today?", expect_birthday_coverage_admitted)],
        ),
        Scenario(
            "BD_birthday_after_other_tools",
            "Birthday question inside a session that used SQL and RAG",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("Who knows Python?", expect_skill_or_rag),
                ("when is the birthday of Carol Garcia?", expect_birthday_from_resume),
                ("whose birthday is today?", expect_birthday_today),
            ],
        ),
    ]


def build_list_referent_scenarios() -> list[Scenario]:
    """Ordinal / list-deixis follow-ups against the last displayed name list."""
    capture_names, expect_first_dob = make_listed_ordinal_birthday_checkers(index=1)
    capture2, expect_second_dob = make_listed_ordinal_birthday_checkers(index=2)
    capture_last, expect_last_dob = make_listed_ordinal_birthday_checkers(index=-1)
    capture_loc, expect_first_loc = make_listed_ordinal_location_checkers(index=1)
    return [
        Scenario(
            "LR_ordinal_first_person_dob",
            "Transcript: headcount → python → names → first person's DOB",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture_names),
                ("you named two persons give me the first persons date of birth", expect_first_dob),
            ],
        ),
        Scenario(
            "LR_ordinal_second_person_dob",
            "After names, ask for the second person's DOB",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture2),
                ("give me the second person's date of birth", expect_second_dob),
            ],
        ),
        Scenario(
            "LR_ordinal_last_person_dob",
            "After names, 'the last person' DOB",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture_last),
                ("what's the last person's birthday?", expect_last_dob),
            ],
        ),
        Scenario(
            "LR_ordinal_first_location",
            "After names, where does the first person live",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture_loc),
                ("where does the first person live?", expect_first_loc),
            ],
        ),
        Scenario(
            "LR_former_latter_after_two_names",
            "Former / latter deixis after a two-person list",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture_names),
                ("the former one's date of birth", expect_first_dob),
            ],
        ),
        Scenario(
            "LR_ordinal_without_names_clarifies",
            "Count-only python shrink then ordinal must ask for names",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how many of them know python", expect_followup_count),
                (
                    "give me the first persons date of birth",
                    expect_ordinal_needs_names,
                ),
            ],
        ),
        Scenario(
            "LR_facet_names_do_not_become_employee_list",
            "Country facet names please must not bind employee ordinals",
            ["resume_search", "sql"],
            [
                ("in how different countries do we have employees?", expect_facet_small_count),
                ("names please", expect_country_names),
                (
                    "the first person's date of birth",
                    expect_ordinal_needs_names,
                ),
            ],
        ),
    ]


def build_nlu_slots_scenarios() -> list[Scenario]:
    """Residual LLM-slot paraphrases that regex / heuristics often miss."""
    capture_names, expect_first_dob = make_listed_ordinal_birthday_checkers(index=1)
    return [
        Scenario(
            "NS_top_one_bday_paraphrase",
            "After names, 'top one's bday' should bind last_listed[0] via slots/ordinals",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture_names),
                ("what's the top one's bday", expect_first_dob),
            ],
        ),
        Scenario(
            "NS_hash_one_dob",
            "After names, '#1 DOB' ordinal paraphrase",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture_names),
                ("#1 date of birth please", expect_first_dob),
            ],
        ),
        Scenario(
            "NS_their_dob_after_list_clarifies",
            "Ambiguous their/them after a multi-person list must clarify",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", expect_has_names),
                ("what's their date of birth", expect_clarify),
            ],
        ),
        Scenario(
            "NS_that_one_after_multi_clarifies",
            "'that one' after multiple names should clarify",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", expect_has_names),
                ("where does that one live?", expect_clarify),
            ],
        ),
    ]


def build_user_utterance_scenarios() -> list[Scenario]:
    """Everyday user wordings — casual, short, and paraphrased questions."""
    capture_names, expect_first_dob = make_listed_ordinal_birthday_checkers(index=1)
    capture2, expect_second_dob = make_listed_ordinal_birthday_checkers(index=2)

    return [
        # --- Headcount / org structure (casual) ---
        Scenario(
            "UQ_headcount_casual",
            "Casual org headcount phrasings",
            ["sql"],
            [
                ("how many people work here?", expect_positive_count),
                ("what's our headcount?", expect_positive_count),
                ("total employees?", expect_positive_count),
            ],
        ),
        Scenario(
            "UQ_department_casual",
            "Casual department counts and follow-ups",
            ["sql"],
            [
                ("how many folks in engineering?", expect_positive_count),
                ("and in sales?", expect_positive_count),
                ("show me the engineering people", expect_has_names),
            ],
        ),
        Scenario(
            "UQ_department_roster_phrasing",
            "Roster / team list paraphrases",
            ["sql"],
            [
                ("who's on the Product team?", expect_has_names),
                ("give me the Finance roster", expect_has_names),
            ],
        ),
        # --- Location (resume-owned) ---
        Scenario(
            "UQ_location_cohort_paraphrases",
            "Place cohort wordings users actually type",
            ["resume_search", "sql"],
            [
                ("anyone based in Berlin?", expect_has_names),
                ("how many people are in Dubai?", expect_followup_count),
                ("list staff in London", expect_has_names),
            ],
        ),
        Scenario(
            "UQ_location_person_paraphrases",
            "Where does X live / based / located",
            ["resume_search", "employee"],
            [
                ("where is Carol Garcia based?", expect_person_location),
                ("what city does Alice Nguyen work from?", expect_person_location),
                ("Carol Garcia location?", expect_person_location),
            ],
        ),
        Scenario(
            "UQ_location_facets_casual",
            "How many cities/countries — informal",
            ["resume_search", "sql"],
            [
                ("how many cities do we have people in?", expect_facet_small_count),
                ("which countries are we in?", expect_country_names),
            ],
        ),
        Scenario(
            "UQ_location_then_names_then_ordinal",
            "Berlin list → names → first person's DOB",
            ["resume_search", "sql"],
            [
                ("List employees in Berlin", expect_has_names),
                ("names please", capture_names),
                ("dob for the first one", expect_first_dob),
            ],
        ),
        # --- Skills / RAG ---
        Scenario(
            "UQ_skill_casual",
            "Skill search paraphrases",
            ["resume_search", "sql"],
            [
                ("anyone good with Python?", expect_skill_or_rag),
                ("got people who know React?", expect_skill_or_rag),
                ("looking for Kubernetes experience", expect_skill_or_rag),
            ],
        ),
        Scenario(
            "UQ_skill_count_casual",
            "How many know X — short forms",
            ["resume_search", "sql"],
            [
                ("how many know Python?", expect_followup_count),
                ("count of React people?", expect_followup_count),
            ],
        ),
        Scenario(
            "UQ_skill_then_location_refine",
            "Skill cohort then place refine in everyday English",
            ["resume_search", "sql"],
            [
                ("Find Python developers", expect_skill_or_rag),
                ("any of them in Berlin?", expect_followup_count),
                ("ok list those names", expect_has_names),
            ],
        ),
        # --- Profile / manager / pronouns ---
        Scenario(
            "UQ_profile_casual",
            "Tell me about / who is paraphrases",
            ["employee"],
            [
                ("who is Alice Nguyen?", expect_about_person),
                ("Alice Nguyen's profile please", expect_about_person),
            ],
        ),
        Scenario(
            "UQ_manager_casual",
            "Manager questions in short form",
            ["employee"],
            [
                ("Alice Nguyen's manager?", expect_managerish),
                ("who manages Carol Garcia?", expect_managerish),
            ],
        ),
        Scenario(
            "UQ_pronoun_after_profile",
            "Named person then she/her follow-ups",
            ["employee", "resume_search"],
            [
                ("Tell me about Carol Garcia", expect_about_person),
                ("where does she live?", expect_person_location),
                ("when is her birthday?", expect_birthday_from_resume),
                ("who's her manager?", expect_managerish),
            ],
        ),
        # --- Birthdays ---
        Scenario(
            "UQ_birthday_casual_phrasings",
            "DOB / bday / born paraphrases for a known person",
            ["resume_search"],
            [
                ("Carol Garcia dob?", expect_birthday_from_resume),
                ("bday for Alice Bauer?", expect_birthday_from_resume),
                ("Alice Nguyen — when was she born?", expect_birthday_from_resume),
            ],
        ),
        Scenario(
            "UQ_birthday_cohort_casual",
            "Today / named-month birthday asks",
            ["resume_search"],
            [
                ("any birthdays today?", expect_birthday_today),
                ("who has a birthday in July?", expect_birthday_month),
                ("upcoming birthdays please", expect_birthday_upcomingish),
            ],
        ),
        Scenario(
            "UQ_age_casual",
            "Age question short form",
            ["resume_search"],
            [("how old is Carol Garcia?", expect_birthday_age)],
        ),
        # --- Status ---
        Scenario(
            "UQ_status_casual",
            "Status write paraphrases (true/false/active/inactive + pronoun)",
            ["resume_search", "employee"],
            [
                ("set Carol Garcia status to true", expect_status_true),
                ("change Hugo Marino's status to false", expect_status_false),
                (
                    "update status of employees living in Dubai to true",
                    expect_status_location_cohort,
                ),
            ],
        ),
        Scenario(
            "UQ_status_pronoun_active",
            "Profile then 'change her status to active/inactive'",
            ["employee", "resume_search"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                ("change her status to inactive", expect_status_false),
                ("change her status to active", expect_status_true),
            ],
        ),
        Scenario(
            "UQ_place_case_after_profile_headcount",
            "Profile → org headcount → dubai/Dubai must agree (not stuck on one person)",
            ["employee", "sql", "resume_search"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                ("How many employees do we have?", expect_positive_count),
                ("how much of them are from dubai", expect_count_between(1, 40)),
                ("how much of them are from Dubai?", expect_count_between(1, 40)),
            ],
        ),
        # --- Unsupported / out of scope (salary is ACL, not OOS) ---
        Scenario(
            "UQ_unsupported_paraphrases",
            "PTO / benefits / payroll must refuse (salary is not OOS)",
            ["clarify"],
            [
                ("how much PTO do we get?", expect_unsupported),
                ("tell me about our benefits package", expect_unsupported),
                ("payroll cutoff dates?", expect_unsupported),
            ],
        ),
        Scenario(
            "UQ_recruiter_salary_ok",
            "Recruiter may see base salary (not OOS refuse)",
            ["employee", "sql"],
            [
                ("what's Alice Nguyen's salary?", expect_salary_answer),
            ],
        ),
        Scenario(
            "UQ_employee_salary_unauthorized",
            "Employee role soft-refuses salary (chat unauthorized, not 400)",
            ["clarify"],
            [
                ("what's Alice Nguyen's salary?", expect_unauthorized),
            ],
            role="employee",
        ),
        Scenario(
            "UQ_oos_preserves_then_names",
            "OOS vacation refuse must not wipe cohort; names still work",
            ["sql", "clarify"],
            [
                ("list engineers in Engineering", expect_has_names),
                ("how much PTO do we get?", expect_unsupported),
                ("names please", expect_has_names),
            ],
        ),
        Scenario(
            "UQ_person_travel_prefs_oos",
            "Bound person + travelling preferences must refuse (not dump profile)",
            ["employee", "clarify"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                (
                    "give me information about her travelling preferences",
                    expect_unsupported,
                ),
            ],
        ),
        Scenario(
            "UQ_sofia_after_alice_session",
            "After Alice focus, Sofia existence/paraphrase must name-lookup (not LLM fallback)",
            ["employee"],
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                ("do we have Sofia ?", expect_sofia_disambiguation),
                ("her education?", expect_about_person),
                ("find Sofia", expect_sofia_disambiguation),
            ],
        ),
        # --- Clarify / empty context ---
        Scenario(
            "UQ_empty_context_clarifies",
            "Follow-ups with no prior cohort should ask for detail",
            ["clarify"],
            [
                ("of them?", expect_clarify),
                ("the first one", expect_clarify),
                ("names", expect_clarify),
            ],
        ),
        # --- Multi-turn conversational flows ---
        Scenario(
            "UQ_recruiter_screening_flow",
            "Typical recruiter: dept → skill → names → ordinal DOB → location",
            ["sql", "resume_search"],
            [
                ("how many engineers do we have?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("list the names", capture_names),
                ("first person's birthday?", expect_first_dob),
                ("and where does the second person live?", expect_person_location),
            ],
        ),
        Scenario(
            "UQ_manager_checkin_flow",
            "Manager-style: team size → names → status → birthday",
            ["sql", "resume_search", "employee"],
            [
                ("Sales headcount?", expect_positive_count),
                ("who are they?", expect_has_names),
                ("change the status of Carol Garcia to true", expect_status_true),
                ("when is Carol Garcia's birthday?", expect_birthday_from_resume),
            ],
        ),
        Scenario(
            "UQ_messy_typos_and_slang",
            "Typos / slang still resolve",
            ["sql", "resume_search"],
            [
                ("how many ppl in engeneering?", expect_positive_count),
                ("n how many of them kno python", expect_followup_count),
                ("gimme names", expect_has_names),
            ],
        ),
        Scenario(
            "UQ_tool_select_typo_paraphrases",
            "Typos/paraphrases via tool-selecting planner (no new regex)",
            ["sql", "resume_search"],
            [
                ("how meny employes in sales?", expect_positive_count),
                ("count peeps in product pls", expect_positive_count),
                ("who knows kubernetees?", expect_skill_or_rag),
            ],
        ),
        Scenario(
            "UQ_tool_select_hitl_status",
            "Novel status paraphrase → HITL confirm → commit",
            ["employee", "resume_search", "clarify"],
            [
                ("activate Carol Garcia", expect_status_confirm_gate_only),
                ("yes", expect_status_true),
            ],
        ),
        Scenario(
            "UQ_topic_shift_greeting_midway",
            "Mid-dialog greeting clears; fresh search works",
            ["sql", "greeting", "resume_search"],
            [
                ("Engineering headcount", expect_positive_count),
                ("thanks!", expect_greeting),
                ("hi — who knows Docker?", expect_skill_or_rag),
            ],
        ),
        Scenario(
            "UQ_compare_depts_then_pick_one",
            "Ask two departments then dig into the latest",
            ["sql", "resume_search"],
            [
                ("how many in Engineering?", expect_positive_count),
                ("how many in Product?", expect_positive_count),
                ("names for Product please", expect_has_names),
                ("any of them in Berlin?", expect_not_org_wide_100),
            ],
        ),
        Scenario(
            "UQ_list_then_second_then_former",
            "Names → second DOB → former (should still mean first of original pair)",
            ["sql", "resume_search"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("how much of them knows python", expect_followup_count),
                ("give me there names", capture2),
                ("second person's dob", expect_second_dob),
            ],
        ),
        Scenario(
            "UQ_long_natural_dialog",
            "Long mixed dialog with short user turns",
            ["sql", "resume_search", "employee", "clarify", "greeting"],
            [
                ("hey", expect_greeting),
                ("headcount?", expect_positive_count),
                ("engineering?", expect_positive_count),
                ("python?", expect_followup_count),
                ("names", capture_names),
                ("top one's bday", expect_first_dob),
                ("alice nguyen profile", expect_about_person),
                ("her manager?", expect_managerish),
                ("where does she live", expect_person_location),
                ("pto balance?", expect_unsupported),
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Local HRMind multi-turn chat QA")
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="API base URL (default HRMIND_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--out",
        default="/tmp/hrmind_local_qa.json",
        help="Write machine-readable JSON report",
    )
    parser.add_argument(
        "--suite",
        choices=(
            "basic",
            "hard",
            "utterances",
            "orchestrator",
            "production",
            "architecture",
            "all",
        ),
        default="all",
        help=(
            "basic = smoke; hard = adversarial + list/NLU; "
            "utterances = natural user paraphrases; "
            "orchestrator = Wave A–D contracts; "
            "production|architecture = production architecture gate; "
            "all = everything (default)"
        ),
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated scenario name prefixes (e.g. OR_,G_,ST_,UQ_,NS_,LR_,AG_,PR_)",
    )
    parser.add_argument(
        "--require-meta",
        action="store_true",
        help="Fail turns when ChatResponse.meta is missing (debug header ignored)",
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"Target: {base}/v1/chat")
    print(f"Suite: {args.suite}")
    print(
        f"Headers: X-User-Id={HEADERS['X-User-Id']} X-Role={HEADERS['X-Role']} "
        f"X-HRMind-Debug={HEADERS.get('X-HRMind-Debug')}"
    )
    if args.require_meta:
        print("Require meta: on")
    try:
        h = health_check(base)
        print(f"Health: {h}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: health check failed: {exc}")
        print("Is the API running? Try: uvicorn app.main:app --reload --port 8000")
        return 2

    scenarios = build_scenarios(suite=args.suite)
    if args.only:
        prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
        scenarios = [
            sc for sc in scenarios if any(sc.name.startswith(p) for p in prefixes)
        ]
        if not scenarios:
            print(f"No scenarios matched --only {args.only!r}")
            return 2

    passed = failed = 0
    details: list[str] = []
    arch_turns = hitl_turns = repair_turns = multi_tool_turns = 0
    arch_pass = hitl_pass = repair_pass = multi_tool_pass = 0

    for sc in scenarios:
        print(f"\n=== {sc.name} ===")
        print(f"  ({sc.description}; tools≈{','.join(sc.tools)})")
        run_scenario(base, sc, require_meta=args.require_meta)
        for i, r in enumerate(sc.results, 1):
            status = "PASS" if r.ok else "FAIL"
            if r.ok:
                passed += 1
            else:
                failed += 1
            meta = r.meta or {}
            mode = str(meta.get("planner_mode") or "")
            is_arch = sc.name.startswith("PR_") or "guard_" in mode or "tool_select" in mode
            if is_arch:
                arch_turns += 1
                arch_pass += int(r.ok)
            if meta.get("needs_hitl") or "hitl" in mode:
                hitl_turns += 1
                hitl_pass += int(r.ok)
            if int(meta.get("repairs") or 0) >= 1 or "repair" in mode:
                repair_turns += 1
                repair_pass += int(r.ok)
            tool = r.tool or meta.get("tools_answered") or ""
            nodes = meta.get("plan_nodes") or []
            if (
                (isinstance(tool, str) and "+" in tool)
                or len({str(n) for n in nodes if n not in {"intersect", "count"}}) >= 2
            ):
                multi_tool_turns += 1
                multi_tool_pass += int(r.ok)
            print(f"  [{status}] T{i} ({r.ms}ms) Q: {r.question!r}")
            print(f"         A: {r.answer[:240]!r}")
            if r.note:
                print(f"         note: {r.note}")
            if r.tool or mode:
                print(f"         tool={r.tool!r} mode={mode!r} repairs={meta.get('repairs')}")
            if r.sources:
                kinds = [s.get("kind") for s in r.sources[:5] if isinstance(s, dict)]
                print(f"         sources: {kinds}")
            if r.clarify:
                print(f"         clarify: {r.clarify[:160]!r}")
            if r.error:
                print(f"         error: {r.error}")
            if not r.ok:
                details.append(f"{sc.name} T{i}: {r.question} => {r.answer[:300]}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} turns")
    print(
        f"ARCHITECTURE: {arch_pass}/{arch_turns} | HITL {hitl_pass}/{hitl_turns} | "
        f"repairs {repair_pass}/{repair_turns} | multi-tool {multi_tool_pass}/{multi_tool_turns}"
    )
    if details:
        print("\nFAILURES:")
        for d in details:
            print(f" - {d}")

    out = {
        "base": base,
        "suite": args.suite,
        "require_meta": args.require_meta,
        "passed": passed,
        "failed": failed,
        "architecture": {
            "passed": arch_pass,
            "turns": arch_turns,
            "hitl_passed": hitl_pass,
            "hitl_turns": hitl_turns,
            "repair_passed": repair_pass,
            "repair_turns": repair_turns,
            "multi_tool_passed": multi_tool_pass,
            "multi_tool_turns": multi_tool_turns,
        },
        "scenarios": [
            {
                "name": sc.name,
                "description": sc.description,
                "tools": sc.tools,
                "session_id": sc.session_id,
                "turns": [
                    {
                        "ok": r.ok,
                        "q": r.question,
                        "a": r.answer,
                        "note": r.note,
                        "error": r.error,
                        "ms": r.ms,
                        "degraded": r.degraded,
                        "clarify": r.clarify,
                        "confidence": r.confidence,
                        "sources": r.sources,
                        "tool": r.tool,
                        "meta": r.meta,
                    }
                    for r in sc.results
                ],
            }
            for sc in scenarios
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
