#!/usr/bin/env python3
"""
Local multi-turn chat QA for HRMind.

Exercises greeting, SQL, RAG (resume_search), employee lookup, manager,
unsupported topics, and context/referent follow-ups against a running API.

Usage:
  source .venv/bin/activate
  python scripts/qa_chat_local.py                 # basic + hard
  python scripts/qa_chat_local.py --suite basic
  python scripts/qa_chat_local.py --suite hard    # adversarial dialogs
  python scripts/qa_chat_local.py --only U_,AG_
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
from typing import Any, Callable

Checker = Callable[[str, dict[str, Any]], tuple[bool, str]]

DEFAULT_BASE = os.environ.get("HRMIND_BASE_URL", "http://localhost:8000").rstrip("/")
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-User-Id": os.environ.get("HRMIND_QA_USER", "local-qa"),
    "X-Role": os.environ.get("HRMIND_QA_ROLE", "recruiter"),
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


@dataclass
class Scenario:
    name: str
    description: str
    tools: list[str]
    turns: list[tuple[str, Checker]]
    results: list[TurnResult] = field(default_factory=list)
    session_id: str | None = None


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
) -> dict[str, Any]:
    body: dict[str, Any] = {"question": question}
    if session_id:
        body["session_id"] = session_id
    req = urllib.request.Request(
        f"{base}/v1/chat",
        data=json.dumps(body).encode(),
        headers=HEADERS,
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
    ok = contains_any(a, "hello", "hi", "help", "hrmind", "assist")
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
    ]
    hits = sum(1 for c in countries if c in a.lower())
    refused = contains_any(a, "don't have access", "do not have access")
    ok = hits >= 3 and not refused
    return ok, f"country hits={hits}" if ok else f"hits={hits}: {a[:180]}"


def expect_city_names(a: str, _r: dict) -> tuple[bool, str]:
    cities = ["berlin", "dubai", "london", "paris", "new york"]
    hits = sum(1 for c in cities if c in a.lower())
    ok = hits >= 2 and not_contains(a, "don't have access", "internal_error")
    return ok, f"city hits={hits}" if ok else f"hits={hits}: {a[:180]}"


def expect_has_names(a: str, _r: dict) -> tuple[bool, str]:
    refused = contains_any(a, "don't have access", "which names should i list")
    has_name = contains_any(
        a,
        "nguyen",
        "smith",
        "garcia",
        "mueller",
        "khan",
        "chen",
        "kim",
        "martin",
        "brown",
        "patel",
        "lopez",
        "alice",
        "ivy",
        "bob",
        "carol",
    )
    ok = has_name and not refused
    return ok, "employee names" if ok else f"refused={refused}: {a[:180]}"


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
        "germany",
        "usa",
        "uk",
        "uae",
        "france",
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
        contains_any(a, "ivy", "chen", "department", "engineer", "position", "employee")
        and len(a) > 20
        and not_contains(a, "internal_error")
    )
    return ok, "profile" if ok else f"got: {a[:180]}"


def expect_managerish(a: str, _r: dict) -> tuple[bool, str]:
    ok = len(a.strip()) > 10 and not_contains(a, "internal_error", "traceback")
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


def expect_clarify(a: str, r: dict) -> tuple[bool, str]:
    clarify = (r.get("clarify") or "") + " " + a
    ok = contains_any(
        clarify,
        "which",
        "clarify",
        "specify",
        "previous",
        "countries",
        "cities",
        "departments",
        "need more",
        "don't have",
    )
    return ok, "clarify-ish" if ok else f"got: {a[:160]}"


def expect_topic_shift_fresh(a: str, _r: dict) -> tuple[bool, str]:
    # After "hello" / start over, a fresh skill question should still answer.
    return expect_skill_or_rag(a, _r)


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


def run_scenario(base: str, sc: Scenario) -> Scenario:
    sid = None
    for q, checker in sc.turns:
        try:
            data = chat(base, q, sid)
            sid = data.get("session_id") or sid
            answer = data.get("answer") or ""
            sources = data.get("sources") or []
            ok, note = checker(answer, data)
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
            ["sql"],
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
            ["employee"],
            [
                ("Tell me about Ivy Chen", expect_about_person),
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
                ("Tell me about Ivy Chen", expect_about_person),
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


def build_scenarios(*, suite: str = "all") -> list[Scenario]:
    basic = _build_basic_scenarios()
    hard = build_hard_scenarios()
    suite = (suite or "all").lower()
    if suite == "basic":
        return basic
    if suite == "hard":
        return hard
    return basic + hard


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
            ["sql"],
            [
                ("in how different countries do we have employees?", expect_positive_count),
                ("names please", expect_country_names),
            ],
        ),
        Scenario(
            "E_sql_cities",
            "City facet → names please",
            ["sql"],
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
            [("Who knows Python?", expect_skill_or_rag)],
        ),
        Scenario(
            "H_rag_react",
            "Resume search for React (seed skill present in resume_chunks)",
            ["resume_search"],
            [("Who knows React?", expect_skill_or_rag)],
        ),
        Scenario(
            "I_rag_count_python",
            "How many know Python (RAG + SQL count)",
            ["resume_search", "sql"],
            [("How many employees know Python?", expect_followup_count)],
        ),
        Scenario(
            "J_hybrid_sql_then_rag",
            "Engineering cohort → of them know python → names (SQL→RAG→SQL)",
            ["sql", "resume_search"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("names please", expect_has_names),
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
            [("Tell me about Ivy Chen", expect_about_person)],
        ),
        Scenario(
            "M_employee_location",
            "Employee tool: where does person live",
            ["employee"],
            [("where the ivy chen lives?", expect_person_location)],
        ),
        Scenario(
            "N_entity_memory",
            "List Berlin → ask where Ivy lives (entity memory)",
            ["sql", "employee"],
            [
                ("List employees in Berlin", expect_has_names),
                ("where does ivy live?", expect_person_location),
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
            "R_topic_shift_clears_referent",
            "Greeting clears cohort; next skill search is fresh",
            ["greeting", "resume_search", "sql"],
            [
                ("How many employees work in Engineering?", expect_positive_count),
                ("hello", expect_greeting),
                ("Who knows React?", expect_topic_shift_fresh),
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
            "Long session mixing SQL, RAG, employee, and follow-ups",
            ["sql", "resume_search", "employee", "greeting"],
            [
                ("How many employees do we have?", expect_positive_count),
                ("How many employees work in Engineering?", expect_positive_count),
                ("how many of them know python?", expect_followup_count),
                ("names please", expect_has_names),
                ("Who knows Kubernetes?", expect_skill_or_rag),
                ("Tell me about Ivy Chen", expect_about_person),
                ("Who is the manager of Alice Nguyen?", expect_managerish),
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
        choices=("basic", "hard", "all"),
        default="all",
        help="basic = smoke suite; hard = adversarial dialogs; all = both (default)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated scenario name prefixes to run (e.g. G_,J_,U_,AG_)",
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"Target: {base}/v1/chat")
    print(f"Suite: {args.suite}")
    print(f"Headers: X-User-Id={HEADERS['X-User-Id']} X-Role={HEADERS['X-Role']}")
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

    for sc in scenarios:
        print(f"\n=== {sc.name} ===")
        print(f"  ({sc.description}; tools≈{','.join(sc.tools)})")
        run_scenario(base, sc)
        for i, r in enumerate(sc.results, 1):
            status = "PASS" if r.ok else "FAIL"
            if r.ok:
                passed += 1
            else:
                failed += 1
            print(f"  [{status}] T{i} ({r.ms}ms) Q: {r.question!r}")
            print(f"         A: {r.answer[:240]!r}")
            if r.note:
                print(f"         note: {r.note}")
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
    if details:
        print("\nFAILURES:")
        for d in details:
            print(f" - {d}")

    out = {
        "base": base,
        "passed": passed,
        "failed": failed,
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
