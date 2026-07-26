#!/usr/bin/env python3
"""End-to-end chat QA against production — multi-turn context + isolated tools."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE = "https://hrmind-agent-production.up.railway.app"
CHAT = f"{BASE}/v1/chat"
HEADERS = {
    "Content-Type": "application/json",
    "X-User-Id": "qa-tester-full",
    "X-Role": "recruiter",
}


@dataclass
class TurnResult:
    question: str
    answer: str
    session_id: str
    degraded: bool
    clarify: str | None
    ok: bool
    note: str = ""
    error: str | None = None
    ms: int = 0


@dataclass
class Scenario:
    name: str
    turns: list[tuple[str, Any]]  # (question, checker callable(answer, resp)->(ok, note))
    results: list[TurnResult] = field(default_factory=list)
    session_id: str | None = None


def chat(question: str, session_id: str | None = None) -> dict:
    body: dict[str, Any] = {"question": question}
    if session_id:
        body["session_id"] = session_id
    req = urllib.request.Request(
        CHAT,
        data=json.dumps(body).encode(),
        headers=HEADERS,
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())
    data["_ms"] = int((time.time() - t0) * 1000)
    return data


def contains_any(text: str, *needles: str) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def not_contains(text: str, *needles: str) -> bool:
    lower = text.lower()
    return all(n.lower() not in lower for n in needles)


def run_scenario(sc: Scenario) -> Scenario:
    sid = None
    for q, checker in sc.turns:
        try:
            data = chat(q, sid)
            sid = data.get("session_id") or sid
            answer = data.get("answer") or ""
            ok, note = checker(answer, data)
            sc.results.append(
                TurnResult(
                    question=q,
                    answer=answer[:500],
                    session_id=sid or "",
                    degraded=bool(data.get("degraded")),
                    clarify=data.get("clarify"),
                    ok=ok,
                    note=note,
                    ms=int(data.get("_ms") or 0),
                )
            )
        except Exception as exc:  # noqa: BLE001
            sc.results.append(
                TurnResult(
                    question=q,
                    answer="",
                    session_id=sid or "",
                    degraded=True,
                    clarify=None,
                    ok=False,
                    note="request failed",
                    error=str(exc),
                )
            )
            break
    sc.session_id = sid
    return sc


# --- Checkers ---

def expect_greeting(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(a, "hello", "hi", "help", "hrmind", "assist")
    return ok, "greeting-like" if ok else f"unexpected: {a[:120]}"


def expect_country_count(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(a, "5", "five") and contains_any(a, "countr")
    return ok, "country cardinality" if ok else f"got: {a[:160]}"


def expect_country_names(a: str, _r: dict) -> tuple[bool, str]:
    countries = ["germany", "usa", "united states", "uk", "united kingdom", "uae", "france", "dubai"]
    hits = sum(1 for c in countries if c in a.lower())
    # Should list countries, not refuse, not dump only first names without country context
    refused = contains_any(a, "don't have access", "do not have access", "specify a department")
    ok = hits >= 3 and not refused
    return ok, f"country hits={hits}" if ok else f"hits={hits} refused={refused}: {a[:200]}"


def expect_engineering_count(a: str, _r: dict) -> tuple[bool, str]:
    # numeric answer expected
    import re

    m = re.search(r"\b(\d+)\b", a)
    ok = m is not None and int(m.group(1)) > 0 and not_contains(a, "don't have that information")
    return ok, f"n={m.group(1) if m else None}" if ok else f"got: {a[:160]}"


def expect_has_names(a: str, _r: dict) -> tuple[bool, str]:
    # employee name listing
    refused = contains_any(a, "don't have access", "which names should i list")
    has_name = contains_any(
        a, "nguyen", "smith", "garcia", "mueller", "khan", "chen", "kim", "martin", "brown"
    )
    ok = has_name and not refused
    return ok, "employee names" if ok else f"refused={refused}: {a[:180]}"


def expect_python_skill(a: str, _r: dict) -> tuple[bool, str]:
    ok = (
        contains_any(a, "python", "answer is", "found", "matching", "employee", "know")
        or any(c.isdigit() for c in a)
    ) and not_contains(a, "don't have that information in the hr data")
    # soft: not a pure greeting
    ok = ok and not (a.lower().startswith("hello") and len(a) < 80)
    return ok, "skill/rag response" if ok else f"got: {a[:180]}"


def expect_person_location(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(a, "berlin", "dubai", "london", "paris", "new york", "germany", "usa", "uk", "uae", "france", "city", "live", "based", "located")
    ok = ok and not_contains(a, "don't have access to employee")
    return ok, "location-ish" if ok else f"got: {a[:180]}"


def expect_unsupported(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(a, "don't have", "do not have", "not have that information", "can't access", "cannot")
    # should NOT invent a headcount as the main answer for vacation
    inventing = a.strip().lower().startswith("the answer is") and any(ch.isdigit() for ch in a)
    ok = ok and not inventing
    return ok, "unsupported handled" if ok else f"got: {a[:180]}"


def expect_managerish(a: str, _r: dict) -> tuple[bool, str]:
    ok = len(a.strip()) > 10 and not_contains(a, "internal_error")
    return ok, "manager reply" if ok else f"got: {a[:160]}"


def expect_city_list(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(a, "berlin", "employee", "matching") and not_contains(a, "internal_error")
    return ok, "city list" if ok else f"got: {a[:160]}"


def expect_followup_count_reasonable(a: str, _r: dict) -> tuple[bool, str]:
    import re

    m = re.search(r"\b(\d+)\b", a)
    if not m:
        return False, f"no number: {a[:160]}"
    n = int(m.group(1))
    # of them know python should be <= org size, typically small
    ok = 0 <= n <= 100 and not_contains(a, "don't have access")
    return ok, f"n={n}" if ok else f"n={n}: {a[:160]}"


def expect_about_person(a: str, _r: dict) -> tuple[bool, str]:
    ok = contains_any(a, "ivy", "chen", "department", "engineer", "position", "employee") and len(a) > 20
    return ok, "profile-ish" if ok else f"got: {a[:180]}"


def build_scenarios() -> list[Scenario]:
    return [
        # --- A. Facet context (the bug we fixed) ---
        Scenario(
            "A1 countries → names please",
            [
                ("in how different countries do we have employees?", expect_country_count),
                ("names please", expect_country_names),
            ],
        ),
        Scenario(
            "A2 which countries direct",
            [
                ("which countries do we have employees in?", expect_country_names),
            ],
        ),
        # --- B. SQL cohort context ---
        Scenario(
            "B1 Engineering count → names → of them python",
            [
                ("How many employees work in Engineering?", expect_engineering_count),
                ("say their names", expect_has_names),
                ("how many of them know python?", expect_followup_count_reasonable),
            ],
        ),
        Scenario(
            "B2 city list → names follow-up",
            [
                ("List employees in Berlin", expect_city_list),
                ("please say their names", expect_has_names),
            ],
        ),
        Scenario(
            "B3 Engineering → of them in Berlin",
            [
                ("How many employees work in Engineering?", expect_engineering_count),
                ("how many of them in Berlin?", expect_followup_count_reasonable),
            ],
        ),
        # --- C. RAG / skills ---
        Scenario(
            "C1 who knows python (RAG alone)",
            [
                ("Who knows Python?", expect_python_skill),
            ],
        ),
        Scenario(
            "C2 how many know python",
            [
                ("How many employees know Python?", expect_followup_count_reasonable),
            ],
        ),
        Scenario(
            "C3 developing in python phrasing",
            [
                ("how much of them are developing in python?", expect_followup_count_reasonable),
            ],
        ),
        Scenario(
            "C4 AWS experience",
            [
                ("Show employees with AWS experience.", expect_python_skill),
            ],
        ),
        # --- D. Hybrid SQL→RAG context ---
        Scenario(
            "D1 Eng count → of them know python → names",
            [
                ("How many employees work in Engineering?", expect_engineering_count),
                ("how many of them know python?", expect_followup_count_reasonable),
                ("names please", expect_has_names),
            ],
        ),
        # --- E. Person / employee tool ---
        Scenario(
            "E1 where does Ivy Chen live",
            [
                ("where the ivy chen lives?", expect_person_location),
            ],
        ),
        Scenario(
            "E2 tell me about Ivy Chen",
            [
                ("Tell me about Ivy Chen", expect_about_person),
            ],
        ),
        Scenario(
            "E3 names then where ivy lives (entity memory)",
            [
                ("List employees in Berlin", expect_city_list),
                ("where does ivy live?", expect_person_location),
            ],
        ),
        # --- F. Unsupported / greeting ---
        Scenario(
            "F1 greeting",
            [
                ("hello", expect_greeting),
            ],
        ),
        Scenario(
            "F2 vacation unsupported",
            [
                ("how much vacation are developers taking?", expect_unsupported),
            ],
        ),
        Scenario(
            "F3 names please with empty session clarifies",
            [
                (
                    "names please",
                    lambda a, _r: (
                        contains_any(a, "which names", "countries", "cities", "departments", "previous", "clarify", "specify")
                        or contains_any(a, "don't have", "need more"),
                        "clarify-ish" if True else a[:120],
                    ),
                ),
            ],
        ),
        # --- G. Manager ---
        Scenario(
            "G1 manager question",
            [
                ("Who is the manager of Alice Nguyen?", expect_managerish),
            ],
        ),
        # --- H. More SQL alone ---
        Scenario(
            "H1 headcount how many employees",
            [
                ("How many employees do we have?", expect_engineering_count),
            ],
        ),
        Scenario(
            "H2 how many different cities",
            [
                (
                    "How many different cities do we have employees in?",
                    lambda a, _r: (
                        any(ch.isdigit() for ch in a) and contains_any(a, "cit", "answer", "have"),
                        a[:160],
                    ),
                ),
            ],
        ),
        Scenario(
            "H3 cities → names please",
            [
                ("How many different cities do we have employees in?", expect_engineering_count),
                (
                    "names please",
                    lambda a, _r: (
                        contains_any(a, "berlin", "dubai", "london", "paris", "new york", "cities", "city")
                        and not_contains(a, "don't have access"),
                        a[:200],
                    ),
                ),
            ],
        ),
    ]


def main() -> int:
    print(f"Target: {CHAT}")
    scenarios = build_scenarios()
    passed = failed = 0
    details: list[str] = []

    for sc in scenarios:
        print(f"\n=== {sc.name} ===")
        run_scenario(sc)
        for i, r in enumerate(sc.results, 1):
            status = "PASS" if r.ok else "FAIL"
            if r.ok:
                passed += 1
            else:
                failed += 1
            line = f"  [{status}] T{i} ({r.ms}ms) Q: {r.question!r}"
            print(line)
            print(f"         A: {r.answer[:220]!r}")
            if r.note:
                print(f"         note: {r.note}")
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
    # machine-readable
    out = {
        "passed": passed,
        "failed": failed,
        "scenarios": [
            {
                "name": sc.name,
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
                    }
                    for r in sc.results
                ],
            }
            for sc in scenarios
        ],
    }
    with open("/tmp/hrmind_qa_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote /tmp/hrmind_qa_results.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
