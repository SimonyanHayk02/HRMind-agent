#!/usr/bin/env python3
"""End-to-end chat QA against production — multi-turn context + isolated tools."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
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
    return ok, "greeting-like" if ok else f"unexpected: {a[:120]}"


def expect_country_count(a: str, _r: dict) -> tuple[bool, str]:
    import re

    m = re.search(r"\b(\d+)\b", a)
    n = int(m.group(1)) if m else None
    # Closed place vocab can grow; accept a small positive country cardinality.
    ok = n is not None and 3 <= n <= 20 and contains_any(a, "countr", "answer", "have")
    return ok, f"country n={n}" if ok else f"got: {a[:160]}"


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
    # employee name listing
    refused = contains_any(a, "don't have access", "which names should i list")
    candidates = seeded_names() or ("nguyen", "smith", "garcia", "khan", "martin")
    has_name = contains_any(a, *candidates)
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
        "which",
        "clarify",
    )
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
    ok = contains_any(
        a, "department", "engineer", "position", "employee", *seeded_names()
    ) and len(a) > 20
    return ok, "profile-ish" if ok else f"got: {a[:180]}"


def expect_status_updated(a: str, _r: dict) -> tuple[bool, str]:
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
    clarify = ((_r.get("clarify") or "") + " " + a).lower()
    if contains_any(
        clarify,
        "confirm status update",
        "confirm",
        "proceed",
        "employees match",
        "which",
    ) and not contains_any(a, "internal_error", "traceback"):
        return True, "status confirm / clarify gate"
    ok, note = expect_status_true(a, _r)
    if not ok:
        return False, note
    import re

    m = re.search(r"\b(\d+)\b", a)
    n = int(m.group(1)) if m else None
    ok = n is not None and n >= 1
    return ok, f"location cohort n={n}" if ok else f"no count: {a[:200]}"


def expect_not_greeting(a: str, _r: dict) -> tuple[bool, str]:
    lower = a.strip().lower()
    greetingish = (
        lower.startswith(("hello", "hi ", "hey", "hi!"))
        or contains_any(a, "how can i help", "happy to help you today", "i'm hrmind")
    ) and len(a) < 220
    if greetingish:
        return False, f"got greeting: {a[:160]}"
    return True, "not-greeting"


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


def expect_skill_and_place(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_not_greeting(a, r)
    if not ok:
        return False, note
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    import re

    m = re.search(r"\b(\d+)\b", a)
    n = int(m.group(1)) if m else None
    if n is not None and n >= 80:
        return False, f"suspiciously large n={n}: {a[:160]}"
    if contains_any(
        a,
        "don't recognize",
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
    return expect_python_skill(a, r)


def expect_birthday_from_resume_soft(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_not_greeting(a, r)
    if not ok:
        return False, note
    return expect_birthday_from_resume(a, r)


_MONTHS_TEXT = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def expect_birthday_from_resume(a: str, r: dict) -> tuple[bool, str]:
    """Birth date recovered from resume text (or namesakes listed with their dates)."""
    if a.strip().startswith("{"):
        return False, f"raw payload: {a[:120]}"
    if contains_any(a, "internal_error", "traceback", "don't have access"):
        return False, f"error-ish: {a[:200]}"
    if not any(m in a.lower() for m in _MONTHS_TEXT):
        return False, f"no date in answer: {a[:200]}"
    kinds = {
        str(s.get("kind", "")).lower()
        for s in (r.get("sources") or [])
        if isinstance(s, dict)
    }
    if kinds and "resume_chunk" not in kinds:
        return False, f"not resume-sourced: {sorted(kinds)}"
    ambiguous = contains_any(a, "employees match", "which one you mean")
    return True, "namesakes listed" if ambiguous else "birth date from resume"


def expect_birthday_wish(a: str, r: dict) -> tuple[bool, str]:
    ok, note = expect_birthday_from_resume(a, r)
    if not ok:
        return False, note
    if contains_any(a, "employees match", "which one you mean"):
        return True, "namesakes listed (wish deferred)"
    ok = contains_any(a, "happy birthday", "birthday wishes")
    return ok, "wish + date" if ok else f"no greeting: {a[:200]}"


def expect_birthday_today(a: str, _r: dict) -> tuple[bool, str]:
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    if contains_any(a, "no employee has a birthday today", "no birthdays today"):
        return True, "none today (honest)"
    ok = contains_any(a, "birthday") and contains_any(a, "happy birthday")
    return ok, "birthdays today" if ok else f"got: {a[:200]}"


def expect_birthday_missing_person(a: str, _r: dict) -> tuple[bool, str]:
    """An unknown name, or a resume with no date, must be refused honestly."""
    ok = contains_any(
        a, "couldn't find a date of birth", "could not find a date of birth"
    )
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


def expect_birthday_coverage_admitted(a: str, _r: dict) -> tuple[bool, str]:
    """Cohort answers must state how much of the corpus they could actually read."""
    if contains_any(a, "internal_error", "traceback"):
        return False, f"error: {a[:160]}"
    ok = contains_any(a, "read from") and contains_any(a, "no date of birth recorded")
    return ok, "coverage stated" if ok else f"no coverage caveat: {a[:200]}"


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
            "E1 where does Alice Nguyen live",
            [
                ("where does carol garcia live?", expect_person_location),
            ],
        ),
        Scenario(
            "E2 tell me about Alice Nguyen",
            [
                ("Tell me about Alice Nguyen", expect_about_person),
            ],
        ),
        Scenario(
            "E3 names then where carol lives (entity memory)",
            [
                ("List employees in Berlin", expect_city_list),
                ("where does carol live?", expect_person_location),
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
        # --- I. Status updates (resume_search resolve → employee.set_status) ---
        Scenario(
            "I1 status by name Carol Garcia",
            [
                ("change the status of Carol Garcia to true", expect_status_true),
            ],
        ),
        Scenario(
            "I2 status by name alice bauer",
            [
                ("change the status of alice bauer to true", expect_status_true),
            ],
        ),
        Scenario(
            "I3 status by location Dubai",
            [
                (
                    "update status of employees which are living in Dubai to true",
                    expect_status_location_cohort,
                ),
            ],
        ),
        Scenario(
            "I4 Dubai status true then false",
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
            "I5 status unique name Hugo Marino",
            [
                ("set Hugo Marino status to true", expect_status_true),
            ],
        ),
        # --- J. Birthdays (resume text only; never SQL) ---
        Scenario(
            "J1 birthday by name",
            [
                ("when is Carol Garcia's birthday?", expect_birthday_from_resume),
            ],
        ),
        Scenario(
            "J2 wish a happy birthday",
            [
                ("say happy birthday to Carol Garcia", expect_birthday_wish),
            ],
        ),
        Scenario(
            "J3 how old is person",
            [
                ("how old is Alice Nguyen?", expect_birthday_from_resume),
            ],
        ),
        Scenario(
            "J4 birthdays today and this month",
            [
                ("whose birthday is today?", expect_birthday_today),
                ("upcoming birthdays", expect_birthday_today),
            ],
        ),
        Scenario(
            "J5 unknown person refused",
            [
                ("when is Zzz Nobody's birthday?", expect_birthday_missing_person),
            ],
        ),
        Scenario(
            "J6 namesakes are all listed",
            [
                ("when is Priya Silva's birthday?", expect_birthday_namesakes),
            ],
        ),
        Scenario(
            "J7 resume without a date of birth",
            [
                ("when is Tomas Khan's birthday?", expect_birthday_missing_person),
            ],
        ),
        Scenario(
            "J8 misspelled name still resolves",
            [
                ("when is Carol Garciaa's birthday?", expect_birthday_typo),
            ],
        ),
        Scenario(
            "J9 cohort answer admits its coverage",
            [
                ("whose birthday is today?", expect_birthday_coverage_admitted),
            ],
        ),
        # --- K. Orchestration / hallucination-control contracts ---
        Scenario(
            "K1 elliptical after list not greeting",
            [
                ("List employees in Engineering", expect_has_names),
                ("her email?", expect_not_greeting),
            ],
        ),
        Scenario(
            "K2 ambiguous pronoun clarifies",
            [
                ("List employees in Berlin", expect_city_list),
                ("where does she live?", expect_clarify),
            ],
        ),
        Scenario(
            "K3 skill and city intersect",
            [
                ("Who knows Python in Berlin?", expect_skill_and_place),
            ],
        ),
        Scenario(
            "K4 skill and city count",
            [
                ("How many employees know React in Dubai?", expect_skill_and_place),
            ],
        ),
        Scenario(
            "K5 ordinal birthday after names",
            [
                ("How many employees work in Engineering?", expect_engineering_count),
                ("how many of them know python?", expect_followup_count_reasonable),
                ("list their names", expect_has_names),
                (
                    "give me the first persons date of birth",
                    expect_birthday_from_resume_soft,
                ),
            ],
        ),
        Scenario(
            "K6 expanded place vocab Munich",
            [
                ("List employees in Munich", expect_not_greeting),
            ],
        ),
        Scenario(
            "K7 status location confirm gate",
            [
                (
                    "update status of employees which are living in Dubai to true",
                    expect_status_location_cohort,
                ),
            ],
        ),
        Scenario(
            "K8 profile then short manager follow-up",
            [
                ("Tell me about Alice Nguyen", expect_about_person),
                ("her manager?", expect_managerish),
                ("where does she live?", expect_person_location),
            ],
        ),
        Scenario(
            "K9 natural skill phrasing",
            [
                ("Who on the team has Python experience?", expect_python_skill),
                ("any of them in Berlin?", expect_followup_count_reasonable),
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
