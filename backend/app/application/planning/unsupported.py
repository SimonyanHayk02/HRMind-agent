from __future__ import annotations

import re

# Topics we cannot answer from employees/resumes schema today.
UNSUPPORTED_TOPIC_RE = re.compile(
    r"\b("
    r"vacation|vacations|pto|time[\s-]*off|sick[\s-]*leave|leave[\s-]*balance|"
    r"annual[\s-]*leave|paid[\s-]*leave|holiday[\s-]*balance|"
    r"benefits?|health[\s-]*insurance|dental|401k|pension|"
    r"payroll|paycheck|bonus(?:es)?|commission|"
    r"stock(?:s)?|equity|rsu|options?|"
    r"visa|immigration|work[\s-]*permit|"
    r"performance[\s-]*review|disciplinary|termination|severance|"
    r"password|ssn|social[\s-]*security|bank[\s-]*account"
    r")\b",
    re.I,
)

UNSUPPORTED_ANSWER = (
    "I don't have that information in the HR data I can access. "
    "I can help with employee headcount, names/profiles, departments, "
    "locations, hire dates, managers, and skills or experience from resumes."
)

COUNT_QUESTION_RE = re.compile(r"\b(how many|how much|count|number of|headcount)\b", re.I)


def is_unsupported_topic(question: str) -> bool:
    return bool(UNSUPPORTED_TOPIC_RE.search(question or ""))


def is_count_question(question: str) -> bool:
    return bool(COUNT_QUESTION_RE.search(question or ""))
