from __future__ import annotations

import re

# Topics we cannot answer from employees/resumes schema today.
# Keep salary out of this list (column ACL). Include lifestyle/personal facts
# that are neither employee columns nor resume-extracted fields.
UNSUPPORTED_TOPIC_RE = re.compile(
    r"\b("
    r"vacation|vacations|pto|time[\s-]*off|sick[\s-]*leave|leave[\s-]*balance|"
    r"annual[\s-]*leave|paid[\s-]*leave|holiday[\s-]*balance|"
    r"benefits?|health[\s-]*insurance|dental|401k|pension|"
    r"payroll|paycheck|bonus(?:es)?|commission|"
    r"stock(?:s)?|equity|rsu|options?|"
    r"visa|immigration|work[\s-]*permit|"
    r"performance[\s-]*review|disciplinary|termination|severance|"
    r"password|ssn|social[\s-]*security|bank[\s-]*account|"
    r"travel(?:l?ing)?(?:\s+preferences?)?|relocation(?:\s+preferences?)?|"
    r"willing(?:ness)?\s+to\s+travel|"
    r"hobbies|personal\s+interests|dietary|allerg(?:y|ies)|"
    r"marital\s+status|spouse|children|religion|"
    r"political\s+affiliat\w*|favorite\s+\w+"
    r")\b",
    re.I,
)

# employees.education is only a level enum (Bootcamp, BSc, …) — no start/end dates.
# Do not confuse with company tenure ("how long has he been here?").
_EDUCATION_TIMELINE_RE = re.compile(
    r"(?:"
    r"\bhow\s+long\s+(?:\w+\s+){0,4}(?:been\s+)?"
    r"(?:learn(?:ing)?|stud(?:y|ying|ied)|enrolled|"
    r"at\s+(?:school|university|college|bootcamp))\b|"
    r"\b(?:when|what\s+year)\s+(?:\w+\s+){0,6}"
    r"(?:graduat\w*|start(?:ed)?\s+(?:school|stud\w*|learn\w*|bootcamp|college|university)|"
    r"enroll\w*)\b|"
    r"\b(?:education|degree|school|university|college|bootcamp)\s+"
    r"(?:start|end|duration|dates?|years?|timeline)\b|"
    r"\b(?:years?|months?)\s+(?:of\s+)?(?:study|schooling|learning|education)\b|"
    r"\bgraduat(?:ion|ed)\s+date\b|"
    r"\bbeen\s+learn(?:ing)?\s+there\b|"
    r"\blearn(?:ing)?\s+there\b"
    r")",
    re.I,
)

UNSUPPORTED_ANSWER = (
    "I don't have that information in the HR data I can access "
    "(PTO, benefits, payroll, equity, visa, and performance reviews need "
    "dedicated HRIS sources that are not connected yet). "
    "I can help with employee headcount, names/profiles, departments, "
    "locations, hire dates, tenure by hire date, managers and direct reports, "
    "and resume facts such as skills, languages, certifications, and birthdays."
)

MISSING_DATA_ANSWER = (
    "I don't have enough information about that to answer. "
)

EDUCATION_TIMELINE_ANSWER = (
    MISSING_DATA_ANSWER
    + "I only know each employee's education level "
    "(for example Bootcamp, BSc, or MBA / MSc), not when they started studying "
    "or how long they spent in school. "
    "I can share their education level, or their hire date / tenure at the company."
)

# Nearest supported alternatives for common unsupported domains (refusal copy only).
UNSUPPORTED_ALTERNATIVES: dict[str, str] = {
    "pto": "I can list hire dates or tenure, but not leave balances.",
    "benefits": "I can share department/role profiles, but not benefits enrollment.",
    "payroll": "I can answer headcount and role questions, but not pay or bonuses.",
    "equity": "I don't have equity/RSU data; try headcount or tenure questions instead.",
    "visa": "I don't have immigration records; I can help with location or hire dates.",
    "performance": "I don't have performance reviews; try skills or certifications from resumes.",
}


def is_unsupported_education_timeline(question: str) -> bool:
    """True for study-duration / graduation-date asks we cannot answer from schema."""
    return bool(_EDUCATION_TIMELINE_RE.search(question or ""))


def unsupported_answer_for(question: str) -> str:
    """Refusal copy, optionally naming the nearest supported alternative."""
    q = (question or "").lower()
    if is_unsupported_education_timeline(q):
        return EDUCATION_TIMELINE_ANSWER
    for key, tip in UNSUPPORTED_ALTERNATIVES.items():
        if key == "pto" and re.search(
            r"\b(pto|vacation|time[\s-]*off|leave[\s-]*balance|annual[\s-]*leave)\b",
            q,
        ):
            return f"{UNSUPPORTED_ANSWER} {tip}"
        if key != "pto" and re.search(rf"\b{re.escape(key)}\b", q):
            return f"{UNSUPPORTED_ANSWER} {tip}"
    if re.search(r"\b(bonus|paycheck|commission)\b", q):
        return f"{UNSUPPORTED_ANSWER} {UNSUPPORTED_ALTERNATIVES['payroll']}"
    if re.search(r"\b(stock|rsu|options?)\b", q):
        return f"{UNSUPPORTED_ANSWER} {UNSUPPORTED_ALTERNATIVES['equity']}"
    if re.search(r"\b(immigration|work[\s-]*permit)\b", q):
        return f"{UNSUPPORTED_ANSWER} {UNSUPPORTED_ALTERNATIVES['visa']}"
    if re.search(r"\b(performance[\s-]*review|disciplinary)\b", q):
        return f"{UNSUPPORTED_ANSWER} {UNSUPPORTED_ALTERNATIVES['performance']}"
    return UNSUPPORTED_ANSWER


COUNT_QUESTION_RE = re.compile(r"\b(how many|how much|count|number of|headcount)\b", re.I)


def is_unsupported_topic(question: str) -> bool:
    q = question or ""
    return bool(UNSUPPORTED_TOPIC_RE.search(q)) or is_unsupported_education_timeline(q)


def is_count_question(question: str) -> bool:
    return bool(COUNT_QUESTION_RE.search(question or ""))
