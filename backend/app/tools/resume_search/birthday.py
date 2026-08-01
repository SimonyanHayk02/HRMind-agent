"""Birth dates live only in resume text, so they are parsed back out of retrieved chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

DOB_LABEL = "Date of Birth"

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "12 March 1991"
_DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.IGNORECASE)
# "March 12, 1991"
_MDY_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
# "1991-03-12"
_ISO_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
# "12/03/1991" — day first, matching the written "12 March 1991" style
_SLASH_RE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")


def format_birth_date(value: date) -> str:
    """Render the canonical CV spelling, e.g. ``12 March 1991``."""
    return f"{value.day} {_MONTH_NAMES[value.month - 1]} {value.year}"


def format_day_month(value: date) -> str:
    return f"{value.day} {_MONTH_NAMES[value.month - 1]}"


def month_name(month: int) -> str:
    return _MONTH_NAMES[month - 1]


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_birth_date(text: str) -> date | None:
    """Pull a date of birth out of free resume text, preferring the labelled line."""
    if not text:
        return None
    for candidate in _dob_candidates(text):
        parsed = _parse_any_date(candidate)
        if parsed is not None:
            return parsed
    return None


def _dob_candidates(text: str) -> list[str]:
    """Only labelled DOB lines — never grab bare dates from Experience prose."""
    labelled = [
        line
        for line in text.splitlines()
        if re.search(r"date\s+of\s+birth|birth\s*date|\bdob\b|born\b", line, re.IGNORECASE)
    ]
    return labelled


def _parse_any_date(text: str) -> date | None:
    m = _DMY_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            return _safe_date(int(m.group(3)), month, int(m.group(1)))
    m = _MDY_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            return _safe_date(int(m.group(3)), month, int(m.group(2)))
    m = _ISO_RE.search(text)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _SLASH_RE.search(text)
    if m:
        return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def today_utc() -> date:
    return datetime.now(UTC).date()


def age_on(born: date, today: date) -> int:
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def next_occurrence(born: date, today: date) -> date:
    """Next calendar anniversary, mapping 29 Feb to 1 March in non-leap years."""
    month, day = born.month, born.day
    for year in (today.year, today.year + 1):
        candidate = _safe_date(year, month, day) or _safe_date(year, 3, 1)
        if candidate and candidate >= today:
            return candidate
    return _safe_date(today.year + 1, month, day) or date(today.year + 1, 3, 1)


@dataclass
class BirthdayFact:
    """One person's birthday as recovered from their resume text."""

    employee_id: str
    name: str
    born: date
    chunk_id: str = ""

    def is_today(self, today: date) -> bool:
        return (self.born.month, self.born.day) == (today.month, today.day)

    def age(self, today: date) -> int:
        return age_on(self.born, today)

    def days_until(self, today: date) -> int:
        return (next_occurrence(self.born, today) - today).days

    def turning(self, today: date) -> int:
        """Age at the next anniversary."""
        return self.age(today) if self.is_today(today) else self.age(today) + 1

    def to_dict(self, today: date) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "birth_date": self.born.isoformat(),
            "birth_date_text": format_birth_date(self.born),
            "age": self.age(today),
            "is_today": self.is_today(today),
            "days_until": self.days_until(today),
            # Provenance: the resume chunk this date was read out of.
            "extracted_from_chunk_id": self.chunk_id,
        }


def facts_from_hits(hits: list[dict[str, Any]]) -> list[BirthdayFact]:
    """Parse one birthday per employee from retrieved resume chunks."""
    out: dict[str, BirthdayFact] = {}
    for hit in hits:
        eid = str(hit.get("employee_id") or "")
        if not eid or eid in out:
            continue
        born = parse_birth_date(str(hit.get("content") or ""))
        if born is None:
            continue
        meta = hit.get("metadata") or {}
        name = (
            hit.get("employee_name")
            or (meta.get("employee_name") if isinstance(meta, dict) else None)
            or "This employee"
        )
        out[eid] = BirthdayFact(
            employee_id=eid,
            name=str(name),
            born=born,
            chunk_id=str(hit.get("id") or ""),
        )
    return list(out.values())


def select_cohort(
    facts: list[BirthdayFact],
    *,
    params: dict[str, Any],
    today: date | None = None,
) -> list[BirthdayFact]:
    """Facts the cohort answer actually talks about, for citation purposes."""
    now = today or today_utc()
    scope = str(params.get("scope") or "")
    month = params.get("month")
    if scope == "closest":
        if not facts:
            return []
        best = min(f.days_until(now) for f in facts)
        return [f for f in facts if f.days_until(now) == best]
    if scope == "upcoming":
        upcoming_days = int(params.get("upcoming_days") or 30)
        return [
            f
            for f in facts
            if f.days_until(now) <= upcoming_days
        ]
    return [
        f
        for f in facts
        if f.is_today(now) or (month and f.born.month == int(month))
    ]


def _in_days(days: int) -> str:
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    return f"in {days} days"


def _people(count: int) -> str:
    return "1 employee has" if count == 1 else f"{count} employees have"


def _join_names(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def build_person_answer(
    facts: list[BirthdayFact],
    *,
    name_asked: str,
    wants_age: bool = False,
    wants_wish: bool = False,
    today: date | None = None,
    note: str | None = None,
) -> str:
    """Answer a question about one named person's birthday.

    ``note`` carries retrieval context worth admitting to the user, such as the
    question having been matched to a name only approximately.
    """
    prefix = f"{note.strip()} " if note and note.strip() else ""
    return prefix + _person_answer(
        facts,
        name_asked=name_asked,
        wants_age=wants_age,
        wants_wish=wants_wish,
        today=today,
    )


def _person_answer(
    facts: list[BirthdayFact],
    *,
    name_asked: str,
    wants_age: bool = False,
    wants_wish: bool = False,
    today: date | None = None,
) -> str:
    now = today or today_utc()
    asked = (name_asked or "").strip() or "that employee"
    if not facts:
        return (
            f"I couldn't find a date of birth in {asked}'s resume. "
            "Only details written in the resumes are available to me."
        )

    if len(facts) > 1:
        listed = "; ".join(
            f"{f.name} — {format_birth_date(f.born)} ({f.age(now)})" for f in facts[:5]
        )
        return (
            f"{len(facts)} employees match {asked}: {listed}. "
            "Tell me which one you mean."
        )

    fact = facts[0]
    born_text = format_birth_date(fact.born)
    if fact.is_today(now):
        return (
            f"It's {fact.name}'s birthday today, turning {fact.age(now)} "
            f"(born {born_text}). Happy birthday, {fact.name}!"
        )

    when = format_day_month(fact.born)
    upcoming = _in_days(fact.days_until(now))
    if wants_age:
        return f"{fact.name} is {fact.age(now)} (born {born_text})."
    if wants_wish:
        return (
            f"{fact.name}'s birthday is on {when} ({upcoming}), "
            f"turning {fact.turning(now)}. Sending early happy birthday wishes!"
        )
    return (
        f"{fact.name} was born on {born_text} and turns {fact.turning(now)} "
        f"on {when} ({upcoming})."
    )


def coverage_caveat(coverage: tuple[int, int] | None) -> str:
    """Admit corpus gaps: a cohort answer is only as complete as the resumes read.

    Retrieving every ``Personal`` chunk guarantees nothing if a resume never had a
    parseable date, so the shortfall is stated rather than implied away.
    """
    if not coverage:
        return ""
    covered, total = coverage
    if total <= 0 or covered >= total:
        return ""
    missing = total - covered
    return (
        f" Read from {covered} of {total} resumes; {missing} "
        f"{'has' if missing == 1 else 'have'} no date of birth recorded."
    )


def build_cohort_answer(
    facts: list[BirthdayFact],
    *,
    scope: str,
    month: int | None = None,
    upcoming_days: int = 30,
    today: date | None = None,
    coverage: tuple[int, int] | None = None,
    among_prior: bool = False,
) -> str:
    """Answer 'whose birthday is today', 'birthdays in July', 'upcoming birthdays'."""
    return _cohort_answer(
        facts,
        scope=scope,
        month=month,
        upcoming_days=upcoming_days,
        today=today,
        among_prior=among_prior,
    ) + coverage_caveat(coverage)


def _cohort_answer(
    facts: list[BirthdayFact],
    *,
    scope: str,
    month: int | None = None,
    upcoming_days: int = 30,
    today: date | None = None,
    among_prior: bool = False,
) -> str:
    now = today or today_utc()

    if scope == "today":
        matches = [f for f in facts if f.is_today(now)]
        if not matches:
            return "No employee has a birthday today, according to the resumes."
        listed = _join_names([f"{f.name} ({f.age(now)})" for f in matches])
        if len(matches) == 1:
            return f"It's {matches[0].name}'s birthday today, turning {matches[0].age(now)}. Happy birthday!"
        return f"{len(matches)} birthdays today: {listed}. Happy birthday to them all!"

    if scope == "closest":
        if not facts:
            if among_prior:
                return "I couldn't find dates of birth for anyone in that previous set."
            return "I couldn't find any dates of birth in the resumes."
        best = min(f.days_until(now) for f in facts)
        matches = sorted(
            (f for f in facts if f.days_until(now) == best),
            key=lambda f: f.name.lower(),
        )
        when = format_day_month(matches[0].born)
        relative = _in_days(best)
        lead = (
            "Among that group, the closest upcoming birthday is "
            if among_prior
            else "The closest upcoming birthday is "
        )
        if len(matches) == 1:
            f0 = matches[0]
            return (
                f"{lead}{f0.name}'s on {when} "
                f"({relative}), turning {f0.turning(now)}."
            )
        listed = _join_names(
            [f"{f.name} ({format_day_month(f.born)})" for f in matches[:5]]
        )
        prefix = "Among that group, " if among_prior else ""
        return (
            f"{prefix}{len(matches)} employees share the closest upcoming birthday on "
            f"{when} ({relative}): {listed}."
        )

    if scope == "month" and month:
        matches = sorted(
            (f for f in facts if f.born.month == month), key=lambda f: f.born.day
        )
        label = month_name(month)
        if not matches:
            return f"No employee has a birthday in {label}, according to the resumes."
        listed = _join_names([f"{f.name} ({format_day_month(f.born)})" for f in matches])
        return (
            f"{_people(len(matches))} a birthday in {label}: {listed}. "
            "Happy birthday to them when the day comes!"
        )

    matches = sorted(
        (f for f in facts if f.days_until(now) <= upcoming_days),
        key=lambda f: f.days_until(now),
    )
    if not matches:
        return (
            f"No birthdays are coming up in the next {upcoming_days} days, "
            "according to the resumes."
        )
    listed = _join_names(
        [
            f"{f.name} ({format_day_month(f.born)}, {_in_days(f.days_until(now))})"
            for f in matches[:10]
        ]
    )
    return (
        f"{_people(len(matches))} a birthday in the next {upcoming_days} days: "
        f"{listed}. Happy birthday to them when the day comes!"
    )
