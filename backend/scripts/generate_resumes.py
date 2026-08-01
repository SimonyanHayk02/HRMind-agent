#!/usr/bin/env python3
"""Generate PDF/DOCX resumes for seeded employees and upsert resume rows.

Location and birth date are assigned here and written into the document text —
the only place the application reads them from. There are no city/country
columns on resumes or employees.
"""
from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import uuid
from datetime import date, timedelta
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.documents.docx_parser import parse_docx
from app.adapters.documents.pdf_parser import parse_pdf
from app.adapters.documents.section_splitter import split_sections
from app.adapters.persistence.sqlalchemy.models import EmployeeModel, ResumeModel
from app.config.settings import get_settings
from app.domain.places import PLACE_PAIRS
from app.tools.resume_search.birthday import (
    DOB_LABEL,
    format_birth_date,
    parse_birth_date,
    today_utc,
)
from app.tools.resume_search.certifications import parse_certifications
from app.tools.resume_search.languages import parse_languages
from app.tools.resume_search.location import Place, parse_location

SEED_PATH = Path("data/seed/employees.json")

SKILLS = [
    "Python",
    "Java",
    "Go",
    "SQL",
    "Kubernetes",
    "React",
    "Machine Learning",
    "NLP",
    "AWS",
    "Docker",
    "Recruiting",
    "Salesforce",
    "Accounting",
    "Product Strategy",
]

# Closed-vocab language packs written as labelled Languages blocks.
LANGUAGE_PACKS = [
    ["English"],
    ["English", "German"],
    ["English", "French"],
    ["English", "Arabic"],
    ["English", "Spanish"],
    ["English", "Hindi"],
    ["English", "Mandarin"],
    ["English", "German", "French"],
    ["English", "Portuguese"],
    ["English", "Dutch"],
]
NO_LANGUAGES_INDEX = 6

# Named certifications mixed into the Certificates section (plus skill cert).
NAMED_CERTS = [
    "AWS Certified",
    "CKA",
    "PMP",
    "CISSP",
    "Certified Scrum Master",
    "AWS Certified Solutions Architect",
    "Google Cloud Professional",
    "Azure Administrator",
    "CPA",
    "SHRM-CP",
]
NO_CERTS_INDEX = 8


# One resume is deliberately generated without a date of birth. Cohort answers
# claim to read the whole corpus, so the corpus needs a real gap to prove that the
# coverage report and the "not in the resume" refusal are honest rather than
# theoretical.
NO_DOB_INDEX = 2
# One resume with an unparseable location exercises the same path for place.
NO_LOCATION_INDEX = 4
# 29 February has no anniversary in most years, so one employee is born on it.
LEAP_DAY_INDEX = 3
# Indexes whose date is dictated here rather than recovered from the existing file.
PINNED_INDEXES = frozenset({0, 1, LEAP_DAY_INDEX})


def derive_birth_date(employee_id: str, *, index: int = -1) -> date:
    """Deterministic birth date; the resume document is the only place it is stored."""
    digest = hashlib.md5(employee_id.encode()).hexdigest()
    h = int(digest[:8], 16)
    today = today_utc()
    age = 24 + (h % 35)
    if index in {0, 1}:
        return date(today.year - age, today.month, today.day)
    if index == LEAP_DAY_INDEX:
        year = today.year - age
        while not calendar.isleap(year):
            year -= 1
        return date(year, 2, 29)
    return date(today.year - age, 1, 1) + timedelta(days=(h // 97) % 365)


def birth_date_for(index: int, employee_id: str, path: Path) -> date | None:
    """The date to write into this resume."""
    if index == NO_DOB_INDEX:
        return None
    if index in PINNED_INDEXES:
        return derive_birth_date(employee_id, index=index)
    return existing_birth_date(path) or derive_birth_date(employee_id, index=index)


def existing_birth_date(path: Path) -> date | None:
    """Recover the date already written into a resume so regeneration keeps it stable."""
    if not path.exists():
        return None
    try:
        text = parse_docx(path) if path.suffix.lower() == ".docx" else parse_pdf(path)
    except Exception:  # noqa: BLE001 - unreadable file falls back to a derived date
        return None
    sections = split_sections(text)
    return parse_birth_date(sections.get("Personal") or text)


def derive_place(employee_id: str, *, index: int) -> Place:
    """Deterministic place from the employee id, round-robin over the catalog."""
    country, city = PLACE_PAIRS[index % len(PLACE_PAIRS)]
    return Place(city=city, country=country)


def place_for(index: int, employee_id: str, path: Path) -> Place | None:
    """The place to write into this resume.

    ``None`` means write an unparseable location so coverage and refusal stay real.
    """
    if index == NO_LOCATION_INDEX:
        return None
    return existing_place(path) or derive_place(employee_id, index=index)


def existing_place(path: Path) -> Place | None:
    if not path.exists():
        return None
    try:
        text = parse_docx(path) if path.suffix.lower() == ".docx" else parse_pdf(path)
    except Exception:  # noqa: BLE001
        return None
    sections = split_sections(text)
    return parse_location(sections.get("Location") or text)


def certifications_for(index: int, skill: str, path: Path) -> list[str] | None:
    if index == NO_CERTS_INDEX:
        return None
    named = NAMED_CERTS[index % len(NAMED_CERTS)]
    pack = [named, f"Professional certificate in {skill}"]
    existing = existing_certifications(path)
    # Keep a prior named cert if present; otherwise rewrite from the pack so
    # closed-vocab cert asks (AWS Certified, CKA, …) stay answerable.
    if existing and any(
        c for c in existing if not c.lower().startswith("professional certificate in")
    ):
        return existing
    return pack


def existing_certifications(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        text = parse_docx(path) if path.suffix.lower() == ".docx" else parse_pdf(path)
    except Exception:  # noqa: BLE001
        return None
    sections = split_sections(text)
    return parse_certifications(sections.get("Certificates") or text)


def languages_for(index: int, path: Path) -> list[str] | None:
    """Languages block for this resume; None means omit the section (coverage gap).

    Prefer the deterministic pack so corpus coverage for German/French/… stays
    real. Only reuse an existing parse when it already includes a non-English
    language (stable multi-lingual resumes across regenerations).
    """
    if index == NO_LANGUAGES_INDEX:
        return None
    pack = list(LANGUAGE_PACKS[index % len(LANGUAGE_PACKS)])
    existing = existing_languages(path)
    if existing and any(lang != "English" for lang in existing):
        return existing
    return pack


def existing_languages(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        text = parse_docx(path) if path.suffix.lower() == ".docx" else parse_pdf(path)
    except Exception:  # noqa: BLE001
        return None
    sections = split_sections(text)
    return parse_languages(sections.get("Languages") or text)


def resume_text(
    emp: dict,
    skill: str,
    born: date | None,
    place: Place | None,
    *,
    langs: list[str] | None = None,
    certs: list[str] | None = None,
) -> str:
    if place is None:
        city, country = "Remote", "Elsewhere"
    else:
        city, country = place.city or "Unknown", place.country or "Unknown"
    personal = (
        f"""Personal
{DOB_LABEL}: {format_birth_date(born)}

"""
        if born is not None
        else ""
    )
    if langs is None:
        langs = ["English"]
    lang_block = (
        f"""Languages
{', '.join(langs)}

"""
        if langs
        else ""
    )
    if certs is None:
        certs = [f"Professional certificate in {skill}"]
    cert_block = (
        f"""Certificates
{chr(10).join(f'- {c}' for c in certs)}
"""
        if certs
        else ""
    )
    return f"""Summary
Experienced {emp['position']} in {emp['department']} based in {city}, {country}.

Location
{city}, {country}

{personal}Experience
- {emp['position']} at Acme Corp
- Delivered projects using {skill}

Skills
{skill}, Communication, Collaboration

Projects
- Internal platform initiative involving {skill}

{lang_block}Education
{emp.get('education') or 'Bachelors'}

{cert_block}"""


def write_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    y = 750
    for line in text.splitlines():
        c.drawString(40, y, line[:100])
        y -= 14
        if y < 40:
            c.showPage()
            y = 750
    c.save()


def write_docx(path: Path, text: str) -> None:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(str(path))


def existing_skill(path: Path) -> str | None:
    """Recover the skill already written into a resume so regeneration keeps it."""
    if not path.exists():
        return None
    try:
        text = parse_docx(path) if path.suffix.lower() == ".docx" else parse_pdf(path)
    except Exception:  # noqa: BLE001
        return None
    first = split_sections(text).get("Skills", "").split(",")[0].strip()
    return first or None


def prune_orphans(out_dir: Path, keep: set[str]) -> list[str]:
    """Drop resume files whose employee no longer exists."""
    removed: list[str] = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.stem not in keep:
            path.unlink()
            removed.append(path.name)
    return removed


async def load_employees(session) -> list[dict]:
    """Build the employee list from the database, which owns the authoritative ids."""
    rows = (await session.execute(select(EmployeeModel))).scalars().all()
    employees: list[dict] = []
    for emp in rows:
        employees.append(
            {
                "id": str(emp.id),
                "first_name": emp.first_name,
                "last_name": emp.last_name,
                "email": emp.email,
                "department": emp.department,
                "position": emp.position,
                "manager_id": str(emp.manager_id) if emp.manager_id else None,
                "hire_date": emp.hire_date.isoformat(),
                "education": emp.education,
                "status": emp.status,
            }
        )
    return employees


async def main() -> None:
    settings = get_settings()
    out_dir = Path(settings.resume_storage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        employees = await load_employees(session)
        existing_resumes = {
            str(r.employee_id): r
            for r in (await session.execute(select(ResumeModel))).scalars().all()
        }

        for i, emp in enumerate(employees):
            existing = existing_resumes.get(emp["id"])
            # Keep the original container so regeneration never orphans the previous file.
            content_type = existing.content_type if existing else ("docx" if i % 7 == 0 else "pdf")
            path = Path(existing.storage_path) if existing else out_dir / f"{emp['id']}.{content_type}"
            skill = existing_skill(path) or SKILLS[i % len(SKILLS)]
            born = birth_date_for(i, emp["id"], path)
            place = place_for(i, emp["id"], path)
            langs = languages_for(i, path)
            certs = certifications_for(i, skill, path)

            text = resume_text(emp, skill, born, place, langs=langs, certs=certs)
            if content_type == "docx":
                write_docx(path, text)
            else:
                write_pdf(path, text)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()

            if existing is None:
                session.add(
                    ResumeModel(
                        id=uuid.uuid4(),
                        employee_id=uuid.UUID(emp["id"]),
                        storage_path=str(path),
                        content_type=content_type,
                        checksum=checksum,
                        parse_version=settings.parse_version,
                        status="pending",
                    )
                )
            else:
                existing.storage_path = str(path)
                existing.content_type = content_type
                existing.checksum = checksum
                existing.status = "pending"
        await session.commit()

    await engine.dispose()

    orphans = prune_orphans(out_dir, {emp["id"] for emp in employees})
    SEED_PATH.write_text(json.dumps(employees, indent=2))
    print(f"Generated {len(employees)} resumes in {out_dir}; refreshed {SEED_PATH}")
    if orphans:
        print(f"Removed {len(orphans)} orphaned resume file(s)")


if __name__ == "__main__":
    asyncio.run(main())
