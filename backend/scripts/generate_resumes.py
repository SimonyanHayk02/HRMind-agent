#!/usr/bin/env python3
"""Generate PDF/DOCX resumes for seeded employees and upsert resume rows."""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.persistence.sqlalchemy.models import EmployeeModel, ResumeModel
from app.config.settings import get_settings

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


def resume_text(emp: dict, skill: str) -> str:
    return f"""Summary
Experienced {emp['position']} in {emp['department']} based in {emp['city']}.

Experience
- {emp['position']} at Acme Corp
- Delivered projects using {skill}

Skills
{skill}, Communication, Collaboration

Projects
- Internal platform initiative involving {skill}

Languages
English

Education
{emp.get('education') or 'Bachelors'}

Certificates
Professional certificate in {skill}
"""


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


async def main() -> None:
    settings = get_settings()
    out_dir = Path(settings.resume_storage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path("data/seed/employees.json")
    if seed_path.exists():
        employees = json.loads(seed_path.read_text())
    else:
        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            rows = (await session.execute(select(EmployeeModel))).scalars().all()
            employees = [
                {
                    "id": str(r.id),
                    "first_name": r.first_name,
                    "last_name": r.last_name,
                    "department": r.department,
                    "position": r.position,
                    "city": r.city,
                    "education": r.education,
                }
                for r in rows
            ]
        await engine.dispose()

    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        for i, emp in enumerate(employees):
            skill = SKILLS[i % len(SKILLS)]
            text = resume_text(emp, skill)
            use_docx = i % 7 == 0
            filename = f"{emp['id']}.{'docx' if use_docx else 'pdf'}"
            path = out_dir / filename
            if use_docx:
                write_docx(path, text)
                content_type = "docx"
            else:
                write_pdf(path, text)
                content_type = "pdf"
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            existing = (
                await session.execute(
                    select(ResumeModel).where(ResumeModel.employee_id == uuid.UUID(emp["id"]))
                )
            ).scalar_one_or_none()
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
    print(f"Generated resumes in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
