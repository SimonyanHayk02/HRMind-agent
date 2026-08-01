#!/usr/bin/env python3
"""Seed ~100 employees with a manager tree.

Location is not seeded here: it lives only in resume documents, assigned by
``generate_resumes.py``.
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.persistence.sqlalchemy.models import Base, EmployeeModel
from app.config.settings import get_settings

DEPARTMENTS = ["Engineering", "People", "Sales", "Finance", "Product", "Operations"]
POSITIONS = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Staff Engineer", "Engineering Manager"],
    "People": ["HR Specialist", "Recruiter", "HR Manager", "People Ops"],
    "Sales": ["Account Executive", "Sales Manager", "SDR"],
    "Finance": ["Accountant", "Financial Analyst", "Finance Manager"],
    "Product": ["Product Manager", "Product Designer", "Head of Product"],
    "Operations": ["Ops Specialist", "Ops Manager", "Coordinator"],
}
FIRST = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Hassan", "Ivy", "Jack",
    "Kara", "Leo", "Maya", "Nina", "Omar", "Paula", "Quinn", "Rita", "Sam", "Tina",
    "Ada", "Boris", "Chloe", "Diego", "Elena", "Farid", "Gita", "Hugo", "Iris", "Jonas",
    "Katya", "Lucas", "Mira", "Noor", "Oscar", "Priya", "Rafael", "Sofia", "Tomas", "Yuki",
]
LAST = [
    "Nguyen", "Smith", "Garcia", "Mueller", "Khan", "Brown", "Martin", "Silva",
    "Andersen", "Chen", "Patel", "Rossi", "Kim", "Ivanov", "Lopez",
    "Okafor", "Dubois", "Haddad", "Novak", "Fischer", "Tanaka", "Costa", "Bauer",
    "Moreau", "Sharma", "Petrov", "Yilmaz", "Berg", "Marino", "Sorensen",
]

# Most people must be findable by name alone, so names are drawn without repeats.
# A small namesake group is kept on purpose: duplicate names are a real HR problem
# and the disambiguation path needs to stay exercised.
NAMESAKE_SOURCE = 10
NAMESAKE_TARGETS = (55, 82)

# Stable ids keep resume filenames, birth dates and golden expectations reproducible
# across reseeds.
EMPLOYEE_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def _employee_id(index: int) -> uuid.UUID:
    return uuid.uuid5(EMPLOYEE_ID_NAMESPACE, f"hrmind-employee-{index}")


def _names(count: int) -> list[tuple[str, str]]:
    """Deterministic, mostly unique (first, last) pairs."""
    pairs = [(first, last) for first in FIRST for last in LAST]
    random.Random(1234).shuffle(pairs)
    chosen = pairs[:count]
    for target in NAMESAKE_TARGETS:
        if target < count:
            chosen[target] = chosen[NAMESAKE_SOURCE]
    return chosen


def _email(first: str, last: str, i: int) -> str:
    return f"{first.lower()}.{last.lower()}{i}@hrmind.example"


def _hire_date(rng: random.Random, *, senior: bool) -> date:
    """Spread hire dates through the recent past so 'this year' / 'last year' work."""
    today = date.today()
    start = date(today.year - (8 if senior else 6), 1, 1)
    end = today - timedelta(days=14)
    if end <= start:
        return start
    return start + timedelta(days=rng.randint(0, (end - start).days))


async def seed() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    tenant_id = uuid.UUID(settings.default_tenant_id)
    rng = random.Random(42)

    employees: list[EmployeeModel] = []
    managers: dict[str, EmployeeModel] = {}
    names = _names(100)
    for i, dept in enumerate(DEPARTMENTS):
        first, last = names[i]
        mgr = EmployeeModel(
            id=_employee_id(i),
            tenant_id=tenant_id,
            first_name=first,
            last_name=last,
            email=_email(first, last, i),
            department=dept,
            position=POSITIONS[dept][-1],
            salary=Decimal(rng.randint(120000, 180000)),
            hire_date=_hire_date(rng, senior=True),
            manager_id=None,
            education="MBA / MSc",
            employment_status="active",
            status=False,
        )
        managers[dept] = mgr
        employees.append(mgr)

    for i in range(len(DEPARTMENTS), 100):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        first, last = names[i]
        emp = EmployeeModel(
            id=_employee_id(i),
            tenant_id=tenant_id,
            first_name=first,
            last_name=last,
            email=_email(first, last, i),
            department=dept,
            position=rng.choice(POSITIONS[dept][:-1]),
            salary=Decimal(rng.randint(55000, 140000)),
            hire_date=_hire_date(rng, senior=False),
            manager_id=managers[dept].id,
            education=rng.choice(
                ["BSc Computer Science", "BA Business", "MSc Data Science", "Bootcamp"]
            ),
            employment_status=rng.choice(["active", "active", "active", "leave"]),
            status=False,
        )
        employees.append(emp)

    async with session_factory() as session:
        await session.execute(text("TRUNCATE resume_chunks, resumes, employees CASCADE"))
        session.add_all(employees)
        await session.commit()

    seed_path = Path("data/seed/employees.json")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": str(e.id),
            "first_name": e.first_name,
            "last_name": e.last_name,
            "email": e.email,
            "department": e.department,
            "position": e.position,
            "manager_id": str(e.manager_id) if e.manager_id else None,
            "hire_date": e.hire_date.isoformat(),
            "education": e.education,
            "status": False,
        }
        for e in employees
    ]
    seed_path.write_text(json.dumps(payload, indent=2))
    print(f"Seeded {len(employees)} employees -> {seed_path}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
