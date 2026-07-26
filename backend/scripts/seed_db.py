#!/usr/bin/env python3
"""Seed ~100 employees with a manager tree."""
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
COUNTRIES = [
    ("Germany", "Berlin"),
    ("USA", "New York"),
    ("UK", "London"),
    ("UAE", "Dubai"),
    ("France", "Paris"),
]
FIRST = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Hassan", "Ivy", "Jack",
    "Kara", "Leo", "Maya", "Nina", "Omar", "Paula", "Quinn", "Rita", "Sam", "Tina",
]
LAST = [
    "Nguyen", "Smith", "Garcia", "Mueller", "Khan", "Brown", "Martin", "Silva",
    "Andersen", "Chen", "Patel", "Rossi", "Kim", "Ivanov", "Lopez",
]


def _email(first: str, last: str, i: int) -> str:
    return f"{first.lower()}.{last.lower()}{i}@hrmind.example"


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
    for i, dept in enumerate(DEPARTMENTS):
        country, city = COUNTRIES[i % len(COUNTRIES)]
        first, last = FIRST[i], LAST[i]
        mgr = EmployeeModel(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            first_name=first,
            last_name=last,
            email=_email(first, last, i),
            department=dept,
            position=POSITIONS[dept][-1],
            salary=Decimal(rng.randint(120000, 180000)),
            hire_date=date(2018, 1, 1) + timedelta(days=rng.randint(0, 1000)),
            country=country,
            city=city,
            manager_id=None,
            education="MBA / MSc",
            employment_status="active",
        )
        managers[dept] = mgr
        employees.append(mgr)

    for i in range(len(DEPARTMENTS), 100):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        country, city = COUNTRIES[i % len(COUNTRIES)]
        first, last = FIRST[i % len(FIRST)], LAST[(i * 3) % len(LAST)]
        emp = EmployeeModel(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            first_name=first,
            last_name=last,
            email=_email(first, last, i),
            department=dept,
            position=rng.choice(POSITIONS[dept][:-1]),
            salary=Decimal(rng.randint(55000, 140000)),
            hire_date=date(2019, 1, 1) + timedelta(days=rng.randint(0, 2000)),
            country=country,
            city=city,
            manager_id=managers[dept].id,
            education=rng.choice(
                ["BSc Computer Science", "BA Business", "MSc Data Science", "Bootcamp"]
            ),
            employment_status=rng.choice(["active", "active", "active", "leave"]),
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
            "country": e.country,
            "city": e.city,
            "hire_date": e.hire_date.isoformat(),
        }
        for e in employees
    ]
    seed_path.write_text(json.dumps(payload, indent=2))
    print(f"Seeded {len(employees)} employees -> {seed_path}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
