from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

from app.domain.employee import Employee
from app.domain.enums import EmploymentStatus, ResumeStatus
from app.domain.resume import Resume
from app.domain.schema_catalog import default_employee_catalog


def test_employee_defaults_status_false() -> None:
    emp = Employee(
        id=uuid4(),
        tenant_id=uuid4(),
        first_name="Ada",
        last_name="Lovelace",
        email="ada@example.com",
        department="Engineering",
        position="Software Engineer",
        salary=Decimal("100000"),
        hire_date=date(2020, 1, 1),
        employment_status=EmploymentStatus.ACTIVE,
    )
    assert emp.status is False


def test_resume_has_no_location_fields() -> None:
    resume = Resume(
        id=uuid4(),
        employee_id=uuid4(),
        storage_path="/tmp/x.pdf",
        content_type="pdf",
        checksum="abc",
        status=ResumeStatus.PENDING,
    )
    assert "city" not in resume.model_fields
    assert "country" not in resume.model_fields


def test_catalog_marks_location_as_resume_sourced() -> None:
    catalog = default_employee_catalog()
    city = catalog.by_name("city")
    country = catalog.by_name("country")
    status = catalog.by_name("status")
    assert city is not None and city.source == "resume"
    assert country is not None and country.source == "resume"
    assert catalog.resume_sourced() == {"city", "country"}
    assert status is not None and status.filterable
    assert "true" in status.values
