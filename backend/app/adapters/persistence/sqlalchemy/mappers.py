from __future__ import annotations

from app.adapters.persistence.sqlalchemy.models import EmployeeModel, ResumeModel
from app.domain.employee import Employee
from app.domain.enums import EmploymentStatus, ResumeStatus
from app.domain.resume import Resume


def employee_to_domain(row: EmployeeModel) -> Employee:
    return Employee(
        id=row.id,
        tenant_id=row.tenant_id,
        first_name=row.first_name,
        last_name=row.last_name,
        email=row.email,
        department=row.department,
        position=row.position,
        salary=row.salary,
        hire_date=row.hire_date,
        manager_id=row.manager_id,
        education=row.education,
        employment_status=EmploymentStatus(row.employment_status),
        status=bool(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def resume_to_domain(row: ResumeModel) -> Resume:
    return Resume(
        id=row.id,
        employee_id=row.employee_id,
        storage_path=row.storage_path,
        content_type=row.content_type,
        checksum=row.checksum,
        parse_version=row.parse_version,
        status=ResumeStatus(row.status),
        error=row.error,
        updated_at=row.updated_at,
    )
