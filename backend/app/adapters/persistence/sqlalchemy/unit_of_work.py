from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.persistence.sqlalchemy.employee_repository import SqlAlchemyEmployeeRepository
from app.adapters.persistence.sqlalchemy.resume_repository import SqlAlchemyResumeRepository
from app.adapters.persistence.sqlalchemy.rls import apply_rls
from app.domain.auth import AuthContext


class SqlAlchemyUnitOfWork:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        auth: AuthContext | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._auth = auth
        self.session: AsyncSession | None = None
        self.employees: SqlAlchemyEmployeeRepository
        self.resumes: SqlAlchemyResumeRepository

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        self.session = self._session_factory()
        if self._auth is not None:
            await apply_rls(self.session, self._auth)
        self.employees = SqlAlchemyEmployeeRepository(self.session)
        self.resumes = SqlAlchemyResumeRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self.session is not None
        if exc:
            await self.session.rollback()
        await self.session.close()

    async def commit(self) -> None:
        assert self.session is not None
        await self.session.commit()

    async def rollback(self) -> None:
        assert self.session is not None
        await self.session.rollback()
