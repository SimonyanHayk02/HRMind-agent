from __future__ import annotations

from arq.connections import RedisSettings

from app.config.settings import get_settings
from app.worker.tasks.ingest_resume import ingest_resume_task


class WorkerSettings:
    functions = [ingest_resume_task]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
