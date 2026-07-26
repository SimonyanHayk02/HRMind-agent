#!/usr/bin/env bash
set -euo pipefail

echo "Starting HRMind (PORT=${PORT:-8000})"
python - <<'PY'
from app.config.settings import get_settings
u = get_settings().database_url
# Log host only (no credentials)
print("DB target:", u.split("@")[-1] if "@" in u else "(no @ in url)")
PY

echo "Running migrations..."
alembic upgrade head

echo "Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
