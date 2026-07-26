.PHONY: install up down migrate seed ingest test lint run worker eval

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

migrate:
	alembic upgrade head

seed:
	python scripts/seed_db.py
	python scripts/generate_resumes.py

ingest:
	python scripts/bulk_ingest.py

test:
	pytest -q

lint:
	ruff check app tests scripts

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	arq app.worker.settings.WorkerSettings

eval:
	python scripts/run_eval.py
