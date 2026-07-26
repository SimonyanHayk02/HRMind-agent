# HRMind — HR AI Agent Backend

Enterprise HR assistant with agentic tool orchestration (not a simple RAG chatbot).

## Architecture

Ports & Adapters layout:

- `app/api` — inbound HTTP
- `app/application` — chat, planning, execution, memory, routing
- `app/domain` — entities, policies, tools/operators contracts
- `app/ports` — interfaces
- `app/adapters` — Postgres, Redis, OpenAI, sqlglot, documents, telemetry
- `app/tools` — greeting, employee, sql, resume_search, clarify

Request path: Rule router → Embedding router → Plan compiler → Plan validator → DAG executor → Response formatter.

## Quick start

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Optional infra
docker compose up -d
alembic upgrade head
python scripts/seed_db.py
python scripts/generate_resumes.py
python scripts/bulk_ingest.py

uvicorn app.main:app --reload --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

Chat smoke:

```bash
curl -s http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Role: recruiter' \
  -H 'X-User-Id: dev' \
  -d '{"question":"hello"}'
```

Worker (optional):

```bash
arq app.worker.settings.WorkerSettings
```

## Tests

```bash
pytest -q
python scripts/run_eval.py
```

## Auth headers (dev)

- `X-User-Id`
- `X-Tenant-Id` (defaults to configured tenant)
- `X-Role` — `recruiter` | `manager` | `employee`
- `X-Department-Id` — required for manager scoping
- `X-Employee-Id` — required for employee self-scope

## Notes

- Without `OPENAI_API_KEY`, FakeLLM + FakeEmbeddings are used.
- Without Redis, in-memory cache/session store is used.
- SQL tool supports `constrained` (deterministic) and `nl2sql` (AST-validated SELECT-only).
- Cache keys are auth-scoped (`tenant + role + permission_hash`).
