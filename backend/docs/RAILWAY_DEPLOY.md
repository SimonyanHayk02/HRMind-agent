# HRMind — Railway Production Deploy Guide

## Important: pgvector

Railway’s default Postgres plugin often **does not** include `pgvector`.
HRMind needs `CREATE EXTENSION vector`.

**Recommended options**
1. **Neon** or **Supabase** Postgres with pgvector enabled (easiest)
2. Or deploy a custom Postgres image `pgvector/pgvector:pg16` as a Railway service

This guide uses **Neon (pgvector) + Railway Redis + Railway web service**.

---

## 1) Prep the repo

From your machine:

```bash
cd /Users/macbook/Desktop/HRMind
git init   # if not already
git add backend
git commit -m "HRMind backend ready for Railway"
# push to GitHub
```

Railway will deploy from the `backend/` folder (set Root Directory = `backend`).

Deploy files already added:
- `backend/Dockerfile` (uses `$PORT`)
- `backend/railway.toml` (migrate + start + healthcheck)
- CORS via `CORS_ORIGINS`

---

## 2) Create Railway project

1. Go to [https://railway.app](https://railway.app) → **New Project**
2. **Deploy from GitHub** → select this repo
3. Set **Root Directory** to `backend`
4. Builder should detect Dockerfile

### Auto-deploy after CI (required)

Deploys are triggered by GitHub Actions **after tests pass on `main`** (see `.github/workflows/ci.yml`).

1. In Railway → your **project** → **Settings** → **Tokens** → create a **Project Token** (scoped to the production environment).
2. In GitHub → repo **Settings** → **Secrets and variables** → **Actions**, add:
   - `RAILWAY_TOKEN` — the project token (required)
   - `RAILWAY_SERVICE_ID` — the **API/web** service name or ID (**required**; do not use Redis). Railway → click the API service → **Settings** → copy **Service ID**, or paste the exact service name shown in the dashboard
3. To avoid **double deploys**, in the Railway web service → **Settings** → **Source**:
   - turn **off** “Wait for CI” if you use Actions deploy, **or**
   - disconnect / disable automatic deploys on push and let Actions be the only deployer

Push to `main` → `test` job → `deploy` job → Railway build.

The deploy job uploads the **repo root** (so `backend/` exists in the snapshot). Keep Railway service **Root Directory** set to `backend`.

---

## 3) Add Redis on Railway

1. In the same project: **New** → **Database** → **Redis**
2. Open Redis → **Variables**
3. Copy `REDIS_URL` (or `REDIS_PRIVATE_URL`)

---

## 4) Add Postgres with pgvector (Neon example)

1. Create a Neon project: [https://neon.tech](https://neon.tech)
2. Enable extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Copy the connection string (prefer pooled URL).
4. Convert for async SQLAlchemy:

```text
postgresql://USER:PASS@HOST/DB?sslmode=require
→
postgresql+asyncpg://USER:PASS@HOST/DB?ssl=require
```

Notes:
- You can paste either form; the app auto-converts `postgresql://` → `postgresql+asyncpg://` and `sslmode=` → `ssl=`
- URL-encode special characters in password (`@` → `%40`, `#` → `%23`, etc.)
- Prefer Neon **pooled** connection string for the web service

### Healthcheck failure (common)

Railway shows **Network › Healthcheck** failure when the process never answers `GET /health` in time.

Usual causes:
1. **`DATABASE_URL` missing / wrong** on the **web service** (not Redis) → `alembic upgrade` exits → uvicorn never starts
2. **pgvector not enabled** on Neon → migration fails
3. **Redis URL missing** (app can fall back, but still set `REDIS_URL`)
4. Timeout too short (repo now uses 120s)

**What to do:** open the failed deploy → **Deploy Logs** (not just Build). Look for `alembic` / `asyncpg` / `ssl` / `password authentication` errors.

Quick checks in Neon SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

---

## 5) Set Railway environment variables

In the **web service** → Variables:

| Variable | Value |
|---|---|
| `APP_ENV` | `production` |
| `REQUIRE_REDIS` | `true` (optional; default required when `APP_ENV=production`) |
| `DATABASE_URL` | `postgresql+asyncpg://...` (from step 4) |
| `REDIS_URL` | from Railway Redis (**required** in production for chat context) |
| `OPENAI_API_KEY` | your key |
| `CHAT_MODEL` | `gpt-4o-mini` (or your choice) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `EMBEDDING_DIMS` | `1536` |
| `DEFAULT_TENANT_ID` | your tenant UUID |
| `CORS_ORIGINS` | your frontend origin(s), comma-separated, e.g. `https://your-app.vercel.app` |
| `LOG_LEVEL` | `INFO` |

Optional:
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

Railway also injects `PORT` automatically.

`railway.toml` start command already runs:

```bash
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## 6) Generate a public domain

Web service → **Settings** → **Networking** → **Generate Domain**

Example: `https://hrmind-api-production.up.railway.app`

Smoke:

```bash
curl https://YOUR-DOMAIN/health
```

Expect: `{"status":"ok"}`

---

## 7) Seed production data (optional)

Only if you want demo employees/resumes in prod.

From local (pointing at prod DB — careful):

```bash
cd backend
source .venv/bin/activate
export DATABASE_URL='postgresql+asyncpg://...'   # prod
export OPENAI_API_KEY='...'
export REDIS_URL='...'                           # optional for ingest script
alembic upgrade head
python scripts/seed_db.py
python scripts/generate_resumes.py
python scripts/bulk_ingest.py
```

For real production, replace seed scripts with your HR data import + resume ingest pipeline.

---

## 8) Connect the web frontend

Point the web app to:

```text
VITE_API_BASE_URL=https://YOUR-DOMAIN
# or NEXT_PUBLIC_API_BASE_URL=...
```

Use the same chat contract as `docs/WEB_FRONTEND_PROMPT.md`:
- `POST /v1/chat`
- headers `X-Role`, `X-User-Id`, etc.

Set `CORS_ORIGINS` to the exact frontend URL.

---

## 9) Post-deploy checklist

```bash
curl https://YOUR-DOMAIN/health

curl -s https://YOUR-DOMAIN/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Role: recruiter' \
  -H 'X-User-Id: prod-test' \
  -d '{"question":"hello"}'
```

- [ ] `/health` ok  
- [ ] greeting works  
- [ ] SQL question works (after seed/import)  
- [ ] resume search works (after ingest)  
- [ ] frontend can call API without CORS errors  
- [ ] OpenAI billing/quota healthy  

---

## 10) Optional: Redis + worker later

ARQ worker is optional for v1. If you add it on Railway:
- duplicate service from same repo/image
- start command: `arq app.worker.settings.WorkerSettings`
- same env vars as API

---

## Common failures

| Problem | Fix |
|---|---|
| `extension "vector" does not exist` | Use Neon/Supabase/pgvector image, run `CREATE EXTENSION vector` |
| `role does not exist` / wrong DB | Check `DATABASE_URL` points to the pgvector DB, not a random plugin |
| App sleeps / cold start | Keep-alive or Railway Pro; hit `/health` |
| CORS blocked | Set `CORS_ORIGINS` to frontend origin |
| 429 OpenAI | Billing/quota on the production key |
| Alembic fails on deploy | Ensure `alembic/` + `alembic.ini` are in the Docker image (already in Dockerfile) |

---

## Minimal architecture on Railway

```text
Browser / Web UI
      │
      ▼
Railway Web Service (FastAPI / Dockerfile)
      │
      ├── Neon/Supabase Postgres + pgvector
      ├── Railway Redis
      └── OpenAI API
```
