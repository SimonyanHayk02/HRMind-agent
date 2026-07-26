# HRMind Web Frontend — Backend Integration Prompt

Copy everything below this line into the web/UI agent or frontend developer brief.

---

## Mission

Build a web chat UI for **HRMind**, an HR AI agent backend. The UI must talk to the existing FastAPI API only. Do **not** reimplement agent/RAG/SQL logic in the frontend. Do **not** invent extra backend endpoints unless the backend is extended later.

**Backend base URL (local):** `http://127.0.0.1:8000`

Interactive docs (optional): `http://127.0.0.1:8000/docs`

---

## Endpoints to use

### 1) Health check

`GET /health`

**Response**
```json
{ "status": "ok" }
```

Use this for a “backend online” indicator.

---

### 2) Chat (primary product endpoint)

`POST /v1/chat`  
`Content-Type: application/json`

#### Request body
```json
{
  "question": "How many employees work in Engineering?",
  "session_id": null
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | User message |
| `session_id` | string \| null | no | Omit or `null` on first message. Reuse the returned `session_id` for follow-ups in the same conversation |

#### Required/optional auth headers (dev auth — no JWT yet)

| Header | Required | Example | Notes |
|---|---|---|---|
| `X-User-Id` | recommended | `dev` | Stable user id for the browser session |
| `X-Role` | recommended | `recruiter` | One of: `recruiter`, `manager`, `employee` |
| `X-Tenant-Id` | optional | UUID | Defaults on backend if omitted |
| `X-Department-Id` | required for manager scope | `Engineering` | Department name used for manager RBAC |
| `X-Employee-Id` | required for employee scope | employee UUID | Self-scope for `employee` role |

Defaults if headers missing: role=`recruiter`, user=`dev-user`.

#### Success response
```json
{
  "session_id": "a0342869-f4e0-46b9-91c2-bea34baf9f76",
  "answer": "The answer is 17.",
  "confidence": 1.0,
  "sources": [
    {
      "kind": "sql",
      "ref": "SELECT COUNT(*) AS count FROM employees e WHERE 1=1 AND e.department = :department",
      "label": null,
      "metadata": {}
    }
  ],
  "clarify": null,
  "trace_id": "6c406957-a003-4832-b796-7c6758917862",
  "degraded": false
}
```

| Field | UI usage |
|---|---|
| `session_id` | Persist in client state; send back on next turn |
| `answer` | Main assistant message bubble |
| `confidence` | Optional confidence badge (0–1) |
| `sources` | Optional “Sources” panel (`kind`, `ref`, `label`) |
| `clarify` | If not null, show as clarifying question / prompt user to disambiguate |
| `trace_id` | Optional debug footer |
| `degraded` | If true, show warning that some capabilities were unavailable |

#### Error response shape
```json
{ "error": "forbidden", "message": "..." }
```
Common codes: `400` validation, `403` forbidden, `404` not found, `500` internal.

---

### 3) Resume ingest (admin/internal — optional for v1 UI)

`POST /v1/ingest/resumes/{employee_id}`

Usually **not** part of the employee chat UI. Skip unless building an admin panel.

---

## Frontend behavior requirements

1. **Single chat composer** posting to `POST /v1/chat`.
2. **Session continuity:** store `session_id` from first response; include it on subsequent requests.
3. **Role selector (dev):** allow choosing `recruiter` | `manager` | `employee` and set `X-Role` accordingly.
   - If `manager`: also collect/send `X-Department-Id` (e.g. `Engineering`).
   - If `employee`: also collect/send `X-Employee-Id` (UUID from seeded employees if needed).
4. **Render `answer` as markdown-friendly plain text** (backend returns plain text today).
5. If `clarify` is present, highlight it and wait for the user’s clarifying reply as the next `question`.
6. If `degraded === true`, show a non-blocking warning.
7. Handle network/backend down using `/health` and failed fetch states.
8. CORS: if the web app runs on another origin (e.g. `localhost:3000`), backend may need CORS enabled — ask backend to add it if browser blocks requests. For local same-origin proxy, proxy `/v1/*` and `/health` to `http://127.0.0.1:8000`.

---

## Example fetch (browser)

```ts
type ChatResponse = {
  session_id: string;
  answer: string;
  confidence: number;
  sources: Array<{ kind: string; ref: string; label: string | null; metadata?: Record<string, unknown> }>;
  clarify: string | null;
  trace_id: string | null;
  degraded: boolean;
};

async function askHrAgent(params: {
  baseUrl: string;
  question: string;
  sessionId?: string | null;
  role?: "recruiter" | "manager" | "employee";
  userId?: string;
  departmentId?: string;
  employeeId?: string;
}): Promise<ChatResponse> {
  const res = await fetch(`${params.baseUrl}/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": params.userId ?? "web-user",
      "X-Role": params.role ?? "recruiter",
      ...(params.departmentId ? { "X-Department-Id": params.departmentId } : {}),
      ...(params.employeeId ? { "X-Employee-Id": params.employeeId } : {}),
    },
    body: JSON.stringify({
      question: params.question,
      session_id: params.sessionId ?? null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message ?? `HTTP ${res.status}`);
  }
  return res.json();
}
```

---

## Curl smoke tests the UI should match

```bash
curl -s http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Role: recruiter' \
  -H 'X-User-Id: web-user' \
  -d '{"question":"hello"}'

curl -s http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Role: recruiter' \
  -H 'X-User-Id: web-user' \
  -d '{"question":"How many employees work in Engineering?","session_id":"<paste-session-id>"}'
```

---

## Out of scope for the web app

- Direct Postgres/Redis/OpenAI access
- Building SQL or vector search in the browser
- JWT login (headers above are the current auth contract)
- Streaming tokens (API returns a full JSON response today)

---

## UX guidance (product)

- First viewport: brand **HRMind**, one chat input, one send action.
- Show role as a simple control (not a dashboard of widgets).
- Display assistant answer + optional sources/clarify/degraded states.
- Keep the UI a conversation, not an admin console.
