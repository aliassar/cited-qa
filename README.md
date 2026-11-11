# cited-qa

[![CI](https://github.com/aliassar/cited-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/aliassar/cited-qa/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A document Q&A API whose answers cite their sources.** Upload contracts, ingest them into
citable passages, and ask natural-language questions. Answers stream back over Server-Sent Events
with every claim tied to the passage it came from, and every question and answer is kept as an
audit trail.

It is a production-shaped FastAPI service rather than a demo: multi-tenant isolation, JWT auth with
refresh rotation, role checks, optimistic concurrency, idempotency keys, keyset pagination, rate
limiting, background ingestion, Alembic migrations and a non-root Docker image, with 54 tests that
need no infrastructure. Every module explains *why* the code looks the way it does, not just what
it does.

---

## Quick start (60 seconds, no infrastructure)

```bash
make install
cp .env.example .env
make seed
make run
```

Then open <http://localhost:8000/docs>.

```bash
make test        # 54 tests, no infrastructure
make lint        # ruff
make typecheck   # mypy
```

With Postgres instead of SQLite:

```bash
docker compose up --build
docker compose exec api python -m scripts.seed   # demo data, safe to re-run
```

---

## Try it from the shell

```bash
TOKEN=$(curl -s localhost:8000/api/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"demo@example.com","password":"demo-password-please-change"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

MATTER=$(curl -s localhost:8000/api/v1/matters \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['items'][0]['id'])")

curl -sN -X POST "localhost:8000/api/v1/matters/$MATTER/questions/stream" \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"question":"What notice period applies to terminate for convenience?"}'
```

You get a Server-Sent Event stream: `question` → `sources` → `delta`… → `done`.

---

## What is in here

| Capability | Where | Why it earns its place |
|---|---|---|
| Async SQLAlchemy 2.0 + connection pooling | `app/db/session.py` | Async only pays off if nothing blocks the event loop |
| JWT auth, access + refresh rotation | `app/core/security.py`, `app/services/auth.py` | Token type confusion, revocation, timing-safe login |
| Role-based access in the route signature | `app/api/deps.py` | Authorisation visible in code review, not buried in handlers |
| Multi-tenant isolation | every service function | `org_id` in the WHERE clause of every query |
| Cursor (keyset) pagination | `app/core/pagination.py` | OFFSET degrades and is unstable under concurrent inserts |
| Optimistic concurrency (`If-Match` / ETag) | `app/services/documents.py` | Two editors, no silent data loss |
| Idempotency keys | `app/api/deps.py`, `app/db/models.py` | Retried POSTs must not duplicate resources |
| Token-bucket rate limiting | `app/core/rate_limit.py` | Plus an honest account of why in-process is not enough |
| Background ingestion + graceful drain | `app/services/ingestion.py` | 202 Accepted, a status state machine, at-least-once safety |
| BM25 retrieval, from scratch | `app/services/retrieval.py` | The mechanics of the "R" in RAG, and where it breaks |
| LLM port + fake/real adapters | `app/services/llm.py` | Tests that never touch the network |
| SSE streaming with citations | `app/api/v1/questions.py`, `app/services/qa.py` | Perceived latency, and errors after a 200 has been sent |
| RFC 9457 error envelope | `app/core/errors.py` | One error shape for the whole API |
| Structured JSON logs + request ids | `app/core/logging.py` | ContextVar correlation across async tasks |
| Alembic migrations (up **and** down) | `alembic/` | Why `create_all` is not a deployment strategy |
| Multi-stage Docker, non-root, healthcheck | `Dockerfile` | Build tools do not belong in the runtime image |

---

## Architecture

```
HTTP  ──►  app/api/          routers + dependencies   (transport: parse, authorise, shape)
             │
             ▼
           app/services/     business logic           (no HTTP, no FastAPI imports)
             │
             ▼
           app/db/           models + session         (persistence)

           app/core/         config, security, errors, logging, pagination, rate limiting
```

Three rules the codebase follows:

1. **Handlers are thin.** Parse → call a service → shape a response. If a handler
   exceeds ~20 lines, logic has leaked into the transport layer.
2. **Services never import FastAPI.** They raise domain errors (`NotFoundError`,
   `ConflictError`); the edge maps those to status codes. That is what lets the
   seed script call the exact same functions the API does.
3. **Everything expensive is built once** in the lifespan and stashed on
   `app.state` — engine, LLM client, rate limiter, worker pool.

### Request flow, end to end

```
POST /api/v1/matters/{id}/questions/stream
  │
  ├─ middleware      request id, timing, security headers
  ├─ dependencies    rate limit → bearer token → user row → role check → DB session
  ├─ router          validate body with Pydantic
  ├─ service         persist question → load chunks → BM25 rank → build prompt
  │                    → stream from the LLM → resolve [S1] markers to citations
  │                    → persist answer (audit trail)
  └─ response        text/event-stream, event by event
```

### The data model

```
Organization ──< User
     │
     └──< Matter ──< Document ──< DocumentChunk
              └──< Question (answer + citations, kept as the audit trail)
```

`org_id` is denormalised onto every table so that **every** query can filter on
it directly — a forgotten join can then never leak another firm's documents.

---

## Deliberate limitations (and what production would do)

| Here | Production |
|---|---|
| In-process `asyncio.Queue` for ingestion | RabbitMQ / SQS / arq — durable, retried, observable, scaled separately |
| In-process rate limiter | Redis with an atomic Lua script, or an edge limiter |
| BM25 only | Hybrid: BM25 + pgvector embeddings, fused with Reciprocal Rank Fusion, then a cross-encoder reranker |
| Extracted text stored in the row | Bytes in S3, row keeps a pointer |
| Plain text and Markdown only; PDFs are rejected with 415 | pdfplumber / PyMuPDF, OCR for scans, in the worker |
| Idempotency key written after the work | Key inserted in the same transaction as the work, so the unique constraint serialises concurrent retries |
| Hard delete | Soft delete (`deleted_at`) — retention obligations and undo |
| The user is re-loaded on every request, so deactivation and role changes apply at once, but a single leaked access token cannot be revoked | Fine at a 15-minute TTL; add a Redis denylist for per-token revocation |
| A reused refresh token is rejected, but the rest of its family is not revoked | Reuse detection: revoke every token descended from the same login |
| Email is unique across organisations, so one person at two firms needs two emails | An organisation selector at login |
| Registering an existing email returns 409, which reveals that it is registered | Accept every registration and confirm by email; the endpoint is rate-limited meanwhile |

---

## Layout

```
app/
  main.py             app factory, lifespan, middleware
  core/               config, logging, errors, security, pagination, rate_limit
  db/                 base (mixins), models, session
  schemas/            Pydantic DTOs — the security allow-list for responses
  api/
    deps.py           auth, roles, pagination, idempotency, rate limiting
    router.py         versioned aggregation
    v1/               auth, matters, documents, questions, health
  services/           auth, matters, documents, ingestion, retrieval, llm, qa
alembic/              migrations
scripts/seed.py       demo data
tests/                54 tests: unit (chunking, BM25) + integration (the real ASGI app)
```

## Using the real Claude API

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... make run
```

`app/services/llm.py` swaps `FakeLLM` for `AnthropicLLM` (async client, streaming,
server-side refusal fallback). Nothing else in the codebase changes — that is the
point of the port/adapter split.

## License

[MIT](LICENSE)
