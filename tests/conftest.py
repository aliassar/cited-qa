"""
Shared test fixtures.

THE STRATEGY
------------
These are INTEGRATION tests: they drive the real ASGI app, the real routing,
the real dependency graph and a real SQLite database (a temp file per test; see
the `settings` fixture for why not in-memory). The only thing faked is the
LLM - because it is slow, costs money, and is non-deterministic, which are the
three properties a test must not have.

Everything else being real is the point. A test suite full of mocked
repositories passes happily while the actual SQL is broken.

WHY httpx.ASGITransport RATHER THAN A LIVE SERVER
--------------------------------------------------
ASGITransport calls the app in-process: no socket, no port, no server to start
or wait for. Tests are fast and there is nothing to leak between them. We use
the async client (not TestClient) so the test runs on the SAME event loop as the
app, which is what lets us await the background ingestion worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app

# Every `async def test_...` is run by pytest-asyncio without needing a marker,
# because pytest.ini sets `asyncio_mode = auto`.


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        # A temp FILE, not ":memory:". In-memory SQLite pins every session to a
        # single shared connection, so the background ingestion worker and the
        # polling request end up inside one another's transactions (see the long
        # comment in app/db/session.py). A file gives each session its own
        # connection and real isolation - which is what production has.
        # pytest's tmp_path is unique per test, so there is nothing to clean up.
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        jwt_secret="test-secret-not-used-anywhere-real",
        # Off by default so unrelated tests are not throttled; the rate-limit
        # test builds its own app with it enabled.
        rate_limit_enabled=False,
        llm_provider="fake",
        # Fewer rounds = far faster tests. NEVER lower this in production: the
        # cost IS the security property.
        bcrypt_rounds=4,
        ingest_worker_count=1,
    )


@pytest_asyncio.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """A client bound to a freshly built app, with the lifespan actually run."""
    app = create_app(settings)

    # Running the lifespan context is what creates the engine, the schema, the
    # rate limiter and the ingestion workers. Skip it and app.state is empty.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Handy for tests that need to reach inside (e.g. the session factory).
            ac.app = app  # type: ignore[attr-defined]
            yield ac


@pytest_asyncio.fixture
async def rate_limited_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """Same app, but with a deliberately tiny rate limit."""
    limited = settings.model_copy(
        update={
            "rate_limit_enabled": True,
            "rate_limit_requests": 3,
            "rate_limit_window_seconds": 60,
        }
    )
    app = create_app(limited)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Helpers used across tests
# ---------------------------------------------------------------------------

DEFAULT_PASSWORD = "correct-horse-battery-staple"


async def register_and_login(
    client: AsyncClient, *, org: str = "Acme Legal", email: str = "lawyer@acme.com"
) -> dict:
    """Create an org + admin user and return their auth artefacts."""
    register = await client.post(
        "/api/v1/auth/register",
        json={
            "org_name": org,
            "email": email,
            "full_name": "Test Lawyer",
            "password": DEFAULT_PASSWORD,
        },
    )
    assert register.status_code == 201, register.text
    user = register.json()

    login = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": DEFAULT_PASSWORD}
    )
    assert login.status_code == 200, login.text
    tokens = login.json()

    return {
        "user": user,
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        # Ready to splat into a request: client.get(url, headers=auth["headers"])
        "headers": {"Authorization": f"Bearer {tokens['access_token']}"},
    }


SAMPLE_CONTRACT = """\
1. Term and Termination

This Agreement commences on 1 January 2026 and continues for an initial term of
twenty-four months. Either party may terminate this Agreement for convenience by
giving ninety (90) days written notice to the other party.

2. Confidentiality

Each party shall keep confidential all Confidential Information disclosed by the
other party and shall not disclose it to any third party without prior written
consent, except to professional advisers bound by equivalent obligations.

3. Limitation of Liability

Neither party's aggregate liability under this Agreement shall exceed the total
fees paid in the twelve months preceding the claim, save for liability which
cannot lawfully be excluded.

4. Governing Law

This Agreement is governed by the laws of the Federal Republic of Germany and
the courts of Berlin have exclusive jurisdiction.
"""


async def upload_document(
    client: AsyncClient,
    headers: dict,
    matter_id: str,
    *,
    name: str = "nda.txt",
    body: str | None = None,
) -> dict:
    """Upload a text document and wait until ingestion has finished."""
    response = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=headers,
        files={"file": (name, (body or SAMPLE_CONTRACT).encode(), "text/plain")},
        data={"title": name},
    )
    assert response.status_code in (200, 202), response.text
    document = response.json()

    if response.status_code == 202:
        document = await wait_for_status(client, headers, document["id"])
    return document


async def wait_for_status(
    client: AsyncClient,
    headers: dict,
    document_id: str,
    target: str = "ready",
    timeout_seconds: float = 5.0,  # not named "timeout": ruff's ASYNC109 flags that
) -> dict:
    """
    Poll the document until ingestion reaches a terminal state.

    Polling rather than sleeping a fixed time: a fixed sleep is either flaky
    (too short on a loaded CI box) or slow (too long everywhere). `sleep(0.01)`
    also yields control to the event loop, which is what lets the in-process
    worker actually run.
    """
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/api/v1/documents/{document_id}", headers=headers)
        assert response.status_code == 200, response.text
        document = response.json()
        if document["status"] in (target, "failed"):
            return document
        await asyncio.sleep(0.01)
    raise AssertionError(f"Document {document_id} never reached status '{target}'")
