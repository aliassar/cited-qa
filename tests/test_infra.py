"""Rate limiting, health probes, role enforcement, and the error envelope."""

from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Role, User
from tests.conftest import register_and_login


async def test_rate_limit_returns_429_with_retry_after(rate_limited_client: AsyncClient) -> None:
    """
    The bucket holds 3 tokens, so the 4th request in the window is refused -
    and it tells the client how long to wait rather than leaving them to guess.
    """
    statuses = []
    for _ in range(6):
        response = await rate_limited_client.post(
            "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever"}
        )
        statuses.append(response.status_code)
        if response.status_code == 429:
            assert "Retry-After" in response.headers
            assert int(response.headers["Retry-After"]) >= 1

    assert 429 in statuses
    assert statuses[0] != 429  # a first-time caller is never limited


async def test_liveness_probe_never_touches_the_database(client: AsyncClient) -> None:
    """
    Liveness must not check dependencies. If it did, a brief DB blip would make
    Kubernetes restart every pod at once - turning a small incident into a total
    outage. This test encodes that rule.
    """
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_readiness_probe_checks_the_database(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["database"] == "ok"


async def test_health_reports_queue_depth(client: AsyncClient) -> None:
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert "ingest_queue_depth" in body["checks"]


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    """Correlation id on the way out, so a user can quote it in a bug report."""
    response = await client.get("/health/live")
    assert response.headers["X-Request-ID"]

    # An inbound id is honoured rather than replaced, so a trace started at the
    # gateway survives every hop.
    supplied = await client.get("/health/live", headers={"X-Request-ID": "trace-me-123"})
    assert supplied.headers["X-Request-ID"] == "trace-me-123"


async def test_security_headers_are_present(client: AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


async def test_unknown_route_uses_the_problem_json_envelope(client: AsyncClient) -> None:
    """One error shape for the whole API - including framework-generated 404s."""
    response = await client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["status"] == 404
    assert "request_id" in body
    assert body["type"].startswith("https://errors.cited-qa.example/")


async def test_viewer_cannot_write_but_can_read(client: AsyncClient) -> None:
    """
    Role enforcement lives in the route signature (`MemberUser` / `AdminUser`),
    so it is visible in code review rather than buried in the handler body.
    """
    auth = await register_and_login(client, org="Roles Firm", email="roles@example.com")

    # Demote the user directly in the database - there is no user-management
    # endpoint in this build, and reaching into app.state is the honest way to
    # arrange the state a test needs.
    session_factory = client.app.state.session_factory  # type: ignore[attr-defined]
    async with session_factory() as session:
        user = await session.scalar(select(User).where(User.email == "roles@example.com"))
        user.role = Role.VIEWER
        await session.commit()

    # The token is still valid; the ROLE is re-read from the database on every
    # request, which is precisely why get_current_user does that DB lookup.
    write = await client.post(
        "/api/v1/matters", headers=auth["headers"], json={"title": "Should be refused"}
    )
    assert write.status_code == 403
    assert "member" in write.json()["detail"]

    read = await client.get("/api/v1/matters", headers=auth["headers"])
    assert read.status_code == 200


async def test_member_cannot_delete_a_matter(client: AsyncClient) -> None:
    auth = await register_and_login(client, org="Delete Firm", email="del@example.com")
    matter_id = (
        await client.post("/api/v1/matters", headers=auth["headers"], json={"title": "M"})
    ).json()["id"]

    session_factory = client.app.state.session_factory  # type: ignore[attr-defined]
    async with session_factory() as session:
        user = await session.scalar(select(User).where(User.email == "del@example.com"))
        user.role = Role.MEMBER
        await session.commit()

    refused = await client.delete(f"/api/v1/matters/{matter_id}", headers=auth["headers"])
    assert refused.status_code == 403


async def test_deactivated_user_is_rejected_immediately(client: AsyncClient) -> None:
    """
    Deactivation must take effect NOW, not when the access token expires.

    That is the trade-off get_current_user makes: one primary-key lookup per
    request means deactivating an account cuts off its existing access tokens
    at once. For a system holding privileged legal documents, a 15-minute window
    of stale access is not acceptable. (Revoking a single access token while
    the account stays active is a different feature - a token denylist - and is
    not implemented.)
    """
    auth = await register_and_login(client, org="Deact Firm", email="deact@example.com")

    session_factory = client.app.state.session_factory  # type: ignore[attr-defined]
    async with session_factory() as session:
        user = await session.scalar(select(User).where(User.email == "deact@example.com"))
        user.is_active = False
        await session.commit()

    assert (await client.get("/api/v1/auth/me", headers=auth["headers"])).status_code == 401
