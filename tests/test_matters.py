"""Matter CRUD, tenant isolation, idempotency and keyset pagination."""

from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.api.deps import IdempotencyGuard
from app.db.models import User
from tests.conftest import register_and_login


async def test_create_and_fetch_matter(client: AsyncClient) -> None:
    auth = await register_and_login(client)

    created = await client.post(
        "/api/v1/matters",
        headers=auth["headers"],
        json={"title": "Acme v. Globex", "client_reference": "ACME-2026-01"},
    )
    assert created.status_code == 201
    matter = created.json()
    assert matter["title"] == "Acme v. Globex"

    fetched = await client.get(f"/api/v1/matters/{matter['id']}", headers=auth["headers"])
    assert fetched.status_code == 200
    assert fetched.json()["id"] == matter["id"]


async def test_tenant_isolation(client: AsyncClient) -> None:
    """
    THE MOST IMPORTANT TEST IN A MULTI-TENANT PRODUCT.

    Firm A's matter must be invisible to Firm B - and must return 404, not 403.
    A 403 would confirm the id exists, which is itself a leak.
    """
    firm_a = await register_and_login(client, org="Firm A", email="a@firm-a.com")
    firm_b = await register_and_login(client, org="Firm B", email="b@firm-b.com")

    created = await client.post(
        "/api/v1/matters", headers=firm_a["headers"], json={"title": "Firm A confidential"}
    )
    matter_id = created.json()["id"]

    leaked = await client.get(f"/api/v1/matters/{matter_id}", headers=firm_b["headers"])
    assert leaked.status_code == 404

    listing = await client.get("/api/v1/matters", headers=firm_b["headers"])
    assert listing.json()["items"] == []


async def test_idempotency_key_prevents_duplicate_creation(client: AsyncClient) -> None:
    """A retried POST must create ONE matter, and return the original response."""
    auth = await register_and_login(client, org="Idem Firm", email="idem@test.com")
    key = str(uuid.uuid4())
    body = {"title": "Only once", "client_reference": "ONE"}

    first = await client.post(
        "/api/v1/matters", headers={**auth["headers"], "Idempotency-Key": key}, json=body
    )
    second = await client.post(
        "/api/v1/matters", headers={**auth["headers"], "Idempotency-Key": key}, json=body
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]  # same resource, not a twin
    assert second.headers.get("Idempotent-Replay") == "true"

    listing = await client.get("/api/v1/matters", headers=auth["headers"])
    assert len(listing.json()["items"]) == 1


async def test_same_key_with_different_body_is_a_conflict(client: AsyncClient) -> None:
    """Reusing a key for a DIFFERENT operation is a client bug; surface it."""
    auth = await register_and_login(client, org="Idem Conflict", email="ic@test.com")
    key = str(uuid.uuid4())

    await client.post(
        "/api/v1/matters",
        headers={**auth["headers"], "Idempotency-Key": key},
        json={"title": "First"},
    )
    mismatch = await client.post(
        "/api/v1/matters",
        headers={**auth["headers"], "Idempotency-Key": key},
        json={"title": "Completely different"},
    )
    assert mismatch.status_code == 409


async def test_cursor_pagination_walks_every_row_exactly_once(client: AsyncClient) -> None:
    """
    Page through 7 matters at 3 per page and assert we see each one once.

    This catches the two classic keyset bugs: an off-by-one at the boundary
    (a row skipped or repeated) and a missing tie-breaker in the sort key.
    """
    auth = await register_and_login(client, org="Paging Firm", email="page@test.com")

    titles = [f"Matter {i:02d}" for i in range(7)]
    for title in titles:
        response = await client.post(
            "/api/v1/matters", headers=auth["headers"], json={"title": title}
        )
        assert response.status_code == 201

    seen: list[str] = []
    cursor: str | None = None
    for _ in range(10):  # bounded loop: never risk an infinite loop in a test
        url = "/api/v1/matters?limit=3" + (f"&cursor={cursor}" if cursor else "")
        page = (await client.get(url, headers=auth["headers"])).json()
        seen.extend(item["title"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert cursor is None, "pagination did not terminate"
    assert sorted(seen) == sorted(titles)
    assert len(seen) == len(set(seen)), "a row was returned on two different pages"


async def test_malformed_cursor_is_a_422_not_a_500(client: AsyncClient) -> None:
    auth = await register_and_login(client, org="Bad Cursor", email="bc@test.com")
    response = await client.get("/api/v1/matters?cursor=!!!not-base64!!!", headers=auth["headers"])
    assert response.status_code == 422


async def test_limit_is_capped(client: AsyncClient) -> None:
    """An unbounded page size is a DoS vector against your own database."""
    auth = await register_and_login(client, org="Cap Firm", email="cap@test.com")
    response = await client.get("/api/v1/matters?limit=100000", headers=auth["headers"])
    assert response.status_code == 422


async def test_idempotency_keys_are_scoped_per_endpoint(client: AsyncClient) -> None:
    """
    The same key on a different endpoint is a different operation: it must
    neither replay the other endpoint's response nor raise a body conflict.
    """
    auth = await register_and_login(client, org="Idem Scope", email="scope@test.com")
    session_factory = client.app.state.session_factory  # type: ignore[attr-defined]

    async with session_factory() as session:
        user = await session.get(User, auth["user"]["id"])
        guard = IdempotencyGuard(session, user)
        await guard.store("k-1", "POST /a", {"x": 1}, 201, {"id": "a"})

        assert await guard.lookup("k-1", "POST /a", {"x": 1}) == (201, {"id": "a"})
        assert await guard.lookup("k-1", "POST /b", {"y": 2}) is None

        # Storing under the other endpoint does not collide with the first.
        await guard.store("k-1", "POST /b", {"y": 2}, 201, {"id": "b"})
        assert await guard.lookup("k-1", "POST /b", {"y": 2}) == (201, {"id": "b"})
