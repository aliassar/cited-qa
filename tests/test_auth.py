"""
Auth tests.

Note what is asserted: not just "it works", but the SECURITY properties -
no password hash in responses, identical errors for unknown user vs wrong
password, refresh tokens rejected where access tokens belong.
"""

from __future__ import annotations

from httpx import AsyncClient

from tests.conftest import DEFAULT_PASSWORD, register_and_login


async def test_register_returns_user_without_password_hash(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "org_name": "Steinberg & Partner",
            "email": "anna@steinberg.com",
            "full_name": "Anna Steinberg",
            "password": DEFAULT_PASSWORD,
        },
    )
    assert response.status_code == 201
    body = response.json()

    assert body["email"] == "anna@steinberg.com"
    assert body["role"] == "admin"  # first user of a new org administers it
    # THE ASSERTION THAT MATTERS: the response schema is an allow-list, so a
    # hash can never leak even if someone returns the ORM object by mistake.
    assert "password_hash" not in body
    assert "password" not in body


async def test_duplicate_org_is_rejected(client: AsyncClient) -> None:
    payload = {
        "org_name": "Duplicate LLP",
        "email": "a@dup.com",
        "full_name": "A",
        "password": DEFAULT_PASSWORD,
    }
    assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 201

    second = await client.post("/api/v1/auth/register", json={**payload, "email": "b@dup.com"})
    assert second.status_code == 409
    assert second.json()["title"] == "Conflict"


async def test_email_registered_at_another_org_is_rejected(client: AsyncClient) -> None:
    """
    Login is by email alone, so an email must identify exactly one account.
    Registering a second organisation with an existing email (in any case) is a
    409, not a second account that login could not tell apart from the first.
    """
    await register_and_login(client, org="First Firm", email="shared@example.com")

    second = await client.post(
        "/api/v1/auth/register",
        json={
            "org_name": "Second Firm",
            "email": "Shared@Example.com",
            "full_name": "Same Person",
            "password": DEFAULT_PASSWORD,
        },
    )
    assert second.status_code == 409
    assert second.json()["title"] == "Conflict"

    # The original account is untouched and still logs in.
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "shared@example.com", "password": DEFAULT_PASSWORD},
    )
    assert login.status_code == 200


async def test_short_password_is_rejected_with_field_level_errors(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"org_name": "X Legal", "email": "x@x.com", "full_name": "X", "password": "short"},
    )
    assert response.status_code == 422
    # Our error envelope names the offending field, so the frontend can put the
    # message next to the right input instead of showing a toast.
    assert any(error["field"] == "password" for error in response.json()["errors"])


async def test_login_failures_are_indistinguishable(client: AsyncClient) -> None:
    """
    Unknown email and wrong password must produce the SAME response.

    A different message (or status) for each is a user-enumeration oracle.
    """
    await register_and_login(client, org="Enum Test", email="known@enum.com")

    wrong_password = await client.post(
        "/api/v1/auth/login", json={"email": "known@enum.com", "password": "not-the-password"}
    )
    unknown_user = await client.post(
        "/api/v1/auth/login", json={"email": "nobody@enum.com", "password": "not-the-password"}
    )

    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json()["detail"] == unknown_user.json()["detail"]


async def test_protected_route_requires_a_token(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer garbage"})
    ).status_code == 401


async def test_refresh_token_cannot_be_used_as_an_access_token(client: AsyncClient) -> None:
    """The `typ` claim exists exactly to stop this token-confusion attack."""
    auth = await register_and_login(client, org="Typ Test", email="typ@test.com")

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {auth['refresh_token']}"}
    )
    assert response.status_code == 401


async def test_refresh_rotates_and_revokes_the_old_token(client: AsyncClient) -> None:
    auth = await register_and_login(client, org="Rotate Test", email="rotate@test.com")

    first = await client.post("/api/v1/auth/refresh", json={"refresh_token": auth["refresh_token"]})
    assert first.status_code == 200
    assert first.json()["refresh_token"] != auth["refresh_token"]

    # Replaying the OLD refresh token must now fail - that is the whole point
    # of rotation: a stolen copy dies the moment the real client refreshes.
    replay = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": auth["refresh_token"]}
    )
    assert replay.status_code == 401


async def test_logout_revokes_the_refresh_token(client: AsyncClient) -> None:
    auth = await register_and_login(client, org="Logout Test", email="out@test.com")

    assert (
        await client.post("/api/v1/auth/logout", json={"refresh_token": auth["refresh_token"]})
    ).status_code == 204

    after = await client.post("/api/v1/auth/refresh", json={"refresh_token": auth["refresh_token"]})
    assert after.status_code == 401
