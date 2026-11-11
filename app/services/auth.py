"""
Authentication service.

The service layer owns BUSINESS rules; it knows nothing about HTTP. It raises
domain errors (AuthenticationError, ConflictError) which the edge translates to
status codes. That separation is what lets the same function be called from an
endpoint, a CLI, or a background job.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import anyio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AuthenticationError, ConflictError
from app.core.security import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.models import Organization, RefreshToken, Role, User


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "org"


async def _hash_in_thread(password: str, rounds: int) -> str:
    """
    bcrypt is CPU-BOUND and takes ~250ms at 12 rounds.

    Running it directly in an `async def` handler blocks the event loop for
    those 250ms - every other request on this worker freezes. `to_thread.run_sync`
    hands it to a worker thread so the loop keeps serving.

    (The GIL is released inside bcrypt's C code, so this genuinely parallelises.
    For pure-Python CPU work you would need a process pool instead.)
    """
    return await anyio.to_thread.run_sync(lambda: hash_password(password, rounds))


async def _verify_in_thread(password: str, hashed: str) -> bool:
    return await anyio.to_thread.run_sync(lambda: verify_password(password, hashed))


async def register_organization(
    session: AsyncSession,
    settings: Settings,
    *,
    org_name: str,
    email: str,
    full_name: str,
    password: str,
) -> User:
    """
    Create an organisation and its first (admin) user.

    Emails are globally unique (see db/models.py::User), so registering a new
    organisation with an email that already belongs to an account - in any
    organisation - is a 409. This does reveal that the address is registered;
    that is the usual trade-off for a sign-up form, and the rate limit on this
    endpoint keeps it from being a cheap enumeration oracle.
    """
    email = email.lower().strip()  # normalise: emails are case-insensitive in practice

    slug = _slugify(org_name)
    existing = await session.scalar(select(Organization).where(Organization.slug == slug))
    if existing is not None:
        raise ConflictError(f"An organisation named '{org_name}' already exists")

    if await session.scalar(select(User.id).where(User.email == email)) is not None:
        raise ConflictError("An account with this email already exists")

    # Hash before touching the database, so no transaction is held open
    # during the ~250ms of deliberate bcrypt work.
    password_hash = await _hash_in_thread(password, settings.bcrypt_rounds)

    org = Organization(name=org_name, slug=slug)
    session.add(org)
    try:
        # flush() sends the INSERT so `org.id` is populated, WITHOUT committing.
        # The whole registration stays one atomic transaction: if the user
        # insert fails, the organisation is rolled back too - no half-created
        # tenants.
        await session.flush()

        user = User(
            org_id=org.id,
            email=email,
            full_name=full_name,
            password_hash=password_hash,
            role=Role.ADMIN,  # first user of a new org administers it
        )
        session.add(user)
        await session.commit()
    except IntegrityError:
        # The checks above are a fast path with a clear message; the unique
        # constraints on organizations.slug and users.email are what actually
        # stop two concurrent registrations from both succeeding.
        await session.rollback()
        raise ConflictError("That organisation or email is already registered") from None
    return user


async def authenticate(
    session: AsyncSession, settings: Settings, *, email: str, password: str
) -> User:
    """
    Verify credentials.

    TWO SECURITY DETAILS:

    1. The error message is identical for "no such user" and "wrong password".
       Distinct messages are a USER ENUMERATION oracle: an attacker can harvest
       which addresses hold accounts, which is valuable on its own and makes
       credential stuffing far more efficient.

    2. When the user does not exist we still run a bcrypt verify against a dummy
       hash. Otherwise the "no such user" path returns in ~1ms and the "wrong
       password" path in ~250ms, and that TIMING DIFFERENCE leaks exactly the
       information the identical message was hiding.
    """
    email = email.lower().strip()
    # Unique constraint on users.email guarantees at most one row.
    user = await session.scalar(select(User).where(User.email == email))

    if user is None:
        # Cost-matching dummy verify. The hash is a real bcrypt hash of a
        # random string, so the work factor matches the genuine path.
        await _verify_in_thread(
            password, "$2b$12$C6UzMDM.H6dfI/f/IKcEe.7nCoiWuTnZa1s1i8kZ3jGYqRz1ry8Xy"
        )
        raise AuthenticationError("Invalid email or password")

    if not await _verify_in_thread(password, user.password_hash):
        raise AuthenticationError("Invalid email or password")

    if not user.is_active:
        raise AuthenticationError("Account is disabled")

    return user


async def issue_token_pair(
    session: AsyncSession, settings: Settings, user: User
) -> tuple[str, str]:
    """Mint an access + refresh pair and record the refresh jti for revocation."""
    access, _, _ = create_token(
        subject=user.id,
        org_id=user.org_id,
        role=user.role,
        token_type="access",  # noqa: S106 - a claim value, not a password
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        ttl_seconds=settings.access_token_ttl_seconds,
    )
    refresh, jti, expires_at = create_token(
        subject=user.id,
        org_id=user.org_id,
        role=user.role,
        token_type="refresh",  # noqa: S106 - a claim value, not a password
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        ttl_seconds=settings.refresh_token_ttl_seconds,
    )

    session.add(RefreshToken(jti=jti, user_id=user.id, expires_at=expires_at))
    await session.commit()
    return access, refresh


async def rotate_refresh_token(
    session: AsyncSession, settings: Settings, refresh_token: str
) -> tuple[str, str]:
    """
    Exchange a refresh token for a new pair, REVOKING the old one.

    This is "refresh token rotation". Each refresh token is single-use, so a
    stolen one is only useful until the legitimate client refreshes - at which
    point the thief's copy is already revoked.

    LIMITATION: reuse detection is not implemented. A fuller implementation
    treats presentation of an already-REVOKED token as proof that two parties
    hold it, and revokes the user's entire token family to force a re-login.
    Here a revoked token is simply rejected.
    """
    payload = decode_token(
        refresh_token,
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expected_type="refresh",
    )

    stored = await session.scalar(select(RefreshToken).where(RefreshToken.jti == payload["jti"]))
    if stored is None or stored.revoked_at is not None:
        raise AuthenticationError("Refresh token is no longer valid")

    user = await session.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise AuthenticationError("Account is disabled")

    stored.revoked_at = datetime.now(UTC)
    await session.flush()
    return await issue_token_pair(session, settings, user)


async def revoke_refresh_token(
    session: AsyncSession, settings: Settings, refresh_token: str
) -> None:
    """Logout. Best-effort: an already-invalid token is not an error."""
    try:
        payload = decode_token(
            refresh_token,
            secret=settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
            expected_type="refresh",
        )
    except AuthenticationError:
        return

    stored = await session.scalar(select(RefreshToken).where(RefreshToken.jti == payload["jti"]))
    if stored is not None and stored.revoked_at is None:
        stored.revoked_at = datetime.now(UTC)
        await session.commit()
