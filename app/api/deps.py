"""
FastAPI dependencies - the composable middle layer.

WHY DEPENDENCY INJECTION IS THE BEST PART OF FASTAPI
-----------------------------------------------------
A dependency is just a function. FastAPI resolves it before the handler runs,
caches it per request, and - crucially - lets tests REPLACE it via
`app.dependency_overrides`. That is how you swap a real database for a test one,
or a real LLM for a fake, without a single mock library or monkeypatch.

Dependencies also compose: `require_role(ADMIN)` depends on `get_current_user`,
which depends on the bearer token and the session. Declare the leaf you need and
FastAPI builds the chain. The result is that authorisation is expressed in the
route SIGNATURE, where it is visible in code review, instead of hidden in an
`if` at the top of a 60-line handler.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, Header, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import (
    AuthenticationError,
    ConflictError,
    PermissionDeniedError,
    RateLimitedError,
)
from app.core.security import decode_token
from app.db.models import ROLE_RANK, IdempotencyKey, Role, User
from app.db.session import get_session
from app.services.llm import LLMClient

# auto_error=False so a MISSING header reaches our handler and produces our
# RFC 9457 error envelope, instead of Starlette's bare {"detail": "..."}.
_bearer = HTTPBearer(auto_error=False)


def get_app_settings(request: Request) -> Settings:
    """
    Return the settings THIS app instance was built with.

    Subtle but important: `get_settings()` is an lru_cached global. If a
    dependency used it directly, every request would read the process-wide
    Settings - so a test that builds an app with rate limiting disabled would
    still be rate limited, because the *dependency* never saw those settings.
    Reading from `app.state` keeps configuration per-app-instance, which is what
    makes the factory pattern in main.py actually work.
    """
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> User:
    """
    Resolve `Authorization: Bearer <jwt>` to a User row.

    We verify the signature (cheap, local) AND load the user (one indexed
    lookup). Why not trust the token alone and skip the query? Because a user
    can be deactivated or have their role changed after the token was issued,
    and a 15-minute window of stale privileges is not acceptable for a system
    holding privileged legal documents. One primary-key lookup is a fine price.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token")

    payload = decode_token(
        credentials.credentials,
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expected_type="access",  # a refresh token must NOT work here
    )

    user = await session.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise AuthenticationError("Account is no longer active")

    # The token's org claim must still match the stored row. They diverge only
    # if the token was tampered with or the user was moved between orgs; either
    # way, refuse rather than trust the claim.
    if user.org_id != payload.get("org"):
        raise AuthenticationError("Token does not match account")

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(minimum: Role):
    """
    Dependency FACTORY: returns a dependency that enforces a role floor.

    Used as `dependencies=[Depends(require_role(Role.ADMIN))]` on a route, or as
    a parameter when the handler also needs the user object.

    Roles are RANKED, so `require_role(MEMBER)` also admits admins. Comparing
    strings for equality instead would mean listing every acceptable role at
    every call site - and forgetting one is a silent privilege bug.
    """

    async def _dependency(user: CurrentUser) -> User:
        if ROLE_RANK.get(Role(user.role), -1) < ROLE_RANK[minimum]:
            raise PermissionDeniedError(
                f"This action requires the '{minimum.value}' role or higher"
            )
        return user

    return _dependency


# Ready-made aliases. Using `Annotated[User, Depends(...)]` (rather than a
# default value of `Depends(...)`) keeps the parameter positional, so it can sit
# before parameters that DO have defaults - and FastAPI forbids mixing the two
# forms on one parameter anyway.
MemberUser = Annotated[User, Depends(require_role(Role.MEMBER))]
AdminUser = Annotated[User, Depends(require_role(Role.ADMIN))]


async def enforce_rate_limit(request: Request, settings: SettingsDep) -> None:
    """
    Per-user (or per-IP) rate limiting.

    Keying by USER id where possible is the right default: an IP can be a whole
    office behind NAT, so IP-keyed limits punish colleagues for one person's
    script. We fall back to IP for unauthenticated routes such as /login - which
    is exactly where you most want a limit, to slow credential stuffing.

    We read the user from the token WITHOUT hitting the database: this runs on
    every request and must stay cheap. A forged token fails properly a moment
    later in get_current_user; here it only affects which bucket it lands in.
    """
    if not settings.rate_limit_enabled:
        return

    key = f"ip:{request.client.host if request.client else 'unknown'}"
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        try:
            payload = decode_token(
                auth.split(" ", 1)[1],
                secret=settings.jwt_secret,
                algorithm=settings.jwt_algorithm,
                expected_type="access",
            )
            key = f"user:{payload['sub']}"
        except AuthenticationError:
            pass  # keep the IP bucket

    allowed, retry_after = request.app.state.rate_limiter.allow(key)
    if not allowed:
        raise RateLimitedError("Rate limit exceeded. Slow down.", retry_after=retry_after)


RateLimited = Depends(enforce_rate_limit)


def get_llm(request: Request) -> LLMClient:
    """
    Hand out the process-wide LLM client (built once in the lifespan).

    Because this is a dependency, a test can do:
        app.dependency_overrides[get_llm] = lambda: MyDeterministicFake()
    and the entire QA pipeline runs offline. That is the payoff of the port /
    adapter split in services/llm.py.
    """
    return request.app.state.llm_client


LLMDep = Annotated[LLMClient, Depends(get_llm)]


class PaginationParams:
    """
    Shared `?limit=&cursor=` parsing.

    `le=100` is not decoration: without an upper bound a client can ask for
    limit=1000000 and turn your list endpoint into a denial-of-service against
    your own database. Always cap page size.
    """

    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
        cursor: Annotated[str | None, Query(description="Opaque cursor from next_cursor")] = None,
    ) -> None:
        self.limit = limit
        self.cursor = cursor


PageParams = Annotated[PaginationParams, Depends()]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class IdempotencyGuard:
    """
    Replay protection for unsafe POSTs. See db/models.py::IdempotencyKey for the
    full rationale.

    Usage in a handler (see api/v1/matters.py::create_matter):

        cached = await idempotency.lookup(key, "POST /matters", body)
        if cached is not None:
            status_code, stored_body = cached   # replay: return this, do no work
            ...
        ...do the work...
        await idempotency.store(key, "POST /matters", body, status_code, response_body)

    Keys are scoped per (user, endpoint): `lookup` and `store` both match on
    the user, the endpoint label and the key. A missing key (header not sent)
    makes both methods no-ops, so idempotency is opt-in for the client.

    KNOWN LIMITATION: we store the result AFTER the work
    succeeds. Two truly simultaneous requests with the same key can therefore
    both start. A production implementation inserts the key row FIRST, inside
    the same transaction as the work, so the unique constraint makes the second
    request block or fail fast. The 24h expiry sweep is also assumed to be a
    separate cron job.
    """

    def __init__(self, session: AsyncSession, user: User) -> None:
        self._session = session
        self._user = user

    @staticmethod
    def _hash(body: dict) -> str:
        # sort_keys=True so {"a":1,"b":2} and {"b":2,"a":1} hash identically -
        # JSON object key order is not semantically meaningful.
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    async def lookup(self, key: str | None, endpoint: str, body: dict) -> tuple[int, dict] | None:
        """
        Return the stored (status, body) for a replay, or None to proceed.

        Raises ConflictError if this key was already used on this endpoint
        with a different request body.
        """
        if not key:
            return None

        record = await self._session.scalar(
            select(IdempotencyKey).where(
                IdempotencyKey.user_id == self._user.id,
                IdempotencyKey.endpoint == endpoint,
                IdempotencyKey.key == key,
            )
        )
        if record is None:
            return None

        if record.request_hash != self._hash(body):
            # Same key, different payload: a client bug we must surface loudly
            # rather than silently returning the wrong resource.
            raise ConflictError(
                "This Idempotency-Key was already used with a different request body"
            )
        return record.response_status, record.response_body

    async def store(
        self, key: str | None, endpoint: str, body: dict, status_code: int, response: dict
    ) -> None:
        """Record the response for (user, endpoint, key) and commit; no-op without a key."""
        if not key:
            return
        self._session.add(
            IdempotencyKey(
                user_id=self._user.id,
                key=key,
                endpoint=endpoint,
                request_hash=self._hash(body),
                response_status=status_code,
                response_body=response,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
            )
        )
        try:
            await self._session.commit()
        except IntegrityError:
            # Lost the race to store the key. The work is done and the client
            # has their answer; swallowing this is correct.
            await self._session.rollback()


async def get_idempotency_guard(session: SessionDep, user: CurrentUser) -> IdempotencyGuard:
    return IdempotencyGuard(session, user)


IdempotencyDep = Annotated[IdempotencyGuard, Depends(get_idempotency_guard)]
IdempotencyKeyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        description="Client-generated unique key making this POST safe to retry",
    ),
]
IfMatchHeader = Annotated[
    int | None,
    Header(
        alias="If-Match",
        description="Version the client last read; rejects the write if it changed",
    ),
]
