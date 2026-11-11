"""
Password hashing and JWT handling.

PASSWORDS
---------
We use bcrypt. Three properties matter:
  1. It is a *slow* KDF by design (tunable via `rounds`). Speed is the enemy;
     a fast hash like SHA-256 lets an attacker try billions of guesses/second.
  2. It salts automatically. The salt is embedded in the output string, so two
     users with the same password get different hashes (defeats rainbow tables).
  3. Verification is constant-time-ish and handled by the library.

bcrypt has a hard 72-byte input limit and silently truncates beyond it, so we
reject longer passwords explicitly instead of pretending they were checked.

Alternatives: Argon2id (OWASP's current first choice, memory-hard), scrypt.
bcrypt remains perfectly acceptable.

TOKENS
------
Two token types:
  * access  - short lived (15 min), sent as `Authorization: Bearer ...`
  * refresh - long lived (14 days), used only to mint new access tokens

Why split them? A stolen access token expires quickly. A refresh token can be
revoked server-side (we store a `jti` and check a revocation list) because it is
presented rarely, so the extra DB lookup is cheap.

The `typ` claim prevents "token confusion": a refresh token must not be
accepted where an access token is expected, and vice versa.

CPU NOTE (asyncio): bcrypt burns CPU for ~250ms. Doing that directly inside an
async endpoint would block the event loop and stall every other request on the
worker. See `app/services/auth.py` - we push it to a thread with
`anyio.to_thread.run_sync`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
import jwt

from app.core.errors import AuthenticationError, ValidationError

TokenType = Literal["access", "refresh"]

# bcrypt truncates silently past 72 bytes -> a 200-char password would be
# equivalent to its first 72 bytes. Reject instead of silently weakening.
_BCRYPT_MAX_BYTES = 72


def hash_password(plain: str, rounds: int = 12) -> str:
    """Return a bcrypt hash string ('$2b$12$<salt><digest>')."""
    raw = plain.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        raise ValidationError("Password must be at most 72 bytes long")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time-ish comparison. Never compare hashes with `==` yourself."""
    raw = plain.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash in the DB (e.g. migrated from another scheme).
        # Fail closed rather than raising a 500.
        return False


def create_token(
    *,
    subject: str,
    org_id: str,
    role: str,
    token_type: TokenType,
    secret: str,
    algorithm: str,
    ttl_seconds: int,
) -> tuple[str, str, datetime]:
    """
    Build a signed JWT.

    Returns (encoded_token, jti, expires_at) - the caller may want to persist
    the jti so the token can be revoked later.

    Standard claims used here:
      sub - subject (the user id)
      iat - issued at
      exp - expiry (PyJWT enforces this on decode)
      jti - unique token id, enables revocation
    Custom claims (`org`, `role`, `typ`) are namespaced by convention only;
    keep them small - the token travels on every request.
    """
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=ttl_seconds)
    jti = uuid.uuid4().hex

    payload: dict[str, Any] = {
        "sub": subject,
        "org": org_id,
        "role": role,
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": jti,
    }
    return jwt.encode(payload, secret, algorithm=algorithm), jti, expires_at


def decode_token(
    token: str,
    *,
    secret: str,
    algorithm: str,
    expected_type: TokenType,
) -> dict[str, Any]:
    """
    Verify signature + expiry + token type.

    SECURITY: we pass `algorithms=[algorithm]` explicitly. Accepting whatever
    the token's own header claims is the classic "alg: none" / algorithm
    confusion vulnerability - an attacker flips HS256 to none, or to HS256
    signed with your *public* RSA key, and forges any identity they like.
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired") from None
    except jwt.InvalidTokenError:
        # Deliberately vague: do not tell an attacker *why* it failed.
        raise AuthenticationError("Invalid authentication token") from None

    if payload.get("typ") != expected_type:
        raise AuthenticationError("Invalid authentication token")
    return payload
