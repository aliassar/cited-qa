"""Auth request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from app.schemas.common import ORMModel


class RegisterRequest(BaseModel):
    org_name: str = Field(min_length=2, max_length=200)
    # EmailStr does real RFC-ish validation. Validate at the EDGE so no invalid
    # value ever reaches the service layer or the database.
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=200)
    # min_length is a floor, not a policy. Real policy = length + a breach-list
    # check (Have I Been Pwned k-anonymity API). Complexity rules ("one capital,
    # one symbol") are actively discouraged by NIST 800-63B: they push users to
    # "Passw0rd!" and add no entropy.
    password: str = Field(min_length=12, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    # "bearer" is the RFC 6750 scheme name; clients send `Authorization: Bearer <token>`.
    token_type: str = "bearer"  # noqa: S105 - a scheme name, not a password
    expires_in: int = Field(description="Access token lifetime in seconds")


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(ORMModel):
    """
    Note what is ABSENT: password_hash. That omission is the security control.
    """

    id: str
    email: str
    full_name: str
    role: str
    org_id: str
    is_active: bool
