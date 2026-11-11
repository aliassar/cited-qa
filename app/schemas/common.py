"""
Shared response envelopes.

WHY DTOs INSTEAD OF RETURNING ORM OBJECTS
------------------------------------------
1. SECURITY. `User` holds `password_hash`. Return the ORM object and Pydantic
   will happily serialise it. An explicit response schema is an allow-list: if
   a field is not on the schema, it cannot leak. This is the single best reason.
2. CONTRACT STABILITY. Renaming a column should not break every client.
3. LAZY LOADING. Serialising an ORM object can trigger unexpected relationship
   loads - N+1 queries at best, MissingGreenlet in async at worst.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas built from ORM instances."""

    # from_attributes=True lets Pydantic read attributes off an arbitrary
    # object (the SQLAlchemy instance) instead of requiring a dict.
    model_config = ConfigDict(from_attributes=True)


class PageResponse(BaseModel, Generic[T]):
    """
    Cursor-paginated list envelope.

    `next_cursor` is null on the last page. The client's loop is simply:
    keep calling with the cursor you were given until it comes back null.
    We do NOT return a total count - see app/core/pagination.py for why
    COUNT(*) on a large table is a trap.
    """

    items: list[T]
    next_cursor: str | None = Field(
        default=None,
        description="Pass as ?cursor= to fetch the next page. Null means no more results.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, str] = Field(default_factory=dict)
