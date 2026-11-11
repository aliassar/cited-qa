"""Document schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class DocumentResponse(ORMModel):
    id: str
    matter_id: str
    title: str
    content_type: str
    size_bytes: int
    checksum: str
    status: str
    failure_reason: str | None
    chunk_count: int
    # Exposed so the client can send it back in `If-Match` on the next write.
    # Optimistic locking only works if the client can see the version it read.
    version: int
    created_at: datetime
    updated_at: datetime


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
