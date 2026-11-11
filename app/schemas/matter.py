"""Matter (case file) schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class MatterCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    client_reference: str | None = Field(default=None, max_length=100)


class MatterUpdate(BaseModel):
    # All-optional = PATCH semantics. `model_dump(exclude_unset=True)` in the
    # service then distinguishes "field omitted" from "field set to null" -
    # a distinction a plain `is None` check silently destroys.
    title: str | None = Field(default=None, min_length=1, max_length=300)
    client_reference: str | None = Field(default=None, max_length=100)


class MatterResponse(ORMModel):
    id: str
    title: str
    client_reference: str | None
    org_id: str
    created_at: datetime
    updated_at: datetime
