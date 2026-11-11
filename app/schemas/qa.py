"""Question-answering schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # How many chunks to retrieve and feed the model. Bounded on purpose:
    # unbounded k means unbounded prompt size, unbounded cost, unbounded
    # latency - and worse answers, because relevant passages get buried.
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    """
    The core requirement of this API, not a nice-to-have.

    A legal answer without a source is unusable: the lawyer still has to verify
    everything, so the tool has saved them nothing. Every claim must point back
    to a specific chunk of a specific document. This is also the honest defence
    against hallucination - you cannot stop a model inventing text, but you can
    make every sentence checkable in one click.
    """

    chunk_id: str
    document_id: str
    document_title: str
    ordinal: int
    quote: str


class AnswerResponse(ORMModel):
    id: str
    matter_id: str
    text: str
    status: str
    answer: str | None
    citations: list[Citation] | None
    model: str | None
    latency_ms: int | None
    created_at: datetime
