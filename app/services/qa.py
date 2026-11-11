"""
Question answering: retrieve -> prompt -> stream -> persist with citations.

THE PIPELINE
------------
    load matter chunks  ->  BM25 rank  ->  take top-k  ->  build prompt
        ->  stream model output  ->  resolve [S1] markers to real citations
        ->  persist question + answer + citations (the audit trail)

WHY WE SEND THE SOURCES TO THE CLIENT FIRST
-------------------------------------------
The first SSE event is `sources`, before a single answer token. The UI can then
render the citation panel immediately, and the user sees which documents are
being used while the answer is still generating. It also means that if
generation fails halfway, the user still has the passages - degraded, but useful.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.pagination import Cursor, Page, build_page
from app.db.models import Question, QuestionStatus
from app.services.ingestion import load_matter_chunks
from app.services.llm import LLMClient
from app.services.matters import get_matter
from app.services.retrieval import ScoredChunk, build_index

logger = get_logger(__name__)

# Matches [S1] and [S2, S3] so we can map markers back to chunk ids.
_CITATION_RE = re.compile(r"\[S(\d+(?:\s*,\s*S?\d+)*)\]")


@dataclass
class ResolvedCitation:
    chunk_id: str
    document_id: str
    document_title: str
    ordinal: int
    quote: str


def resolve_citations(answer: str, chunks: list[ScoredChunk]) -> list[ResolvedCitation]:
    """
    Turn the model's [S1] markers into real, checkable citations.

    We only return sources the model ACTUALLY cited, in the order it first cited
    them. That matters: listing all k retrieved passages as "sources" implies the
    answer rests on all of them, which is misleading. If a passage was retrieved
    but unused, it is not a citation.

    Out-of-range markers (the model invents [S9] when we sent 5 passages) are
    dropped rather than trusted. Always validate model output against your own
    data before showing it as fact.
    """
    seen: dict[int, ResolvedCitation] = {}

    for match in _CITATION_RE.finditer(answer):
        for raw in match.group(1).split(","):
            index_str = raw.strip().lstrip("S")
            if not index_str.isdigit():
                continue
            index = int(index_str)
            if not (1 <= index <= len(chunks)):
                continue  # hallucinated marker
            if index in seen:
                continue
            chunk = chunks[index - 1]
            quote = chunk.text.strip().replace("\n", " ")
            seen[index] = ResolvedCitation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                ordinal=chunk.ordinal,
                quote=quote[:500],  # keep the payload bounded
            )

    return [seen[k] for k in sorted(seen)]


async def retrieve_for_question(
    session: AsyncSession, *, org_id: str, matter_id: str, question: str, top_k: int
) -> list[ScoredChunk]:
    await get_matter(session, org_id=org_id, matter_id=matter_id)  # tenancy + 404
    rows = await load_matter_chunks(session, org_id, matter_id)
    if not rows:
        return []
    return build_index(rows).search(question, top_k)


async def create_question(
    session: AsyncSession, *, org_id: str, matter_id: str, user_id: str, text: str
) -> Question:
    """Persist the question BEFORE calling the model, so nothing is lost if the
    generation crashes. The row is the anchor the answer is attached to."""
    question = Question(
        org_id=org_id,
        matter_id=matter_id,
        asked_by=user_id,
        text=text,
        status=QuestionStatus.PENDING,
    )
    session.add(question)
    await session.commit()
    await session.refresh(question)
    return question


async def answer_question(
    session: AsyncSession,
    llm: LLMClient,
    *,
    org_id: str,
    matter_id: str,
    user_id: str,
    text: str,
    top_k: int,
) -> Question:
    """
    Non-streaming path: collect the whole answer, then return the saved row.

    Useful for API clients, batch jobs and tests. Same code path as streaming -
    we just drain the generator instead of forwarding it.
    """
    started = time.perf_counter()
    question = await create_question(
        session, org_id=org_id, matter_id=matter_id, user_id=user_id, text=text
    )
    chunks = await retrieve_for_question(
        session, org_id=org_id, matter_id=matter_id, question=text, top_k=top_k
    )

    parts: list[str] = []
    try:
        async for piece in llm.stream_answer(text, chunks):
            parts.append(piece)
    except Exception:
        logger.exception("answer_generation_failed", extra={"question_id": question.id})
        question.status = QuestionStatus.FAILED
        await session.commit()
        raise

    answer = "".join(parts)
    citations = resolve_citations(answer, chunks)

    question.answer = answer
    question.citations = [c.__dict__ for c in citations]
    question.status = QuestionStatus.ANSWERED
    question.model = llm.model_name
    question.latency_ms = int((time.perf_counter() - started) * 1000)
    await session.commit()
    await session.refresh(question)
    return question


async def stream_question_answer(
    session: AsyncSession,
    llm: LLMClient,
    *,
    org_id: str,
    matter_id: str,
    user_id: str,
    text: str,
    top_k: int,
) -> AsyncIterator[tuple[str, dict]]:
    """
    Streaming path. Yields (event_name, payload) pairs; the endpoint formats
    them as Server-Sent Events.

    EVENT SEQUENCE
        question  {id}                      - so the client can link/poll it
        sources   {sources: [...]}          - render the citation panel now
        delta     {text: "..."}             - many of these
        done      {citations, latency_ms}   - final, validated citations
        error     {message}                 - terminal failure

    WHY SSE AND NOT WEBSOCKETS
    --------------------------
    This traffic is one-directional: server -> client. SSE is plain HTTP, so it
    inherits auth headers, proxies, compression and reconnection (the browser's
    EventSource retries automatically). WebSockets are a second protocol to
    secure, scale and debug, and you need them only for genuine bidirectional
    traffic, which this is not.

    ONE REAL CAVEAT: the response has already started (HTTP 200 sent) by the
    time generation can fail, so you cannot switch to a 500. That is why errors
    are an in-band `error` event, and why the client must treat "stream ended
    without `done`" as a failure.
    """
    started = time.perf_counter()

    question = await create_question(
        session, org_id=org_id, matter_id=matter_id, user_id=user_id, text=text
    )
    yield "question", {"id": question.id, "matter_id": matter_id}

    chunks = await retrieve_for_question(
        session, org_id=org_id, matter_id=matter_id, question=text, top_k=top_k
    )
    yield (
        "sources",
        {
            "sources": [
                {
                    "marker": f"S{i}",
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "document_title": c.document_title,
                    "ordinal": c.ordinal,
                    "score": round(c.score, 4),
                }
                for i, c in enumerate(chunks, start=1)
            ]
        },
    )

    parts: list[str] = []
    try:
        async for piece in llm.stream_answer(text, chunks):
            parts.append(piece)
            yield "delta", {"text": piece}
    except Exception as exc:
        logger.exception("stream_generation_failed", extra={"question_id": question.id})
        question.status = QuestionStatus.FAILED
        await session.commit()
        yield "error", {"message": "Answer generation failed", "detail": str(exc)[:200]}
        return

    answer = "".join(parts)
    citations = resolve_citations(answer, chunks)

    question.answer = answer
    question.citations = [c.__dict__ for c in citations]
    question.status = QuestionStatus.ANSWERED
    question.model = llm.model_name
    question.latency_ms = int((time.perf_counter() - started) * 1000)
    await session.commit()

    yield (
        "done",
        {
            "question_id": question.id,
            "citations": [c.__dict__ for c in citations],
            "latency_ms": question.latency_ms,
            "model": llm.model_name,
        },
    )


async def get_question(session: AsyncSession, *, org_id: str, question_id: str) -> Question:
    question = await session.scalar(
        select(Question).where(Question.id == question_id, Question.org_id == org_id)
    )
    if question is None:
        raise NotFoundError("Question not found")
    return question


async def list_questions(
    session: AsyncSession, *, org_id: str, matter_id: str, limit: int, cursor: str | None
) -> Page:
    stmt = select(Question).where(Question.org_id == org_id, Question.matter_id == matter_id)
    if cursor:
        position = Cursor.decode(cursor)
        stmt = stmt.where(
            (Question.created_at < position.created_at)
            | ((Question.created_at == position.created_at) & (Question.id < position.id))
        )
    stmt = stmt.order_by(Question.created_at.desc(), Question.id.desc()).limit(limit + 1)
    rows = list(await session.scalars(stmt))
    return build_page(rows, limit)
