"""
Document ingestion: text -> chunks, run off the request path.

WHY THIS IS ASYNCHRONOUS
------------------------
Parsing a 200-page PDF, chunking it and (in production) embedding every chunk
takes tens of seconds. Doing that inside the upload request means:
  * the client sits on an open connection until it times out;
  * a deploy mid-request loses the work entirely;
  * one big upload occupies a worker that should be serving other users.

So the endpoint does the minimum durable thing - persist the bytes, write a row
with status=PENDING - and returns 202 Accepted with a status URL. The heavy work
happens behind the scenes and the client polls. "202 + a status field" is the
honest HTTP shape for work that is not done yet.

*** WHAT THIS IS NOT ***
The queue here is an in-process `asyncio.Queue`. That means:
  * the queue dies with the process - a crash or a deploy loses queued work;
  * it does not spread across replicas;
  * there are no retries with backoff and no dead-letter queue;
  * observability is limited to the current depth (exposed on /health) - no
    per-message age, redelivery counts or consumer lag.
In production you use a real broker - RabbitMQ, Redis + arq, Celery, or
SQS - so that work is durable, retried, observable, and
processed by workers you can scale independently of your web tier. The status
column in the database is the state machine either way; only the transport
changes. This module is deliberately structured so that swapping the transport
touches `enqueue()` and the worker loop and nothing else.

IDEMPOTENCY OF THE WORKER
-------------------------
Real brokers deliver AT LEAST ONCE. A worker must therefore tolerate processing
the same document twice: we delete existing chunks before writing new ones, so
a re-run converges to the same state instead of duplicating passages.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger, new_request_id, request_id_ctx
from app.db.models import Document, DocumentChunk, DocumentStatus

logger = get_logger(__name__)

# Chunk sizing, in characters (production would count TOKENS, via the model's
# tokenizer or `client.messages.count_tokens`).
#
# THE TRADE-OFF:
#   too small -> a clause gets split in half; neither half is retrievable, and
#                the model sees a fragment without its condition.
#   too large -> retrieval returns mostly irrelevant text around the one
#                relevant sentence; you pay for those tokens and the signal is
#                diluted.
# ~1200 chars (~300 tokens) is a reasonable default for contract prose.
CHUNK_TARGET_CHARS = 1200

# Overlap so a sentence spanning a boundary still appears intact in one chunk.
# Without it, "The Agreement terminates on..." | "...31 December 2027" is
# unretrievable by either half. Costs ~15% storage; well worth it.
CHUNK_OVERLAP_CHARS = 180

# Paragraph-first splitting: legal documents are structured by clause, and a
# clause boundary is a far better cut point than an arbitrary character offset.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int


def chunk_text(
    text: str,
    target: int = CHUNK_TARGET_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, preferring paragraph boundaries.

    Strategy:
      1. Split on blank lines (paragraphs / clauses).
      2. Greedily pack paragraphs into a chunk until adding the next would
         exceed `target`.
      3. A single paragraph longer than `target` is hard-split with overlap.
      4. Track char offsets so a citation can point at the exact span.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    cursor = 0  # running offset into the ORIGINAL text, for citation spans

    paragraphs = _PARAGRAPH_RE.split(text)
    buffer: list[str] = []
    buffer_start = 0
    buffer_len = 0

    def flush() -> None:
        nonlocal ordinal, buffer, buffer_len, buffer_start
        if not buffer:
            return
        body = "\n\n".join(buffer)
        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=body,
                char_start=buffer_start,
                char_end=buffer_start + len(body),
            )
        )
        ordinal += 1
        buffer = []
        buffer_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Locate this paragraph in the original text to keep offsets truthful.
        found = text.find(para, cursor)
        para_start = found if found != -1 else cursor
        cursor = para_start + len(para)

        if len(para) > target:
            # Oversized paragraph (a wall-of-text clause): flush what we have,
            # then hard-split with a sliding window.
            flush()
            step = max(1, target - overlap)
            for offset in range(0, len(para), step):
                piece = para[offset : offset + target]
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        text=piece,
                        char_start=para_start + offset,
                        char_end=para_start + offset + len(piece),
                    )
                )
                ordinal += 1
                if offset + target >= len(para):
                    break
            continue

        if buffer and buffer_len + len(para) > target:
            flush()

        if not buffer:
            buffer_start = para_start
        buffer.append(para)
        buffer_len += len(para) + 2

    flush()
    return chunks


# ---------------------------------------------------------------------------
# The background worker.
# ---------------------------------------------------------------------------


class IngestionQueue:
    """
    Owns the queue and the worker tasks. Started/stopped by the app lifespan.

    The worker takes the SESSION FACTORY, not a session: a request-scoped
    session belongs to a request that has already finished by the time the
    worker runs. Sharing one across tasks corrupts its internal state.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        worker_count: int = 2,
        maxsize: int = 1000,
    ) -> None:
        # A BOUNDED queue. Unbounded means an upload spike is absorbed into RAM
        # until the process is OOM-killed. Bounded means backpressure: the
        # producer waits, or you shed load with a 503, which is a decision you
        # get to make instead of a crash you do not.
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self._session_factory = session_factory
        self._worker_count = worker_count
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        self._workers = [
            asyncio.create_task(self._worker_loop(i), name=f"ingest-worker-{i}")
            for i in range(self._worker_count)
        ]
        logger.info("ingest_workers_started", extra={"count": self._worker_count})

    async def stop(self) -> None:
        """
        GRACEFUL SHUTDOWN. Drain first, then cancel.

        `join()` waits until every queued item has been marked done, so an
        in-flight document is not abandoned half-processed on deploy. Only then
        do we cancel the (now idle) workers.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout=10.0)
        except TimeoutError:
            logger.warning("ingest_drain_timeout", extra={"pending": self._queue.qsize()})

        for task in self._workers:
            task.cancel()
        # return_exceptions=True so one worker's CancelledError does not mask
        # the others' shutdown.
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("ingest_workers_stopped")

    async def enqueue(self, document_id: str) -> None:
        await self._queue.put(document_id)

    @property
    def depth(self) -> int:
        """Exposed on /health - a queue that only grows is your first alert."""
        return self._queue.qsize()

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            document_id = await self._queue.get()
            # Give the background job its own correlation id so its log lines
            # are groupable, exactly like a request's.
            request_id_ctx.set(new_request_id())
            try:
                await self._process(document_id)
            except asyncio.CancelledError:
                # Cooperative cancellation: re-raise so the task actually stops.
                raise
            except Exception:
                # A worker that dies on one bad document stops processing every
                # other document. Log it, mark the row failed, keep the loop alive.
                logger.exception(
                    "ingest_failed", extra={"document_id": document_id, "worker": worker_id}
                )
                await self._mark_failed(document_id, "Unexpected ingestion error")
            finally:
                # Must always run, or `queue.join()` in stop() hangs forever.
                self._queue.task_done()

    async def _process(self, document_id: str) -> None:
        async with self._session_factory() as session:
            await ingest_document(session, document_id)

    async def _mark_failed(self, document_id: str, reason: str) -> None:
        # A SEPARATE session: the failing one may be in a broken transaction,
        # and every statement on it would raise PendingRollbackError.
        async with self._session_factory() as session:
            document = await session.get(Document, document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.failure_reason = reason[:500]
                await session.commit()


async def ingest_document(session: AsyncSession, document_id: str) -> None:
    """
    Chunk one document and move it through the status state machine:
    PENDING -> PROCESSING -> READY (or FAILED if there is no text to index).

    This is the single implementation of ingestion. The background worker calls
    it with a fresh session per job; `ingest_now` calls it inline.
    """
    document = await session.get(Document, document_id)
    if document is None:
        logger.warning("ingest_document_missing", extra={"document_id": document_id})
        return

    # Commit PROCESSING on its own, so a client polling the document sees that
    # work has started rather than a PENDING row that looks stuck.
    document.status = DocumentStatus.PROCESSING
    await session.commit()

    chunks = chunk_text(document.extracted_text or "")

    if not chunks:
        document.status = DocumentStatus.FAILED
        document.failure_reason = "No extractable text found in document"
        await session.commit()
        return

    # Idempotency: wipe prior chunks so a redelivery converges rather than
    # duplicating. (delete() with a WHERE, not a Python loop - one statement
    # instead of N.)
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))

    session.add_all(
        [
            DocumentChunk(
                org_id=document.org_id,
                matter_id=document.matter_id,
                document_id=document.id,
                ordinal=c.ordinal,
                text=c.text,
                char_start=c.char_start,
                char_end=c.char_end,
            )
            for c in chunks
        ]
    )

    document.chunk_count = len(chunks)
    document.status = DocumentStatus.READY
    document.failure_reason = None
    await session.commit()

    logger.info("ingest_completed", extra={"document_id": document_id, "chunks": len(chunks)})


async def ingest_now(session: AsyncSession, document_id: str) -> None:
    """
    Ingest a document inline, bypassing the queue. Used by the seed script.

    A one-shot script would exit before the background workers got to its
    documents, so it ingests synchronously instead. This runs exactly the same
    `ingest_document` the worker runs - same state transitions, same chunking -
    just on the caller's session rather than a worker's.
    """
    await ingest_document(session, document_id)


async def load_matter_chunks(session: AsyncSession, org_id: str, matter_id: str):
    """
    Fetch every chunk of a matter for retrieval, as plain tuples.

    Two deliberate choices:
      * `org_id` in the WHERE clause even though `matter_id` already implies it -
        defence in depth against a cross-tenant leak.
      * a JOIN to fetch document titles in ONE query. Loading chunks and then
        reading `chunk.document.title` in a loop is the N+1 problem: 1 query
        becomes 1 + N, and in async SQLAlchemy the lazy load raises outright.
    """
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            Document.title,
            DocumentChunk.ordinal,
            DocumentChunk.text,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.org_id == org_id,
            DocumentChunk.matter_id == matter_id,
            Document.status == DocumentStatus.READY,
        )
        .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
    )
    result = await session.execute(stmt)
    return [tuple(row) for row in result.all()]
