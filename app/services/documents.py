"""
Document service: upload, list, update-with-optimistic-locking, delete.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.pagination import Cursor, Page, build_page
from app.db.models import Document, DocumentStatus
from app.services.matters import get_matter


def compute_checksum(payload: bytes) -> str:
    """SHA-256 of the raw bytes: dedupe key, integrity check, audit evidence."""
    return hashlib.sha256(payload).hexdigest()


def extract_text(payload: bytes, content_type: str) -> str:
    """
    Turn uploaded bytes into plain text.

    Only plain text and markdown are supported, and they are decoded here, in
    the upload request. Decoding is cheap, and doing it at upload time means a
    file we cannot read is rejected immediately with a 422 instead of being
    accepted and failing later in the background.

    Real extraction is a different matter - pdfplumber / PyMuPDF for text PDFs,
    OCR (Tesseract, AWS Textract) for scans - and is slow enough that it would
    belong in the ingestion worker, not the request path. That is out of scope
    here, which is why the upload allow-list (`Settings.allowed_upload_types`)
    contains only text types.

    `errors="replace"` because real-world files carry broken encodings, and a
    UnicodeDecodeError should not become a 500.
    """
    if content_type in ("text/plain", "text/markdown"):
        return payload.decode("utf-8", errors="replace")
    raise ValidationError(f"Unsupported content type: {content_type}")


async def create_document(
    session: AsyncSession,
    *,
    org_id: str,
    matter_id: str,
    title: str,
    content_type: str,
    payload: bytes,
) -> tuple[Document, bool]:
    """
    Persist an uploaded document in PENDING state.

    Returns (document, created). `created=False` means this exact file already
    existed in this matter and we are returning the existing row - upload
    becomes naturally idempotent on content.

    THE RACE, and why the try/except is not optional:
    two concurrent uploads of the same file both SELECT, both find nothing, both
    INSERT. Only the unique constraint (matter_id, checksum) can stop the
    duplicate, because only the database sees both transactions. A check-then-
    act in Python is a TOCTOU bug. So we let the DB decide and handle the
    IntegrityError - "ask forgiveness, not permission", enforced by a constraint.
    """
    await get_matter(session, org_id=org_id, matter_id=matter_id)  # 404 if not ours

    checksum = compute_checksum(payload)

    existing = await session.scalar(
        select(Document).where(Document.matter_id == matter_id, Document.checksum == checksum)
    )
    if existing is not None:
        return existing, False

    document = Document(
        org_id=org_id,
        matter_id=matter_id,
        title=title,
        content_type=content_type,
        size_bytes=len(payload),
        checksum=checksum,
        # In production: the S3 key. Here: a deterministic logical path.
        storage_key=f"org/{org_id}/matter/{matter_id}/{checksum}",
        extracted_text=extract_text(payload, content_type),
        status=DocumentStatus.PENDING,
    )
    session.add(document)

    try:
        await session.commit()
    except IntegrityError:
        # The other transaction won. Roll back and return their row - the
        # caller's intent ("this file should exist here") is satisfied either way.
        await session.rollback()
        existing = await session.scalar(
            select(Document).where(Document.matter_id == matter_id, Document.checksum == checksum)
        )
        if existing is None:
            raise
        return existing, False

    await session.refresh(document)
    return document, True


async def get_document(session: AsyncSession, *, org_id: str, document_id: str) -> Document:
    document = await session.scalar(
        select(Document).where(Document.id == document_id, Document.org_id == org_id)
    )
    if document is None:
        raise NotFoundError("Document not found")
    return document


async def list_documents(
    session: AsyncSession,
    *,
    org_id: str,
    matter_id: str,
    limit: int,
    cursor: str | None,
    status: str | None,
) -> Page:
    stmt = select(Document).where(Document.org_id == org_id, Document.matter_id == matter_id)

    if status:
        stmt = stmt.where(Document.status == status)

    if cursor:
        position = Cursor.decode(cursor)
        stmt = stmt.where(
            (Document.created_at < position.created_at)
            | ((Document.created_at == position.created_at) & (Document.id < position.id))
        )

    stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc()).limit(limit + 1)
    rows = list(await session.scalars(stmt))
    return build_page(rows, limit)


async def update_document(
    session: AsyncSession,
    *,
    org_id: str,
    document_id: str,
    changes: dict,
    expected_version: int | None,
) -> Document:
    """
    Update with OPTIMISTIC CONCURRENCY CONTROL.

    Two layers, and they catch different things:

    1. `expected_version` comes from the client's `If-Match` header. Comparing
       it to the row we just read catches the common case with a clear message,
       before we even attempt the write.

    2. The `version_id_col` on the model puts `AND version = :v` into the UPDATE
       itself. That is what closes the remaining window: between our SELECT and
       our UPDATE another transaction can still commit. Only the database can
       adjudicate that, and it does so by matching zero rows -> StaleDataError.

    Layer 1 without layer 2 is a check-then-act race. Layer 2 alone works but
    gives worse errors. Use both.
    """
    document = await get_document(session, org_id=org_id, document_id=document_id)

    if expected_version is not None and document.version != expected_version:
        raise ConflictError(
            "Document was modified by another user. Re-read it and retry.",
            current_version=document.version,
        )

    for field, value in changes.items():
        setattr(document, field, value)

    try:
        await session.commit()
    except StaleDataError:
        await session.rollback()
        raise ConflictError("Document was modified concurrently. Re-read it and retry.") from None

    await session.refresh(document)
    return document


async def delete_document(session: AsyncSession, *, org_id: str, document_id: str) -> None:
    document = await get_document(session, org_id=org_id, document_id=document_id)
    await session.delete(document)  # chunks cascade
    await session.commit()
