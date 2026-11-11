"""
The domain model.

THE DOMAIN (so the schema makes sense)
--------------------------------------
cited-qa: a legal document workspace.

  Organization   a law firm. The tenant boundary - EVERY query is scoped to it.
    User         a lawyer or paralegal, with a role.
    Matter       a case/engagement; the folder documents live in.
      Document   an uploaded contract, filing, or memo.
        Chunk    a passage of that document, produced by ingestion; the unit we
                 retrieve and cite. Citing a whole 80-page PDF is useless; a
                 lawyer needs the paragraph.
      Question   a natural-language question asked against a matter, answered
                 by an LLM strictly from retrieved chunks, with citations.

MULTI-TENANCY
-------------
`org_id` is denormalised onto every table, even where it could be derived by
joining up to the matter. That is deliberate: it lets every single query filter
on `org_id` directly, so a forgotten join can never leak another firm's data. In
Postgres you would additionally enable Row Level Security as a second, database-
enforced net. Defence in depth: for legal documents, a cross-tenant leak is not
a bug, it is a breach.

WHY STRING ENUMS AND NOT sqlalchemy.Enum
-----------------------------------------
Native Postgres ENUM types require a migration (and an exclusive lock) to add a
value. A CHECK-constrained VARCHAR with a Python `StrEnum` in front gives the
same type safety in the application with far cheaper schema evolution.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Role(StrEnum):
    """Ordered least- to most-privileged. See `app/api/deps.py::require_role`."""

    VIEWER = "viewer"  # read only
    MEMBER = "member"  # + create/edit matters and documents, ask questions
    ADMIN = "admin"  # + delete matters and documents


ROLE_RANK = {Role.VIEWER: 0, Role.MEMBER: 1, Role.ADMIN: 2}


class DocumentStatus(StrEnum):
    """
    The ingestion state machine:
        PENDING -> PROCESSING -> READY
                             \\-> FAILED
    Exposing this to the client is what makes async ingestion honest: the upload
    returns 202 immediately and the UI polls (or subscribes to) the status
    rather than pretending the document is instantly searchable.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class QuestionStatus(StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    FAILED = "failed"


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    users: Mapped[list[User]] = relationship(back_populates="organization")


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        # Emails are unique GLOBALLY, not per organisation. Login takes only an
        # email and a password, so an email must identify exactly one account;
        # with per-org uniqueness the same address could exist at two firms and
        # login would have no way to choose between them. LIMITATION: a person
        # who works with two firms needs a separate email per firm. Supporting
        # one email across firms would need an org selector at login (or a
        # user <-> org membership table), which this codebase does not have.
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint("role in ('viewer','member','admin')", name="role_valid"),
    )

    org_id: Mapped[str] = mapped_column(
        # ondelete="CASCADE" is the DATABASE-level rule. It is what actually
        # runs; the ORM's `cascade` argument only affects objects loaded in the
        # session. State both, or you get orphans when you delete via SQL.
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=Role.MEMBER)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    organization: Mapped[Organization] = relationship(back_populates="users")


class RefreshToken(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Server-side record of issued refresh tokens, so they can be REVOKED.

    A plain JWT is valid until it expires - you cannot un-issue it. For the
    long-lived refresh token that is unacceptable (logout, password change,
    stolen device must all kill the session immediately), so we keep the `jti`
    here and check it on every refresh.

    Access tokens have no equivalent table. Every request does load the USER
    row (see api/deps.py::get_current_user), so deactivating an account or
    changing its role takes effect on the very next request. What is NOT
    possible is revoking one specific access token while the account stays
    active: there is no per-token denylist, so a leaked access token remains
    usable until its 15-minute TTL runs out.
    """

    __tablename__ = "refresh_tokens"

    jti: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class Matter(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "matters"
    __table_args__ = (
        # Composite index in exactly the order the list query sorts:
        #   WHERE org_id = ? ORDER BY created_at DESC, id DESC
        # Equality column first, then the range/sort columns. This single index
        # serves both the filter and the keyset pagination - no sort step.
        Index("ix_matters_org_created", "org_id", "created_at", "id"),
    )

    org_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    client_reference: Mapped[str | None] = mapped_column(String(100), default=None)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="matter", cascade="all, delete-orphan"
    )


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_matter_created", "matter_id", "created_at", "id"),
        Index("ix_documents_org_status", "org_id", "status"),
        # Same file uploaded twice to the same matter is a no-op, enforced by
        # the DATABASE rather than by a read-then-write in Python (which races).
        UniqueConstraint("matter_id", "checksum", name="uq_documents_matter_id_checksum"),
        CheckConstraint("status in ('pending','processing','ready','failed')", name="status_valid"),
    )

    org_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    matter_id: Mapped[str] = mapped_column(
        ForeignKey("matters.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    # SHA-256 of the bytes: dedupe key + integrity check + audit evidence that
    # the stored document is the one that was uploaded.
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    # Here the extracted text lives in the row. In production the BYTES go to
    # object storage (S3) and the row keeps only a pointer: databases are
    # expensive per GB, back up slowly, and blobs blow out the buffer cache that
    # your hot queries depend on.
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text, default=None)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default=DocumentStatus.PENDING)
    failure_reason: Mapped[str | None] = mapped_column(String(500), default=None)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ------------------------------------------------------------------
    # OPTIMISTIC CONCURRENCY CONTROL.
    #
    # SQLAlchemy adds `version` to the WHERE clause of every UPDATE:
    #     UPDATE documents SET title=?, version=6 WHERE id=? AND version=5
    # If another transaction already bumped it to 6, zero rows match and
    # SQLAlchemy raises StaleDataError. We translate that to HTTP 409.
    #
    # This is how two lawyers editing the same document stop silently
    # overwriting each other ("last write wins" is a data-loss bug, not a
    # policy). The client sends the version it read via the If-Match header;
    # a mismatch means "someone changed this, re-read and retry".
    #
    # Optimistic (this) suits low-contention workloads: no locks held, no
    # blocking. Pessimistic (SELECT ... FOR UPDATE) suits high contention,
    # at the cost of held locks and deadlock risk.
    # ------------------------------------------------------------------
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __mapper_args__ = {"version_id_col": version}

    matter: Mapped[Matter] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    A retrievable passage. `ordinal` preserves document order so a citation can
    say "clause 4, page 12" rather than "somewhere in this PDF".
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_document_chunks_document_id_ordinal"),
        Index("ix_chunks_org_matter", "org_id", "matter_id"),
    )

    org_id: Mapped[str] = mapped_column(String(32), nullable=False)
    matter_id: Mapped[str] = mapped_column(String(32), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # In production this row would also carry `embedding vector(1536)` with a
    # pgvector HNSW index, and retrieval would be hybrid: BM25 keyword search
    # UNION semantic search, fused with Reciprocal Rank Fusion. See
    # app/services/retrieval.py - we implement the keyword half for real.
    document: Mapped[Document] = relationship(back_populates="chunks")


class Question(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    An asked question and its answer. Persisting these is not optional in legal
    tech: you must be able to show, months later, exactly what the model was
    asked, what sources it was given, and what it replied. That is the audit
    trail regulators and malpractice insurers ask for.
    """

    __tablename__ = "questions"
    __table_args__ = (Index("ix_questions_matter_created", "matter_id", "created_at", "id"),)

    org_id: Mapped[str] = mapped_column(String(32), nullable=False)
    matter_id: Mapped[str] = mapped_column(
        ForeignKey("matters.id", ondelete="CASCADE"), nullable=False
    )
    asked_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=QuestionStatus.PENDING)
    answer: Mapped[str | None] = mapped_column(Text, default=None)
    # JSON column: portable across SQLite and Postgres (JSONB there).
    # Shape: [{"chunk_id": ..., "document_id": ..., "document_title": ..., "quote": ...}]
    citations: Mapped[list | None] = mapped_column(JSON, default=None)
    model: Mapped[str | None] = mapped_column(String(100), default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)


class IdempotencyKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Makes unsafe POSTs safe to retry.

    THE PROBLEM: client POSTs "create matter", the response is lost to a network
    blip, the client retries -> two matters. HTTP says POST is not idempotent,
    so the client cannot fix this alone; the server must.

    THE CONTRACT:
      * client generates a unique `Idempotency-Key` per logical operation
      * first request: we execute, then store status + response body
      * replay with the same key: we return the STORED response, without
        re-executing
      * same key with a DIFFERENT request body: 409. The key identifies one
        specific operation; reusing it for another is a client bug we must not
        paper over. That is why we store `request_hash`.

    Stripe's API works exactly like this.
    """

    __tablename__ = "idempotency_keys"
    # Scoped per (user, endpoint, key): one user's key can never collide with
    # another user's, and reusing a key on a different endpoint is a separate
    # operation rather than a false "different request body" conflict.
    __table_args__ = (
        UniqueConstraint(
            "user_id", "endpoint", "key", name="uq_idempotency_keys_user_id_endpoint_key"
        ),
    )

    user_id: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(200), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    response_body: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Old keys must be swept (a nightly job deleting created_at < now - 24h);
    # otherwise this table grows forever.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
