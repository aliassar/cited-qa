"""
Matter service: tenant-scoped CRUD with keyset pagination.

Every function takes `org_id` and puts it in the WHERE clause. There is no
"get by id" that skips it. That is the tenancy invariant, and it is enforced by
making the unsafe version impossible to call rather than by remembering.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.pagination import Cursor, Page, build_page
from app.db.models import Matter


async def create_matter(
    session: AsyncSession, *, org_id: str, user_id: str, title: str, client_reference: str | None
) -> Matter:
    matter = Matter(
        org_id=org_id, created_by=user_id, title=title, client_reference=client_reference
    )
    session.add(matter)
    await session.commit()
    # refresh() re-SELECTs so server-side defaults (created_at) are populated
    # on the instance we are about to serialise.
    await session.refresh(matter)
    return matter


async def get_matter(session: AsyncSession, *, org_id: str, matter_id: str) -> Matter:
    """
    NOTE: `org_id` is in the WHERE clause, not checked after the fetch.

    Fetching by id and then doing `if matter.org_id != org_id: raise` also works,
    but it is one forgotten line away from a cross-tenant leak, and it briefly
    pulls another tenant's row into memory. Filtering in SQL is safer by
    construction.

    Returning 404 (not 403) for another org's id is deliberate: 403 would
    confirm the resource exists, which is itself an information leak.
    """
    matter = await session.scalar(
        select(Matter).where(Matter.id == matter_id, Matter.org_id == org_id)
    )
    if matter is None:
        raise NotFoundError("Matter not found")
    return matter


async def list_matters(
    session: AsyncSession, *, org_id: str, limit: int, cursor: str | None, search: str | None
) -> Page:
    stmt = select(Matter).where(Matter.org_id == org_id)

    if search:
        # ILIKE is fine for a title field on a small table. For real search over
        # document bodies you want a GIN index on a tsvector column - a leading-
        # wildcard LIKE cannot use a B-tree index and degrades to a full scan.
        stmt = stmt.where(Matter.title.ilike(f"%{search}%"))

    if cursor:
        position = Cursor.decode(cursor)
        # Row-value comparison: "strictly before this (created_at, id) pair".
        # Postgres supports the tuple form `(a, b) < (:a, :b)` natively; the
        # expanded form below is portable and produces the same result.
        stmt = stmt.where(
            (Matter.created_at < position.created_at)
            | ((Matter.created_at == position.created_at) & (Matter.id < position.id))
        )

    # +1 row to learn whether another page exists - see build_page().
    stmt = stmt.order_by(Matter.created_at.desc(), Matter.id.desc()).limit(limit + 1)
    rows = list(await session.scalars(stmt))
    return build_page(rows, limit)


async def update_matter(
    session: AsyncSession, *, org_id: str, matter_id: str, changes: dict
) -> Matter:
    matter = await get_matter(session, org_id=org_id, matter_id=matter_id)
    # `changes` comes from `model_dump(exclude_unset=True)`, so only fields the
    # client actually sent are applied. Omitted != null.
    for field, value in changes.items():
        setattr(matter, field, value)
    await session.commit()
    await session.refresh(matter)
    return matter


async def delete_matter(session: AsyncSession, *, org_id: str, matter_id: str) -> None:
    matter = await get_matter(session, org_id=org_id, matter_id=matter_id)
    # Documents, chunks and questions go with it via ON DELETE CASCADE.
    # In a real legal product you would SOFT delete (set `deleted_at`) instead:
    # retention obligations and "undo" both require the rows to still exist.
    await session.delete(matter)
    await session.commit()
