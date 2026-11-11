"""Matter endpoints: tenant-scoped CRUD, cursor pagination, idempotent create."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.deps import (
    AdminUser,
    CurrentUser,
    IdempotencyDep,
    IdempotencyKeyHeader,
    MemberUser,
    PageParams,
    RateLimited,
    SessionDep,
)
from app.schemas.common import PageResponse
from app.schemas.matter import MatterCreate, MatterResponse, MatterUpdate
from app.services import matters as matter_service

router = APIRouter(prefix="/matters", tags=["matters"], dependencies=[RateLimited])


@router.post(
    "",
    response_model=MatterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a matter (safe to retry with an Idempotency-Key)",
)
async def create_matter(
    payload: MatterCreate,
    session: SessionDep,
    idempotency: IdempotencyDep,
    response: Response,
    user: MemberUser,
    idempotency_key: IdempotencyKeyHeader = None,
) -> MatterResponse:
    body = payload.model_dump()

    # Replay? Return exactly what we returned the first time - no second matter.
    cached = await idempotency.lookup(idempotency_key, "POST /matters", body)
    if cached is not None:
        cached_status, cached_body = cached
        response.status_code = cached_status
        # Tell the client this was a replay; invaluable when debugging retries.
        response.headers["Idempotent-Replay"] = "true"
        return MatterResponse.model_validate(cached_body)

    matter = await matter_service.create_matter(
        session,
        org_id=user.org_id,
        user_id=user.id,
        title=payload.title,
        client_reference=payload.client_reference,
    )
    result = MatterResponse.model_validate(matter)

    await idempotency.store(
        idempotency_key,
        "POST /matters",
        body,
        status.HTTP_201_CREATED,
        # mode="json" so datetimes become ISO strings the JSON column can hold.
        result.model_dump(mode="json"),
    )
    return result


@router.get("", response_model=PageResponse[MatterResponse], summary="List matters")
async def list_matters(
    session: SessionDep,
    user: CurrentUser,
    page: PageParams,
    search: Annotated[
        str | None, Query(max_length=200, description="Case-insensitive title match")
    ] = None,
) -> PageResponse[MatterResponse]:
    result = await matter_service.list_matters(
        session, org_id=user.org_id, limit=page.limit, cursor=page.cursor, search=search
    )
    return PageResponse[MatterResponse](
        items=[MatterResponse.model_validate(m) for m in result.items],
        next_cursor=result.next_cursor,
    )


@router.get("/{matter_id}", response_model=MatterResponse, summary="Fetch one matter")
async def get_matter(matter_id: str, session: SessionDep, user: CurrentUser) -> MatterResponse:
    matter = await matter_service.get_matter(session, org_id=user.org_id, matter_id=matter_id)
    return MatterResponse.model_validate(matter)


@router.patch(
    "/{matter_id}",
    response_model=MatterResponse,
    summary="Partially update a matter",
)
async def update_matter(
    matter_id: str,
    payload: MatterUpdate,
    session: SessionDep,
    user: MemberUser,
) -> MatterResponse:
    # PATCH, not PUT: exclude_unset means we touch only the fields the client
    # actually sent. A PUT would require the full representation and would
    # silently null out anything omitted.
    matter = await matter_service.update_matter(
        session,
        org_id=user.org_id,
        matter_id=matter_id,
        changes=payload.model_dump(exclude_unset=True),
    )
    return MatterResponse.model_validate(matter)


@router.delete(
    "/{matter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a matter and everything in it (admin only)",
)
async def delete_matter(
    matter_id: str,
    session: SessionDep,
    user: AdminUser,
) -> None:
    await matter_service.delete_matter(session, org_id=user.org_id, matter_id=matter_id)
