"""
Document endpoints: upload (202 + async ingestion), list, update with
optimistic locking, delete.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status

from app.api.deps import (
    AdminUser,
    CurrentUser,
    IfMatchHeader,
    MemberUser,
    PageParams,
    RateLimited,
    SessionDep,
    SettingsDep,
)
from app.core.errors import PayloadTooLargeError, UnsupportedMediaTypeError, ValidationError
from app.schemas.common import PageResponse
from app.schemas.document import DocumentResponse, DocumentUpdate
from app.services import documents as document_service

router = APIRouter(tags=["documents"], dependencies=[RateLimited])


@router.post(
    "/matters/{matter_id}/documents",
    response_model=DocumentResponse,
    # 202 ACCEPTED, not 201. The row exists but the document is not yet
    # searchable - ingestion runs in the background. 201 would be a lie, and the
    # client would query for chunks that are not there yet. Status codes are a
    # contract, not decoration.
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document; ingestion continues in the background",
)
async def upload_document(
    matter_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    response: Response,
    user: MemberUser,
    file: Annotated[UploadFile, File(description="The document to ingest")],
    title: Annotated[str | None, Form(description="Defaults to the filename")] = None,
) -> DocumentResponse:
    content_type = file.content_type or "application/octet-stream"
    if content_type not in settings.allowed_upload_types:
        # ALLOW-LIST, never a deny-list. A deny-list is a list of the attacks you
        # already thought of; anything you forgot is permitted by default.
        raise UnsupportedMediaTypeError(
            f"'{content_type}' is not accepted. Allowed: {', '.join(settings.allowed_upload_types)}"
        )

    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        # NOTE: this checks AFTER buffering, which is the easy version. A
        # hardened service rejects on Content-Length up front and streams the
        # body to disk/S3 with a running byte cap, so a 10GB upload never
        # reaches memory at all. Real defence belongs at the reverse proxy too
        # (nginx client_max_body_size / ALB limits).
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_bytes} byte limit")
    if not payload:
        # 422, not 413: the request is well-formed and not too large, the file
        # simply has no content to ingest.
        raise ValidationError("Uploaded file is empty")

    document, created = await document_service.create_document(
        session,
        org_id=user.org_id,
        matter_id=matter_id,
        title=title or file.filename or "Untitled document",
        content_type=content_type,
        payload=payload,
    )

    if created:
        # Hand the id to the background worker. The row is already committed, so
        # the worker cannot race ahead of a transaction that has not landed.
        await request.app.state.ingest_queue.enqueue(document.id)
    else:
        # Byte-identical file already in this matter -> 200, not 202: nothing
        # new was accepted for processing.
        response.status_code = status.HTTP_200_OK

    # Location tells the client exactly where to poll for status. This is what
    # makes 202 usable rather than a shrug.
    response.headers["Location"] = f"/api/v1/documents/{document.id}"
    return DocumentResponse.model_validate(document)


@router.get(
    "/matters/{matter_id}/documents",
    response_model=PageResponse[DocumentResponse],
    summary="List documents in a matter",
)
async def list_documents(
    matter_id: str,
    session: SessionDep,
    user: CurrentUser,
    page: PageParams,
    doc_status: Annotated[
        str | None, Query(alias="status", description="pending|processing|ready|failed")
    ] = None,
) -> PageResponse[DocumentResponse]:
    result = await document_service.list_documents(
        session,
        org_id=user.org_id,
        matter_id=matter_id,
        limit=page.limit,
        cursor=page.cursor,
        status=doc_status,
    )
    return PageResponse[DocumentResponse](
        items=[DocumentResponse.model_validate(d) for d in result.items],
        next_cursor=result.next_cursor,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Fetch a document (poll this for ingestion status)",
)
async def get_document(
    document_id: str, session: SessionDep, user: CurrentUser, response: Response
) -> DocumentResponse:
    document = await document_service.get_document(
        session, org_id=user.org_id, document_id=document_id
    )
    # ETag = the current version. The client echoes it back in If-Match on the
    # next write, which is what makes optimistic locking work end to end.
    response.headers["ETag"] = f'"{document.version}"'
    return DocumentResponse.model_validate(document)


@router.patch(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Update a document; send If-Match to avoid clobbering someone's edit",
)
async def update_document(
    document_id: str,
    payload: DocumentUpdate,
    session: SessionDep,
    response: Response,
    user: MemberUser,
    if_match: IfMatchHeader = None,
) -> DocumentResponse:
    document = await document_service.update_document(
        session,
        org_id=user.org_id,
        document_id=document_id,
        changes=payload.model_dump(exclude_unset=True),
        expected_version=if_match,
    )
    response.headers["ETag"] = f'"{document.version}"'
    return DocumentResponse.model_validate(document)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks (admin only)",
)
async def delete_document(
    document_id: str,
    session: SessionDep,
    user: AdminUser,
) -> None:
    await document_service.delete_document(session, org_id=user.org_id, document_id=document_id)
