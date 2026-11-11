"""
Question answering endpoints - the core feature: grounded answers with
citations back to specific passages.

Two shapes for the same pipeline:
  POST /matters/{id}/questions         -> JSON, wait for the whole answer
  POST /matters/{id}/questions/stream  -> Server-Sent Events, token by token
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse

from app.api.deps import (
    CurrentUser,
    LLMDep,
    MemberUser,
    PageParams,
    RateLimited,
    SessionDep,
)
from app.core.logging import get_logger
from app.schemas.common import PageResponse
from app.schemas.qa import AnswerResponse, AskRequest
from app.services import qa as qa_service

logger = get_logger(__name__)
router = APIRouter(tags=["questions"], dependencies=[RateLimited])


def _sse(event: str, data: dict) -> str:
    """
    Format one Server-Sent Event.

    The wire format is deliberately trivial:
        event: delta\\n
        data: {"text":"..."}\\n
        \\n                      <- the BLANK LINE terminates the event

    Two things people get wrong:
      * forgetting the trailing blank line - the browser buffers forever;
      * putting a raw newline inside `data:` - it must be one line per data
        field, which is why we json.dumps (it escapes newlines for us).
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post(
    "/matters/{matter_id}/questions",
    response_model=AnswerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ask a question and wait for the complete answer",
)
async def ask(
    matter_id: str,
    payload: AskRequest,
    session: SessionDep,
    llm: LLMDep,
    user: MemberUser,
) -> AnswerResponse:
    question = await qa_service.answer_question(
        session,
        llm,
        org_id=user.org_id,
        matter_id=matter_id,
        user_id=user.id,
        text=payload.question,
        top_k=payload.top_k,
    )
    return AnswerResponse.model_validate(question)


@router.post(
    "/matters/{matter_id}/questions/stream",
    summary="Ask a question and stream the answer as Server-Sent Events",
    # response_class rather than response_model: the body is a stream, so there
    # is no single schema for FastAPI to validate or document.
    response_class=StreamingResponse,
)
async def ask_streaming(
    matter_id: str,
    payload: AskRequest,
    request: Request,
    session: SessionDep,
    llm: LLMDep,
    user: MemberUser,
) -> StreamingResponse:
    async def event_source() -> AsyncIterator[str]:
        try:
            async for event, data in qa_service.stream_question_answer(
                session,
                llm,
                org_id=user.org_id,
                matter_id=matter_id,
                user_id=user.id,
                text=payload.question,
                top_k=payload.top_k,
            ):
                # BACKPRESSURE / CANCELLATION: if the user closes the tab, keep
                # streaming and you burn model tokens for nobody. Checking
                # is_disconnected() lets us stop early. (Starlette also throws
                # into the generator on disconnect; this is the explicit belt.)
                if await request.is_disconnected():
                    logger.info("sse_client_disconnected", extra={"matter_id": matter_id})
                    break
                yield _sse(event, data)
        except Exception as exc:
            # The 200 status line is long gone, so we cannot switch to a 500.
            # The only honest option is an in-band error event.
            logger.exception("sse_stream_failed")
            yield _sse("error", {"message": "Stream failed", "detail": str(exc)[:200]})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            # Do not let any proxy buffer the stream - buffering turns a
            # token-by-token feed back into one big blob at the end.
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx-specific: disable output buffering
        },
    )


@router.get(
    "/matters/{matter_id}/questions",
    response_model=PageResponse[AnswerResponse],
    summary="Question history for a matter (the audit trail)",
)
async def list_questions(
    matter_id: str, session: SessionDep, user: CurrentUser, page: PageParams
) -> PageResponse[AnswerResponse]:
    result = await qa_service.list_questions(
        session, org_id=user.org_id, matter_id=matter_id, limit=page.limit, cursor=page.cursor
    )
    return PageResponse[AnswerResponse](
        items=[AnswerResponse.model_validate(q) for q in result.items],
        next_cursor=result.next_cursor,
    )


@router.get(
    "/questions/{question_id}",
    response_model=AnswerResponse,
    summary="Fetch one question with its answer and citations",
)
async def get_question(question_id: str, session: SessionDep, user: CurrentUser) -> AnswerResponse:
    question = await qa_service.get_question(session, org_id=user.org_id, question_id=question_id)
    return AnswerResponse.model_validate(question)
