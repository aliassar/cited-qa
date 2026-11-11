"""
Error model and exception handlers.

THE PRINCIPLE
-------------
Domain code raises *domain* exceptions (`NotFoundError`, `ConflictError`). It
does not know or care about HTTP. A thin translation layer at the edge maps
those to status codes. This keeps services reusable (a CLI or a worker can call
the same service without dragging HTTPException along) and keeps the HTTP
contract in one auditable place.

THE WIRE FORMAT: RFC 9457 "Problem Details"
-------------------------------------------
Every error response has the same shape:

    {
      "type":   "https://errors.cited-qa.example/conflict",
      "title":  "Conflict",
      "status": 409,
      "detail": "Document was modified by someone else",
      "request_id": "9f2c...",
      "errors": [ ... ]          # only for validation failures
    }

A stable machine-readable envelope means the frontend writes one error handler
instead of one per endpoint, and support can search by request_id.

SECURITY NOTE
-------------
`detail` is written for the caller and must never leak internals (SQL, stack
traces, file paths). Unexpected exceptions are logged in full server-side and
returned as a generic 500.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)

# Spelled out rather than taken from `starlette.status`: the constant names
# for these two were renamed upstream, and hard-coding the number keeps the
# project working across Starlette versions. The numbers themselves are the
# stable part of HTTP; the constant names are not.
HTTP_422_UNPROCESSABLE_CONTENT = 422
HTTP_413_CONTENT_TOO_LARGE = 413

_ERROR_BASE = "https://errors.cited-qa.example"


class AppError(Exception):
    """Base class for every expected, *handled* failure in the domain."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    title: str = "Internal Server Error"
    slug: str = "internal-error"

    def __init__(self, detail: str | None = None, **extra: Any) -> None:
        self.detail = detail or self.title
        self.extra = extra  # optional structured payload (e.g. current_version)
        super().__init__(self.detail)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    title = "Not Found"
    slug = "not-found"


class PermissionDeniedError(AppError):
    # 403 = "we know who you are, you may not do this".
    status_code = status.HTTP_403_FORBIDDEN
    title = "Forbidden"
    slug = "forbidden"


class AuthenticationError(AppError):
    # 401 = "we do not know who you are / your token is bad".
    status_code = status.HTTP_401_UNAUTHORIZED
    title = "Unauthorized"
    slug = "unauthorized"


class ConflictError(AppError):
    # 409 = the request is valid but conflicts with current state.
    # Used for optimistic-locking version mismatches and duplicate keys.
    status_code = status.HTTP_409_CONFLICT
    title = "Conflict"
    slug = "conflict"


class ValidationError(AppError):
    # 422 = syntactically fine, semantically invalid (business rule violated).
    status_code = HTTP_422_UNPROCESSABLE_CONTENT
    title = "Unprocessable Entity"
    slug = "validation-error"


class RateLimitedError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    title = "Too Many Requests"
    slug = "rate-limited"

    def __init__(self, detail: str | None = None, retry_after: int = 1) -> None:
        super().__init__(detail)
        self.retry_after = retry_after


class PayloadTooLargeError(AppError):
    status_code = HTTP_413_CONTENT_TOO_LARGE
    title = "Payload Too Large"
    slug = "payload-too-large"


class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    title = "Unsupported Media Type"
    slug = "unsupported-media-type"


def _problem(
    status_code: int,
    title: str,
    slug: str,
    detail: str,
    **extra: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"{_ERROR_BASE}/{slug}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "request_id": request_id_ctx.get(),
    }
    body.update({k: v for k, v in extra.items() if v is not None})
    # The RFC-registered content type. Clients can branch on it.
    return JSONResponse(
        status_code=status_code, content=body, media_type="application/problem+json"
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Wire the handlers onto the app. Called once from the app factory."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        headers = {}
        if isinstance(exc, RateLimitedError):
            # Tell a well-behaved client exactly how long to back off.
            headers["Retry-After"] = str(exc.retry_after)
        response = _problem(exc.status_code, exc.title, exc.slug, exc.detail, **exc.extra)
        response.headers.update(headers)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI raises this when Pydantic rejects the request body/query/path.
        # We reshape it into our envelope so clients see ONE error format.
        errors = [
            {
                # loc looks like ("body", "title"); drop the first segment for readability
                "field": ".".join(str(p) for p in err["loc"][1:]) or str(err["loc"][0]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
        return _problem(
            HTTP_422_UNPROCESSABLE_CONTENT,
            "Unprocessable Entity",
            "validation-error",
            "Request validation failed",
            errors=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Covers 404 on unknown routes, 405 method-not-allowed, and any
        # HTTPException raised by third-party middleware.
        return _problem(
            exc.status_code,
            exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            "http-error",
            exc.detail if isinstance(exc.detail, str) else "HTTP Error",
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # LAST RESORT. Log everything, tell the client nothing.
        logger.exception("unhandled_exception", extra={"error_type": type(exc).__name__})
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal Server Error",
            "internal-error",
            "An unexpected error occurred. Quote the request_id when reporting this.",
        )
