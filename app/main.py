"""
Application factory, lifespan, and middleware.

WHY A FACTORY (`create_app()`) INSTEAD OF A MODULE-LEVEL `app = FastAPI()`
---------------------------------------------------------------------------
Tests need their OWN app: different settings, a throwaway database, a fake LLM.
A module-level singleton forces monkeypatching and leaks state between tests.
A factory means every test builds a clean, isolated instance.

MIDDLEWARE ORDER
----------------
Starlette runs middleware in the order added, OUTERMOST FIRST on the way in and
in reverse on the way out. So the request-id middleware is added first: it must
wrap everything, including the error handlers, or errors would be logged with no
correlation id - exactly when you need one most.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import install_exception_handlers
from app.core.logging import configure_logging, get_logger, new_request_id, request_id_ctx
from app.core.rate_limit import TokenBucketLimiter
from app.db.base import Base
from app.db.session import create_engine, create_session_factory
from app.services.ingestion import IngestionQueue
from app.services.llm import build_llm_client

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(
        level="DEBUG" if settings.debug else "INFO",
        json_logs=settings.environment != "local",  # readable logs while developing
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Startup and shutdown, in one place.

        Everything expensive and long-lived is built ONCE here and stashed on
        `app.state`: the DB engine (holds a connection pool), the LLM client
        (holds an HTTP pool), the rate limiter, the worker pool. Building any of
        these per request would mean a new TCP/TLS handshake every time.

        The code after `yield` is the shutdown half. Draining the queue and
        disposing the engine there is what makes a deploy graceful instead of
        abrupt.
        """
        engine = create_engine(settings)
        session_factory = create_session_factory(engine)

        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.rate_limiter = TokenBucketLimiter(
            capacity=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        app.state.llm_client = build_llm_client(settings)

        # create_all is a DEVELOPMENT convenience. Production schema changes go
        # through Alembic (see alembic/), which is reviewable, ordered and
        # reversible. Never point create_all at a production database.
        if settings.environment in ("local", "test"):
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

        app.state.ingest_queue = IngestionQueue(
            session_factory,
            worker_count=settings.ingest_worker_count,
            maxsize=settings.ingest_queue_maxsize,
        )
        await app.state.ingest_queue.start()

        logger.info("app_started", extra={"environment": settings.environment})
        yield
        # ---- shutdown ----
        await app.state.ingest_queue.stop()
        await engine.dispose()  # close every pooled connection
        logger.info("app_stopped")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "A legal document workspace API: upload documents, ingest them into "
            "citable passages, and ask grounded questions with sources."
        ),
        lifespan=lifespan,
        root_path=settings.root_path,
        # Hide interactive docs in production. They are a free map of your API
        # surface for anyone probing it; keep them behind your VPN or off.
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None if settings.environment == "production" else "/redoc",
    )

    # ---------------------------------------------------------------- middleware

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """
        Assign a request id, time the request, log the outcome.

        We honour an inbound `X-Request-ID` so a trace started at the gateway (or
        by the frontend) continues here instead of being renamed at every hop.
        """
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        finally:
            # Reset the ContextVar even on failure, so the value cannot leak
            # into whatever task reuses this context next.
            request_id_ctx.reset(token)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        # Server-Timing shows up in the browser devtools waterfall for free.
        response.headers["Server-Timing"] = f"app;dur={duration_ms:.1f}"

        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                # route.path, not the raw URL: "/api/v1/matters/{matter_id}"
                # groups sensibly in metrics, whereas the raw path creates one
                # unbounded time series per id (a classic Prometheus cardinality
                # explosion). `route` is absent for 404s, hence the fallback.
                "path": getattr(request.scope.get("route"), "path", request.url.path),
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id,
            },
        )
        return response

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Cheap, high-value hardening headers."""
        response = await call_next(request)
        # Stop browsers from MIME-sniffing a JSON response into something executable.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"  # clickjacking
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.scheme == "https":
            # HSTS: only ever meaningful over HTTPS.
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    app.add_middleware(
        CORSMiddleware,
        # An EXPLICIT origin list. `allow_origins=["*"]` together with
        # allow_credentials=True is rejected by browsers anyway, and a wildcard
        # on a credentialed API is a serious misconfiguration.
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "If-Match",
            "X-Request-ID",
        ],
        # Without expose_headers the browser hides these from JavaScript, and
        # your optimistic-locking client can never read the ETag it needs.
        expose_headers=["X-Request-ID", "ETag", "Location", "Idempotent-Replay"],
    )

    # minimum_size avoids the CPU cost of compressing tiny payloads, where the
    # gzip header can make the response bigger than the body.
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    install_exception_handlers(app)
    app.include_router(api_router)
    return app


# The ASGI entrypoint uvicorn imports: `uvicorn app.main:app`.
app = create_app()
