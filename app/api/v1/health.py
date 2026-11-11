"""
Health endpoints.

THREE PROBES, THREE DIFFERENT QUESTIONS - and conflating them causes outages:

  /health/live   "is the process alive?"        Kubernetes RESTARTS the pod if
                 this fails. It must therefore check NOTHING external. If you
                 check the database here, a brief DB blip restarts every pod in
                 the fleet simultaneously - you turn a small problem into a
                 total outage.

  /health/ready  "should traffic be routed here?"  Checks dependencies. Failing
                 removes the pod from the load balancer without killing it, so
                 it can recover and rejoin.

  /health        human-facing summary with version and queue depth.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app import __version__
from app.api.deps import SessionDep, SettingsDep
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe - no dependency checks")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe - checks dependencies")
async def ready(session: SessionDep, response: Response) -> dict[str, str]:
    try:
        # The cheapest possible round trip that proves the pool can hand out a
        # working connection and the server answers.
        await session.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not-ready", "database": "unreachable"}
    return {"status": "ready", "database": "ok"}


@router.get("/health", response_model=HealthResponse, summary="Human-readable health summary")
async def health(request: Request, settings: SettingsDep, session: SessionDep) -> HealthResponse:
    checks: dict[str, str] = {}
    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"

    checks["llm_provider"] = settings.llm_provider
    # A queue depth that only grows is your earliest warning that ingestion has
    # stalled - alert on this before users notice documents stuck in "pending".
    checks["ingest_queue_depth"] = str(request.app.state.ingest_queue.depth)

    return HealthResponse(
        status="ok" if checks["database"] == "ok" else "degraded",
        version=__version__,
        environment=settings.environment,
        checks=checks,
    )
