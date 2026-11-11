"""
Structured logging + request correlation.

WHY STRUCTURED (JSON) LOGS
--------------------------
`logger.info("user %s uploaded %s", uid, name)` produces a string a human can
read and a machine cannot query. In production you ship logs to Datadog / Loki /
CloudWatch and you want to ask "show me every 5xx for org X in the last hour".
That requires *fields*, not prose. So we emit one JSON object per line.

WHY A REQUEST ID
----------------
A single user action can fan out across middleware, a DB call, a background
task and an LLM call. A correlation id (`request_id`) threaded through all of
them is what turns "something broke" into "here are the 14 log lines for that
exact request". We store it in a ContextVar so it is available anywhere in the
call stack without passing it through every function signature.

ContextVar is the asyncio-safe equivalent of thread-local storage: each task
gets its own view, so two concurrent requests never see each other's id.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

# The ContextVar itself. Default "-" so logging never crashes outside a request.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }

        # Anything passed via logger.info("...", extra={"key": value}) shows up
        # as an attribute on the record. We copy the non-standard ones across so
        # callers can attach structured context freely.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # default=str so a stray UUID/datetime never blows up the log pipeline.
        return json.dumps(payload, default=str)


# Attributes that logging puts on every record; we do not want them duplicated.
_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Install our formatter on the root logger. Call once, at startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers = [handler]  # replace, do not append (avoids duplicate lines)
    root.setLevel(level)

    # Uvicorn installs its own handlers; disable propagation-duplication by
    # letting them bubble up to our root handler instead.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
