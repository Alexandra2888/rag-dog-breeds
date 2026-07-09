"""Structured logging setup (structlog) shared by every entrypoint.

`setup_logging()` routes *all* stdlib logging (the existing
`logging.getLogger(__name__)` calls across the codebase) through a single
structlog `ProcessorFormatter`, so nothing at the call sites needs to change.
Output is one JSON object per line when `LOG_JSON=true` (prod), or a colorized
console format in dev.

A `request_id` is carried in a contextvar via `structlog.contextvars`, so once the
FastAPI middleware binds it, every log line emitted while handling that request —
including logs from database.py / rag_service.py / embeddings.py — carries the id.
"""
from __future__ import annotations

import logging
import sys

import structlog

from src.config import settings

_configured = False


def setup_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configure structlog + stdlib once. Idempotent."""
    global _configured
    level = (level or settings.log_level).upper()
    json_logs = settings.log_json if json_logs is None else json_logs

    # Processors shared by structlog-native loggers and foreign (stdlib) records,
    # so both carry request_id, level, logger name, and an ISO timestamp.
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = (
        structlog.processors.JSONRenderer() if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Trim third-party chatter so our lines stay readable.
    for noisy in ("httpx", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None):
    """A structlog logger (accepts structured kwargs). Sets up logging on first use."""
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)


def bind_request_id(request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
