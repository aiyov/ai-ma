"""Structured event logs with request id propagation."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
logger = logging.getLogger("ai_ma")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


def configure(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s rid=%(request_id)s %(message)s"))
    handler.addFilter(_RequestIdFilter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)


def event(kind: str, **fields: Any) -> None:
    parts = [f"{key}={_format(value)}" for key, value in fields.items() if value is not None]
    logger.info("%s %s", kind, " ".join(parts))


@contextmanager
def timed(kind: str, **fields: Any) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    finally:
        event(kind, duration_ms=round((time.perf_counter() - started) * 1000), **fields, **extra)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ",".join(_format(item) for item in value)
    text = str(value).replace(" ", "_")
    return text[:120]
