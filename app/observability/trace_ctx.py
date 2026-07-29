from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any


_trace_var: ContextVar[Any | None] = ContextVar(
    "globex_langfuse_trace",
    default=None,
)
_span_var: ContextVar[Any | None] = ContextVar(
    "globex_langfuse_span",
    default=None,
)


def set_langfuse_trace(trace: Any) -> Token:
    return _trace_var.set(trace)


def get_langfuse_trace() -> Any | None:
    return _trace_var.get()


def reset_langfuse_trace(token: Token) -> None:
    _trace_var.reset(token)


def set_current_span(span: Any) -> Token:
    return _span_var.set(span)


def get_current_span() -> Any | None:
    return _span_var.get()


def reset_current_span(token: Token) -> None:
    _span_var.reset(token)


__all__ = [
    "get_current_span",
    "get_langfuse_trace",
    "reset_current_span",
    "reset_langfuse_trace",
    "set_current_span",
    "set_langfuse_trace",
]
