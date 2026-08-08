from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_user_id_var: ContextVar[str | None] = ContextVar(
    "globex_user_id",
    default=None,
)


def get_current_user_id() -> str | None:
    return _user_id_var.get()


@contextmanager
def bind_user_context(user_id: str | None) -> Iterator[None]:
    normalized = user_id.strip() if user_id else None
    token = _user_id_var.set(normalized or None)
    try:
        yield
    finally:
        _user_id_var.reset(token)


__all__ = ["bind_user_context", "get_current_user_id"]
