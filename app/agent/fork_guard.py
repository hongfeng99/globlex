from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


class ForkLimitExceeded(RuntimeError):
    pass


_fork_depth: ContextVar[int] = ContextVar(
    "globex_fork_depth",
    default=0,
)


def get_fork_depth() -> int:
    return _fork_depth.get()


@contextmanager
def enter_fork(
    max_depth: int = 2,
) -> Iterator[int]:
    current = get_fork_depth()
    if current >= max_depth:
        raise ForkLimitExceeded(
            f"fork 深度已达到上限 {max_depth}"
        )

    token = _fork_depth.set(current + 1)
    try:
        yield current + 1
    finally:
        _fork_depth.reset(token)


__all__ = [
    "ForkLimitExceeded",
    "enter_fork",
    "get_fork_depth",
]
