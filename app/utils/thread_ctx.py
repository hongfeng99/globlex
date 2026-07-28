from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.api.context import (
    get_session_dir,
    get_thread_id,
    require_session_dir,
    require_thread_id,
    reset_thread_context,
    set_thread_context,
)


@contextmanager
def bind_thread_context(
    thread_id: str,
    session_dir: Path,
) -> Iterator[None]:
    """
    在指定代码范围内绑定请求上下文。

    离开 with 代码块时，无论正常结束还是发生异常，
    都会自动恢复之前的上下文。

    使用方式：

        with bind_thread_context(
            thread_id="task-001",
            session_dir=Path("output/task-001"),
        ):
            current_thread_id = require_thread_id()
    """

    tokens = set_thread_context(
        thread_id=thread_id,
        session_dir=session_dir,
    )

    try:
        yield
    finally:
        reset_thread_context(tokens)


__all__ = [
    "bind_thread_context",
    "get_thread_id",
    "get_session_dir",
    "require_thread_id",
    "require_session_dir",
]