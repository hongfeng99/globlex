from __future__ import annotations

from typing import Any

from app.observability.trace_ctx import (
    get_langfuse_trace,
)


def create_langfuse_handler(
    thread_id: str,
) -> Any | None:
    trace = get_langfuse_trace()
    if trace is None:
        return None
    try:
        from langfuse.callback import (
            CallbackHandler,
        )

        return CallbackHandler(
            trace_id=getattr(
                trace, "id", thread_id
            ),
            session_id=thread_id,
        )
    except (ImportError, TypeError):
        return None


__all__ = ["create_langfuse_handler"]
