from __future__ import annotations

from typing import Any


def estimate_message_chars(
    message: Any,
) -> int:
    return len(str(getattr(message, "content", message)))


def compression_boundary(
    messages: list[Any],
    *,
    max_chars: int = 48_000,
    keep_recent: int = 8,
) -> int:
    """Return the exclusive index of old messages to summarize."""

    if len(messages) <= keep_recent:
        return 0
    total = sum(
        estimate_message_chars(message)
        for message in messages
    )
    if total <= max_chars:
        return 0

    boundary = max(0, len(messages) - keep_recent)
    while (
        boundary > 0
        and sum(
            estimate_message_chars(message)
            for message in messages[boundary:]
        )
        < max_chars // 2
    ):
        boundary -= 1
    return boundary


__all__ = [
    "compression_boundary",
    "estimate_message_chars",
]
