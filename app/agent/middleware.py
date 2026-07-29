from __future__ import annotations

from collections import deque
from hashlib import sha256
from typing import Any


MAX_TOOL_RESULT_CHARS = 16_000


def truncate_tool_result(
    value: Any,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
) -> Any:
    """Limit oversized tool observations before they re-enter the loop."""

    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    removed = len(value) - max_chars
    return (
        value[:max_chars]
        + f"\n\n[工具结果已截断，省略 {removed} 个字符]"
    )


class LoopDetected(RuntimeError):
    pass


class LoopDetector:
    """Detect repeated identical tool calls in a rolling window."""

    def __init__(
        self,
        window: int = 6,
        repeat_threshold: int = 4,
    ) -> None:
        self._calls: deque[str] = deque(
            maxlen=window
        )
        self.repeat_threshold = repeat_threshold

    def observe(
        self,
        tool_name: str,
        args: Any,
    ) -> None:
        fingerprint = sha256(
            f"{tool_name}:{args!r}".encode()
        ).hexdigest()
        self._calls.append(fingerprint)
        if (
            sum(
                call == fingerprint
                for call in self._calls
            )
            >= self.repeat_threshold
        ):
            raise LoopDetected(
                f"检测到重复工具调用：{tool_name}"
            )


__all__ = [
    "LoopDetected",
    "LoopDetector",
    "truncate_tool_result",
]
