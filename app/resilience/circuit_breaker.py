from __future__ import annotations

import asyncio
import time
from collections import deque
from enum import Enum
from typing import Any, Awaitable, Callable


class CircuitOpenError(RuntimeError):
    pass


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        tool_name: str,
        failure_threshold: float = 0.30,
        window_size: int = 100,
        recovery_timeout: float = 300,
    ) -> None:
        self.tool_name = tool_name
        self.failure_threshold = (
            failure_threshold
        )
        self.window_size = window_size
        self.recovery_timeout = (
            recovery_timeout
        )
        self._results: deque[bool] = deque(
            maxlen=window_size
        )
        self._state = State.CLOSED
        self._open_since = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> State:
        if (
            self._state == State.OPEN
            and time.monotonic()
            - self._open_since
            >= self.recovery_timeout
        ):
            self._state = State.HALF_OPEN
        return self._state

    @property
    def failure_rate(self) -> float:
        if not self._results:
            return 0.0
        return 1 - (
            sum(self._results) / len(self._results)
        )

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async with self._lock:
            if self.state == State.OPEN:
                raise CircuitOpenError(
                    f"{self.tool_name} 熔断中"
                )
        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._record(False)
            raise
        await self._record(True)
        return result

    async def _record(
        self, succeeded: bool
    ) -> None:
        async with self._lock:
            self._results.append(succeeded)
            if self._state == State.HALF_OPEN:
                if succeeded:
                    self._state = State.CLOSED
                    self._results.clear()
                else:
                    self._trip()
                return
            if (
                len(self._results)
                >= min(10, self.window_size)
                and self.failure_rate
                >= self.failure_threshold
            ):
                self._trip()

    def _trip(self) -> None:
        self._state = State.OPEN
        self._open_since = time.monotonic()


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "State",
]
