from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float
    thread_id: str = field(compare=False)
    query: str = field(compare=False)
    user_id: str | None = field(
        compare=False, default=None
    )
    payload: Any = field(
        compare=False, default=None
    )


class PriorityRequestQueue:
    def __init__(
        self,
        normal_workers: int = 8,
        heavy_workers: int = 4,
    ) -> None:
        self.normal_queue: list[
            PrioritizedRequest
        ] = []
        self.heavy_queue: list[
            PrioritizedRequest
        ] = []
        self._normal_capacity = normal_workers
        self._heavy_capacity = heavy_workers
        self._normal_sem = asyncio.Semaphore(
            normal_workers
        )
        self._heavy_sem = asyncio.Semaphore(
            heavy_workers
        )
        self._lock = asyncio.Lock()

    async def put(
        self,
        request: PrioritizedRequest,
        *,
        heavy: bool = False,
    ) -> None:
        async with self._lock:
            heapq.heappush(
                self.heavy_queue
                if heavy
                else self.normal_queue,
                request,
            )

    async def get(
        self, *, heavy: bool = False
    ) -> PrioritizedRequest:
        queue = (
            self.heavy_queue
            if heavy
            else self.normal_queue
        )
        while True:
            async with self._lock:
                if queue:
                    return heapq.heappop(queue)
            await asyncio.sleep(0.01)

    async def dynamic_rebalance(self) -> None:
        # Semaphores are replaced only when their queues are idle.
        if (
            not self.normal_queue
            and len(self.heavy_queue) > 10
        ):
            self._heavy_capacity = 6
            self._heavy_sem = asyncio.Semaphore(6)
        elif not self.heavy_queue:
            self._heavy_capacity = 4
            self._heavy_sem = asyncio.Semaphore(4)


request_queue = PriorityRequestQueue()


__all__ = [
    "PrioritizedRequest",
    "PriorityRequestQueue",
    "request_queue",
]
