from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from fastapi import WebSocket

from app.config import env_int


class ConnectionManager:
    """
    管理 thread_id 与 WebSocket 的对应关系。

    当前设计：
        一个 thread_id 只保留一个最新的 WebSocket。

    用户刷新页面时，新连接会覆盖旧连接。
    """

    def __init__(self) -> None:
        # thread_id -> 当前有效 WebSocket
        self.active: dict[str, WebSocket] = {}

        # 防止多个协程同时修改 active。
        self._lock = asyncio.Lock()

        # 保存最近事件，使 POST 创建任务后才建立的 WebSocket
        # 也能收到任务启动阶段产生的消息和最终结果。
        self._history: dict[
            str, deque[dict[str, Any]]
        ] = defaultdict(
            lambda: deque(
                maxlen=env_int(
                    "WS_EVENT_BUFFER_SIZE",
                    200,
                    minimum=1,
                )
            )
        )

    @staticmethod
    def _normalize_thread_id(
        thread_id: str,
    ) -> str:
        """
        清理并校验 thread_id。
        """

        normalized_thread_id = thread_id.strip()

        if not normalized_thread_id:
            raise ValueError(
                "thread_id 不能为空字符串。"
            )

        return normalized_thread_id

    async def connect(
        self,
        websocket: WebSocket,
        thread_id: str,
    ) -> None:
        """
        接受 WebSocket，并将其绑定到 thread_id。

        如果相同 thread_id 已经存在连接，
        新连接会覆盖旧连接。
        """

        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )

        await websocket.accept()

        # 先重放积压事件，再把连接标记为实时连接。循环读取可以
        # 覆盖重放期间新产生的事件，避免出现时间窗口。
        replayed = 0
        while True:
            async with self._lock:
                history = self._history.get(
                    normalized_thread_id
                )
                pending = (
                    list(history)[replayed:]
                    if history is not None
                    else []
                )
                if not pending:
                    self.active[
                        normalized_thread_id
                    ] = websocket
                    break

            for payload in pending:
                await websocket.send_json(payload)
            replayed += len(pending)

    async def disconnect(
        self,
        websocket: WebSocket,
        thread_id: str,
    ) -> None:
        """
        删除已经断开的 WebSocket。

        必须判断对象身份。

        原因：
            用户刷新页面后，新连接可能已经覆盖旧连接。
            此时旧连接迟到的 disconnect 事件，
            不能误删新连接。
        """

        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )

        async with self._lock:
            current_websocket = self.active.get(
                normalized_thread_id
            )

            if current_websocket is websocket:
                del self.active[
                    normalized_thread_id
                ]

    async def send_to_thread(
        self,
        payload: dict[str, Any],
        thread_id: str,
    ) -> bool:
        """
        把 JSON 消息发送给指定 thread_id。

        返回：
            发送成功返回 True。

            当前没有 WebSocket，或者发送失败，
            返回 False。
        """

        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )

        # 只在读取连接表时持有锁。
        #
        # 不要在发送网络消息期间一直占用锁，
        # 否则慢连接可能阻塞其他连接操作。
        async with self._lock:
            self._history[
                normalized_thread_id
            ].append(dict(payload))
            websocket = self.active.get(
                normalized_thread_id
            )

        if websocket is None:
            # 前端尚未连接，或者已经断开。
            # 监控事件允许被静默丢弃。
            return False

        try:
            await websocket.send_json(payload)
        except Exception:
            # 发送失败通常表示连接已经失效。
            await self.disconnect(
                websocket=websocket,
                thread_id=normalized_thread_id,
            )

            return False

        return True

    async def clear_history(
        self,
        thread_id: str,
    ) -> None:
        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )
        async with self._lock:
            self._history.pop(
                normalized_thread_id,
                None,
            )

    async def is_connected(
        self,
        thread_id: str,
    ) -> bool:
        """
        判断指定 thread_id 是否已有连接。

        主要用于测试和调试。
        """

        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )

        async with self._lock:
            return (
                normalized_thread_id
                in self.active
            )


# 全项目共享一个连接管理器。
manager = ConnectionManager()
