from typing import Any

import pytest

from app.api.connection import (
    ConnectionManager,
)


class FakeWebSocket:
    """
    测试使用的假 WebSocket。

    不需要启动真正的 FastAPI 服务。
    """

    def __init__(
        self,
        *,
        should_fail: bool = False,
    ) -> None:
        self.accepted = False
        self.should_fail = should_fail

        self.sent_payloads: list[
            dict[str, Any]
        ] = []

    async def accept(self) -> None:
        """
        模拟接受连接。
        """

        self.accepted = True

    async def send_json(
        self,
        payload: dict[str, Any],
    ) -> None:
        """
        模拟发送 JSON。
        """

        if self.should_fail:
            raise RuntimeError(
                "模拟 WebSocket 已断开。"
            )

        self.sent_payloads.append(payload)


@pytest.mark.asyncio
async def test_connect_and_send() -> None:
    """
    验证连接登记和消息发送。
    """

    manager = ConnectionManager()
    websocket = FakeWebSocket()

    await manager.connect(
        websocket=websocket,  # type: ignore[arg-type]
        thread_id="thread-A",
    )

    assert websocket.accepted is True

    assert await manager.is_connected(
        "thread-A"
    )

    sent = await manager.send_to_thread(
        payload={
            "event": "tool_start",
        },
        thread_id="thread-A",
    )

    assert sent is True

    assert websocket.sent_payloads == [
        {
            "event": "tool_start",
        }
    ]


@pytest.mark.asyncio
async def test_missing_connection_is_ignored() -> None:
    """
    没有 WebSocket 时应返回 False，
    而不是抛出异常。
    """

    manager = ConnectionManager()

    sent = await manager.send_to_thread(
        payload={
            "event": "tool_start",
        },
        thread_id="thread-missing",
    )

    assert sent is False


@pytest.mark.asyncio
async def test_old_disconnect_does_not_remove_new_connection() -> None:
    """
    验证页面重连后：

    旧连接迟到的断开事件，
    不会删除新连接。
    """

    manager = ConnectionManager()

    old_websocket = FakeWebSocket()
    new_websocket = FakeWebSocket()

    await manager.connect(
        websocket=old_websocket,  # type: ignore[arg-type]
        thread_id="thread-A",
    )

    # 模拟页面刷新，新连接覆盖旧连接。
    await manager.connect(
        websocket=new_websocket,  # type: ignore[arg-type]
        thread_id="thread-A",
    )

    # 旧连接现在才发出断开事件。
    await manager.disconnect(
        websocket=old_websocket,  # type: ignore[arg-type]
        thread_id="thread-A",
    )

    # 新连接仍然存在。
    assert await manager.is_connected(
        "thread-A"
    )

    await manager.send_to_thread(
        payload={
            "message": "发送给新连接",
        },
        thread_id="thread-A",
    )

    assert old_websocket.sent_payloads == []

    assert new_websocket.sent_payloads == [
        {
            "message": "发送给新连接",
        }
    ]


@pytest.mark.asyncio
async def test_send_failure_removes_connection() -> None:
    """
    验证发送失败后会清理失效连接。
    """

    manager = ConnectionManager()

    websocket = FakeWebSocket(
        should_fail=True
    )

    await manager.connect(
        websocket=websocket,  # type: ignore[arg-type]
        thread_id="thread-A",
    )

    sent = await manager.send_to_thread(
        payload={
            "event": "tool_end",
        },
        thread_id="thread-A",
    )

    assert sent is False

    assert not await manager.is_connected(
        "thread-A"
    )