from pathlib import Path
from typing import Any

import pytest

import app.api.monitor as monitor_module
from app.api.monitor import Monitor
from app.utils.thread_ctx import (
    bind_thread_context,
)


@pytest.mark.asyncio
async def test_monitor_ignores_missing_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    没有 thread_id 上下文时，
    Monitor 应静默跳过发送。
    """

    async def fake_send_to_thread(
        payload: dict[str, Any],
        thread_id: str,
    ) -> bool:
        raise AssertionError(
            "没有 thread_id 时不应该发送消息。"
        )

    monkeypatch.setattr(
        monitor_module.manager,
        "send_to_thread",
        fake_send_to_thread,
    )

    current_monitor = Monitor()

    sent = await current_monitor.report_tool_start(
        tool_name="item_search",
        args={
            "query": "无线耳机",
        },
    )

    assert sent is False


@pytest.mark.asyncio
async def test_report_tool_start_uses_current_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 tool_start 会自动使用当前 ContextVar
    中保存的 thread_id。
    """

    recorded_messages: list[
        tuple[str, dict[str, Any]]
    ] = []

    async def fake_send_to_thread(
        payload: dict[str, Any],
        thread_id: str,
    ) -> bool:
        recorded_messages.append(
            (
                thread_id,
                payload,
            )
        )

        return True

    monkeypatch.setattr(
        monitor_module.manager,
        "send_to_thread",
        fake_send_to_thread,
    )

    current_monitor = Monitor()

    with bind_thread_context(
        thread_id="thread-A",
        session_dir=tmp_path / "thread-A",
    ):
        sent = (
            await current_monitor.report_tool_start(
                tool_name="item_search",
                args={
                    "query": "无线耳机",
                    "platform": "amazon",
                },
            )
        )

    assert sent is True
    assert len(recorded_messages) == 1

    target_thread_id, payload = (
        recorded_messages[0]
    )

    assert target_thread_id == "thread-A"

    assert payload["type"] == (
        "monitor_event"
    )
    assert payload["event"] == "tool_start"
    assert payload["message"] == (
        "正在调用 item_search"
    )
    assert payload["thread_id"] == (
        "thread-A"
    )

    assert payload["data"] == {
        "tool_name": "item_search",
        "args": {
            "query": "无线耳机",
            "platform": "amazon",
        },
    }

    assert isinstance(
        payload["timestamp"],
        str,
    )


@pytest.mark.asyncio
async def test_report_tool_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证工具结束事件包含执行耗时。
    """

    recorded_payloads: list[
        dict[str, Any]
    ] = []

    async def fake_send_to_thread(
        payload: dict[str, Any],
        thread_id: str,
    ) -> bool:
        recorded_payloads.append(payload)
        return True

    monkeypatch.setattr(
        monitor_module.manager,
        "send_to_thread",
        fake_send_to_thread,
    )

    current_monitor = Monitor()

    with bind_thread_context(
        thread_id="thread-B",
        session_dir=tmp_path / "thread-B",
    ):
        sent = (
            await current_monitor.report_tool_end(
                tool_name="web_search",
                duration_ms=125,
            )
        )

    assert sent is True

    payload = recorded_payloads[0]

    assert payload["event"] == "tool_end"
    assert payload["message"] == (
        "web_search 完成"
    )

    assert payload["data"] == {
        "tool_name": "web_search",
        "duration_ms": 125,
    }


@pytest.mark.asyncio
async def test_report_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 fork 事件包含子 Agent 的 thread_id。
    """

    recorded_payloads: list[
        dict[str, Any]
    ] = []

    async def fake_send_to_thread(
        payload: dict[str, Any],
        thread_id: str,
    ) -> bool:
        recorded_payloads.append(payload)
        return True

    monkeypatch.setattr(
        monitor_module.manager,
        "send_to_thread",
        fake_send_to_thread,
    )

    current_monitor = Monitor()

    demands = "请分别搜索四个平台的无线耳机"

    with bind_thread_context(
        thread_id="main-thread",
        session_dir=tmp_path / "main-thread",
    ):
        sent = await current_monitor.report_fork(
            sub_thread_id="sub-12345678",
            demands=demands,
        )

    assert sent is True

    payload = recorded_payloads[0]

    assert payload["event"] == "fork"

    assert payload["data"] == {
        "sub_thread_id": "sub-12345678",
        "demands": demands,
    }


@pytest.mark.asyncio
async def test_report_task_result_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证任务结果事件和错误事件。
    """

    recorded_payloads: list[
        dict[str, Any]
    ] = []

    async def fake_send_to_thread(
        payload: dict[str, Any],
        thread_id: str,
    ) -> bool:
        recorded_payloads.append(payload)
        return True

    monkeypatch.setattr(
        monitor_module.manager,
        "send_to_thread",
        fake_send_to_thread,
    )

    current_monitor = Monitor()

    with bind_thread_context(
        thread_id="thread-result",
        session_dir=tmp_path / "thread-result",
    ):
        await current_monitor.report_task_result(
            final_answer="已经完成商品搜索。"
        )

        await current_monitor.report_error(
            error_type="RuntimeError",
            message="商品接口调用失败。",
        )

    task_payload = recorded_payloads[0]
    error_payload = recorded_payloads[1]

    assert task_payload["event"] == (
        "task_result"
    )
    assert task_payload["data"] == {
        "final_answer": "已经完成商品搜索。"
    }

    assert error_payload["event"] == "error"
    assert error_payload["message"] == (
        "商品接口调用失败。"
    )
    assert error_payload["data"] == {
        "error_type": "RuntimeError",
    }


@pytest.mark.asyncio
async def test_negative_duration_is_rejected() -> None:
    """
    验证负数执行耗时会被拒绝。
    """

    current_monitor = Monitor()

    with pytest.raises(
        ValueError,
        match="duration_ms 不能小于 0",
    ):
        await current_monitor.report_tool_end(
            tool_name="item_search",
            duration_ms=-1,
        )