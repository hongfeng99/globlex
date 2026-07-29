from __future__ import annotations

from datetime import datetime
from typing import Any

from app.api.connection import manager
from app.api.context import get_thread_id
from app.observability.alerts import (
    tool_rt_monitor,
)


class Monitor:
    """
    统一封装 Globex 的实时进度事件。

    Agent 和工具只需要调用 report_xxx() 方法，
    不需要关心：

    1. 当前 thread_id 如何获取；
    2. WebSocket 保存在哪里；
    3. 消息发送给哪个前端；
    4. 事件 JSON 如何组装；
    5. 时间戳如何生成。
    """

    async def _emit(
        self,
        event: str,
        message: str,
        data: dict[str, Any],
    ) -> bool:
        """
        创建统一格式的监控事件，并推送给当前任务。

        参数：
            event:
                事件名称，例如 tool_start。

            message:
                面向前端用户的事件说明。

            data:
                当前事件携带的结构化数据。

        返回：
            发送成功返回 True。

            当前没有 thread_id、前端未连接，
            或 WebSocket 发送失败时返回 False。
        """

        # Monitor 不需要调用者显式传入 thread_id，
        # 而是从当前 ContextVar 中自动取得。
        thread_id = get_thread_id()

        if thread_id is None:
            # 某些单元测试或离线脚本不在请求上下文中。
            # 这时不发送事件，也不抛出异常。
            return False

        payload: dict[str, Any] = {
            "type": "monitor_event",
            "event": event,
            "message": message,
            "data": data,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
        }

        return await manager.send_to_thread(
            payload=payload,
            thread_id=thread_id,
        )

    async def report_tool_start(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        """
        上报工具开始执行事件。
        """

        return await self._emit(
            event="tool_start",
            message=f"正在调用 {tool_name}",
            data={
                "tool_name": tool_name,
                "args": args,
            },
        )

    async def report_tool_end(
        self,
        tool_name: str,
        duration_ms: int,
    ) -> bool:
        """
        上报工具执行结束事件。
        """

        if duration_ms < 0:
            raise ValueError(
                "duration_ms 不能小于 0。"
            )

        tool_rt_monitor.record(
            tool_name, duration_ms
        )
        return await self._emit(
            event="tool_end",
            message=f"{tool_name} 完成",
            data={
                "tool_name": tool_name,
                "duration_ms": duration_ms,
            },
        )

    async def report_fork(
        self,
        sub_thread_id: str,
        demands: str,
    ) -> bool:
        """
        上报主 Agent 派发子 AgentLoop 的事件。
        """

        normalized_sub_thread_id = (
            sub_thread_id.strip()
        )

        if not normalized_sub_thread_id:
            raise ValueError(
                "sub_thread_id 不能为空字符串。"
            )

        return await self._emit(
            event="fork",
            message="派发子 AgentLoop",
            data={
                "sub_thread_id": (
                    normalized_sub_thread_id
                ),
                # 防止用户任务特别长，
                # 监控事件只展示前 200 个字符。
                "demands": demands[:200],
            },
        )

    async def report_task_result(
        self,
        final_answer: str,
    ) -> bool:
        """
        上报主任务最终结果。
        """

        return await self._emit(
            event="task_result",
            message="任务完成",
            data={
                "final_answer": final_answer,
            },
        )

    async def report_error(
        self,
        error_type: str,
        message: str,
    ) -> bool:
        """
        上报任务执行错误。
        """

        return await self._emit(
            event="error",
            message=message,
            data={
                "error_type": error_type,
            },
        )


# 整个项目共享一个 Monitor。
monitor = Monitor()
