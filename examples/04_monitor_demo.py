import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.api.connection import manager
from app.api.monitor import monitor
from app.utils.thread_ctx import (
    bind_thread_context,
)


class ConsoleWebSocket:
    """
    模拟浏览器 WebSocket。

    后端发送的 JSON 会直接打印到终端。
    """

    async def accept(self) -> None:
        print("[WebSocket] 连接已接受")

    async def send_json(
        self,
        payload: dict[str, Any],
    ) -> None:
        print()
        print("[WebSocket] 收到事件：")

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )


async def main() -> None:
    """
    模拟工具开始、结束、fork 和任务完成。
    """

    thread_id = "thread-demo"

    websocket = ConsoleWebSocket()

    await manager.connect(
        websocket=websocket,  # type: ignore[arg-type]
        thread_id=thread_id,
    )

    try:
        with bind_thread_context(
            thread_id=thread_id,
            session_dir=(
                Path("output")
                / thread_id
            ),
        ):
            await monitor.report_tool_start(
                tool_name="item_search",
                args={
                    "query": "无线耳机",
                    "platform": "amazon",
                },
            )

            start_time = time.perf_counter()

            # 模拟工具执行。
            await asyncio.sleep(0.1)

            duration_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            await monitor.report_tool_end(
                tool_name="item_search",
                duration_ms=duration_ms,
            )

            await monitor.report_fork(
                sub_thread_id="sub-12345678",
                demands=(
                    "搜索另外三个平台的无线耳机"
                ),
            )

            await monitor.report_task_result(
                final_answer=(
                    "已经完成无线耳机搜索。"
                ),
            )

    finally:
        await manager.disconnect(
            websocket=websocket,  # type: ignore[arg-type]
            thread_id=thread_id,
        )


if __name__ == "__main__":
    asyncio.run(main())