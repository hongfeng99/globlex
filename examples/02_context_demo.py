import asyncio
from pathlib import Path

from app.api.context import (
    require_session_dir,
    require_thread_id,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


async def child_agent(
    child_name: str,
) -> None:
    """
    模拟由主 Agent fork 出来的子 Agent。
    """

    await asyncio.sleep(0.1)

    print(
        f"[{child_name}] "
        f"thread_id={require_thread_id()}, "
        f"session_dir={require_session_dir()}"
    )


async def handle_request(
    thread_id: str,
) -> None:
    """
    模拟处理一个用户请求。
    """

    session_dir = (
        Path("output")
        / thread_id
    )

    with bind_thread_context(
        thread_id=thread_id,
        session_dir=session_dir,
    ):
        print(
            f"[主任务开始] "
            f"thread_id={require_thread_id()}"
        )

        # 在当前上下文中创建子任务。
        #
        # asyncio.create_task 会复制当前 Context，
        # 因此两个子 Agent 都能取得父任务的
        # thread_id 和 session_dir。
        await asyncio.gather(
            asyncio.create_task(
                child_agent("子Agent-1")
            ),
            asyncio.create_task(
                child_agent("子Agent-2")
            ),
        )

        print(
            f"[主任务结束] "
            f"thread_id={require_thread_id()}"
        )


async def main() -> None:
    """
    同时模拟两个用户请求。
    """

    await asyncio.gather(
        handle_request("thread-A"),
        handle_request("thread-B"),
    )


if __name__ == "__main__":
    asyncio.run(main())