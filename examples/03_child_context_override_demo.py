import asyncio
from pathlib import Path
from uuid import uuid4

from app.api.context import (
    require_session_dir,
    require_thread_id,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


async def child_agent(
    child_name: str,
) -> str:
    """
    模拟真正的子 AgentLoop。

    在真实项目中，这里将替换为：

        await sub_agent.ainvoke(...)
    """

    await asyncio.sleep(0.1)

    message = (
        f"[{child_name}] "
        f"thread_id={require_thread_id()}, "
        f"session_dir={require_session_dir()}"
    )

    print(message)

    return message


async def dispatch_child_agent(
    child_name: str,
) -> str:
    """
    模拟 dispatch_tool 的核心上下文逻辑。

    子 Agent：

    1. 使用独立的 sub thread_id；
    2. 继承主 Agent 的 session_dir；
    3. 执行结束后自动恢复调用前的上下文。
    """

    # 当前还处于主 Agent 的上下文中。
    parent_thread_id = require_thread_id()
    parent_session_dir = require_session_dir()

    # 为当前子 Agent 创建独立 thread_id。
    sub_thread_id = (
        f"{parent_thread_id}"
        f"-sub-{uuid4().hex[:8]}"
    )

    print(
        f"[创建子任务] "
        f"parent_thread_id={parent_thread_id}, "
        f"sub_thread_id={sub_thread_id}"
    )

    # 在当前子任务范围内覆盖上下文：
    #
    # thread_id 改为子 Agent 自己的 ID；
    # session_dir 继续使用主 Agent 的目录。
    with bind_thread_context(
        thread_id=sub_thread_id,
        session_dir=parent_session_dir,
    ):
        return await child_agent(
            child_name
        )


async def handle_request(
    thread_id: str,
) -> None:
    """
    模拟一个用户请求以及对应的主 AgentLoop。
    """

    session_dir = (
        Path("output")
        / thread_id
    )

    with bind_thread_context(
        thread_id=thread_id,
        session_dir=session_dir,
    ):
        print()
        print("=" * 70)

        print(
            f"[主 Agent 开始] "
            f"thread_id={require_thread_id()}, "
            f"session_dir={require_session_dir()}"
        )

        # asyncio.gather 会并发执行两个子任务。
        #
        # 每个 dispatch_child_agent 都会在自己的 Task 中，
        # 将 thread_id 改成独立的 sub-xxx。
        await asyncio.gather(
            dispatch_child_agent(
                "子Agent-1"
            ),
            dispatch_child_agent(
                "子Agent-2"
            ),
        )

        # 两个子任务结束后，
        # 主任务的上下文没有受到影响。
        print(
            f"[主 Agent 恢复] "
            f"thread_id={require_thread_id()}, "
            f"session_dir={require_session_dir()}"
        )

        print("=" * 70)


async def main() -> None:
    """
    同时处理两个用户请求。
    """

    await asyncio.gather(
        handle_request("thread-A"),
        handle_request("thread-B"),
    )


if __name__ == "__main__":
    asyncio.run(main())