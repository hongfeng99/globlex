from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from app.agent.llm import get_llm
from app.agent.prompts import get_system_prompt
from app.api.monitor import monitor
from app.utils.thread_ctx import (
    bind_thread_context,
    require_session_dir,
)


def _last_message_content(
    result: dict[str, Any],
) -> str:
    messages = result.get("messages")

    if (
        not isinstance(messages, list)
        or not messages
    ):
        raise RuntimeError(
            "子 AgentLoop 未返回消息。"
        )

    last_message = messages[-1]

    if isinstance(last_message, BaseMessage):
        content = last_message.content
    elif isinstance(last_message, tuple):
        content = last_message[-1]
    else:
        content = getattr(
            last_message,
            "content",
            last_message,
        )

    if isinstance(content, str):
        return content

    return str(content)


@tool
async def dispatch_tool(
    demands: str,
) -> str:
    """
    派一个同质子 AgentLoop 执行 demands，并返回最终回复。

    适用条件（任一即可）：
    1. 能并行：多个子任务可以同时跑；
    2. 上下文要隔离：子任务输出很大，不应污染主 loop；
    3. 调用链不少于 3：子任务内部还要多轮 Think → Act。
    """

    normalized_demands = demands.strip()

    if not normalized_demands:
        raise ValueError(
            "demands 不能为空字符串。"
        )

    sub_thread_id = (
        f"sub-{uuid4().hex[:8]}"
    )
    parent_session_dir = (
        require_session_dir()
    )

    await monitor.report_fork(
        sub_thread_id,
        normalized_demands,
    )

    # 延迟导入可避免 registry 在注册 dispatch_tool 时
    # 产生模块循环依赖。
    from app.agent.registry import (
        FULL_TOOL_SET,
    )

    sub_agent = create_agent(
        model=get_llm(),
        tools=FULL_TOOL_SET,
        system_prompt=get_system_prompt(),
    )

    with bind_thread_context(
        thread_id=sub_thread_id,
        session_dir=parent_session_dir,
    ):
        result = await sub_agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        normalized_demands,
                    )
                ]
            },
            config={
                "configurable": {
                    "thread_id": (
                        sub_thread_id
                    ),
                }
            },
        )

    return _last_message_content(result)


__all__ = [
    "dispatch_tool",
]
