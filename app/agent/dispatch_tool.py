from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from app.agent.fork_guard import (
    ForkLimitExceeded,
    enter_fork,
)
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
    if not isinstance(messages, list) or not messages:
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
            last_message, "content", last_message
        )
    return (
        content
        if isinstance(content, str)
        else str(content)
    )


@tool
async def dispatch_tool(demands: str) -> str:
    """
    派一个同质子 AgentLoop 执行独立需求。

    适合可并行、需要隔离大上下文，或内部仍需至少三轮
    Think/Act 的子任务。单平台、单关键词检索不要 fork。
    """

    normalized_demands = demands.strip()
    if not normalized_demands:
        raise ValueError(
            "demands 不能为空字符串。"
        )

    sub_thread_id = f"sub-{uuid4().hex[:8]}"
    parent_session_dir = require_session_dir()
    await monitor.report_fork(
        sub_thread_id, normalized_demands
    )

    # 延迟导入，避免注册 dispatch_tool 时形成循环依赖。
    from app.agent.tool_registry import (
        FULL_TOOL_SET,
    )

    sub_agent = create_agent(
        model=get_llm(),
        tools=FULL_TOOL_SET,
        system_prompt=get_system_prompt(),
    )

    try:
        with enter_fork(max_depth=2):
            with bind_thread_context(
                thread_id=sub_thread_id,
                session_dir=parent_session_dir,
            ):
                result = await asyncio.wait_for(
                    sub_agent.ainvoke(
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
                    ),
                    timeout=90,
                )
    except ForkLimitExceeded as exc:
        return f"子任务未执行：{exc}"
    except TimeoutError:
        return "子任务执行超过 90 秒，已停止。"

    return _last_message_content(result)


__all__ = ["dispatch_tool"]
