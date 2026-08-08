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
from app.agent.middleware import AGENT_MIDDLEWARE
from app.agent.request_context import (
    SUPPORTED_PLATFORMS,
    get_original_request,
    platforms_in_text,
    reserve_dispatch,
)
from app.agent.prompts import get_system_prompt
from app.api.monitor import monitor
from app.config import env_float, env_int
from app.harness.phase_machine import Phase, phase_machine
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


async def _run_sub_agent(
    normalized_demands: str,
    parent_session_dir: Any,
) -> str:
    sub_thread_id = f"sub-{uuid4().hex[:8]}"
    await monitor.report_fork(
        sub_thread_id, normalized_demands
    )

    # 延迟导入，避免注册 dispatch_tool 时形成循环依赖。
    from app.agent.tool_registry import (
        CORE_TOOL_SET,
    )

    sub_agent_prompt = (
        get_system_prompt()
        + "\n\n<sub_agent_scope>"
        "你是一个独立的平台检索子 AgentLoop。"
        "必须调用工具获取商品，不得凭空生成候选。"
        "先执行 item_search 并检查候选是否命中平台、品类和硬约束；"
        "如果候选为空、品类偏移或约束命中不足，可以改写查询、"
        "调整 top_k 后再次调用 item_search，最多进行 3 轮召回。"
        "不得搜索 demands 之外的平台，也不得派生新的子 Agent。"
        "完成后只返回精简的平台候选摘要和无匹配原因，"
        "由主 Agent 负责跨平台合流与最终总结。"
        "</sub_agent_scope>"
    )

    sub_agent = create_agent(
        model=get_llm(),
        tools=CORE_TOOL_SET,
        system_prompt=sub_agent_prompt,
        middleware=AGENT_MIDDLEWARE,
    )

    try:
        with enter_fork(
            max_depth=env_int(
                "FORK_MAX_DEPTH",
                2,
                minimum=1,
            )
        ):
            with phase_machine.bind(
                Phase.SEARCHING,
                fixed=True,
            ):
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
                                },
                                "recursion_limit": env_int(
                                    "SUB_AGENT_LOOP_MAX_ITERATIONS",
                                    30,
                                    minimum=1,
                                ),
                            },
                        ),
                        timeout=env_float(
                            "SUB_AGENT_TIMEOUT_SEC",
                            180,
                            minimum=1,
                        ),
                    )
    except ForkLimitExceeded as exc:
        return f"子任务未执行：{exc}"
    except TimeoutError:
        return "子任务执行超时，已停止。"
    except Exception as exc:
        # 单个平台模型调用失败时，保留其他平台子 Agent 的成功结果。
        return (
            "子任务执行失败："
            f"{type(exc).__name__}: {exc}"
        )

    return _last_message_content(result)


@tool
async def dispatch_tool(demands: str) -> str:
    """
    派一个或按平台拆分的多个同质子 AgentLoop 执行独立需求。

    同一 demands 同时包含多个受支持平台时，会强制拆为
    每个平台一个并行子任务，避免单个子 Agent 串行搜索四次。
    """

    normalized_demands = demands.strip()
    if not normalized_demands:
        raise ValueError(
            "demands 不能为空字符串。"
        )

    original_request = get_original_request()
    original_platforms = platforms_in_text(original_request)
    demand_platforms = platforms_in_text(normalized_demands)
    if original_platforms:
        mentioned = original_platforms
    elif original_request:
        # 用户未指定平台时，Globex 默认比较全部四个平台；
        # 不允许模型自行缩减成单平台需求。
        mentioned = set(SUPPORTED_PLATFORMS)
    else:
        mentioned = demand_platforms or set(SUPPORTED_PLATFORMS)

    reservation_demands = normalized_demands
    missing_platform_labels = [
        platform
        for platform in SUPPORTED_PLATFORMS
        if platform in mentioned
        and platform not in platforms_in_text(reservation_demands)
    ]
    if missing_platform_labels:
        reservation_demands += (
            "\n目标平台："
            + "、".join(missing_platform_labels)
        )

    allowed, reason = reserve_dispatch(reservation_demands)
    if not allowed:
        return f"子任务未执行：{reason}"

    parent_session_dir = require_session_dir()
    platforms = [
        platform
        for platform in SUPPORTED_PLATFORMS
        if platform in mentioned
    ]
    if len(platforms) <= 1:
        return await _run_sub_agent(
            normalized_demands,
            parent_session_dir,
        )

    scoped_demands = [
        (
            f"{normalized_demands}\n\n"
            f"你只负责 {platform} 平台。"
            "不得搜索其他平台，也不得再次调用 dispatch_tool。"
        )
        for platform in platforms
    ]
    results = await asyncio.gather(
        *(
            _run_sub_agent(demand, parent_session_dir)
            for demand in scoped_demands
        )
    )
    return "\n\n".join(
        f"[{platform}] {result}"
        for platform, result in zip(platforms, results)
    )


__all__ = ["dispatch_tool"]
