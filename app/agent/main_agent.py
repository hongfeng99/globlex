from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

from app.agent.llm import get_llm
from app.agent.prompts import get_system_prompt
from app.agent.tool_registry import FULL_TOOL_SET
from app.agent.middleware import AGENT_MIDDLEWARE
from app.agent.offline_fallback import build_clarification
from app.agent.request_context import bind_request_context
from app.api.monitor import monitor
from app.budget.limits import get_user_token_limit
from app.budget.token_budget import init_budget
from app.config import env_float, env_int
from app.memory.store import preference_store
from app.memory.context import bind_user_context
from app.memory.injector import inject_strategies
from app.harness.setup import setup_harness
from app.harness.phase_machine import Phase, phase_machine


def _final_content(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError(
            "主 AgentLoop 未返回消息。"
        )
    last = messages[-1]
    content = (
        last.content
        if isinstance(last, BaseMessage)
        else getattr(last, "content", last)
    )
    return (
        content
        if isinstance(content, str)
        else str(content)
    )


async def build_main_agent(
    *,
    user_id: str | None = None,
    query: str = "",
) -> Any:
    preferences = (
        await preference_store.render(
            user_id, query
        )
        if user_id
        else ""
    )
    system_prompt = get_system_prompt(
        preferences
    )
    if user_id:
        strategies = (
            await preference_store
            .read_relevant_strategies(query)
        )
        system_prompt = inject_strategies(
            system_prompt, strategies
        )
    return create_agent(
        model=get_llm(),
        tools=FULL_TOOL_SET,
        system_prompt=system_prompt,
        middleware=AGENT_MIDDLEWARE,
    )


async def run_main_agent(
    user_message: str,
    *,
    thread_id: str,
    user_id: str | None = None,
    timeout_seconds: float | None = None,
) -> str:
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else env_float(
            "AGENT_TIMEOUT_SEC",
            300,
            minimum=1,
        )
    )
    recursion_limit = env_int(
        "AGENT_LOOP_MAX_ITERATIONS",
        30,
        minimum=1,
    )
    harness = None
    harness_context: dict[str, Any] = {}
    try:
        harness = setup_harness()
        harness_context = await harness.run(
            "on_session_start",
            {
                "query": user_message,
                "original_query": user_message,
                "thread_id": thread_id,
                "user_id": user_id,
            },
        )
        clarification = build_clarification(user_message)
        if clarification is not None:
            await harness.run(
                "on_session_end",
                {
                    **harness_context,
                    "final_answer": clarification,
                    "trajectory": [],
                },
            )
            await monitor.report_task_result(clarification)
            return clarification
        agent = await build_main_agent(
            user_id=user_id,
            query=user_message,
        )
        init_budget(
            get_user_token_limit(user_id)
        )
        with bind_request_context(user_message):
            with bind_user_context(user_id):
                with phase_machine.bind(Phase.PLANNING):
                    result = await asyncio.wait_for(
                        agent.ainvoke(
                            {
                                "messages": [
                                    ("user", user_message)
                                ]
                            },
                            config={
                                "configurable": {
                                    "thread_id": thread_id
                                },
                                "recursion_limit": recursion_limit,
                            },
                        ),
                        timeout=effective_timeout,
                    )
        final_answer = _final_content(result)
        await harness.run(
            "on_session_end",
            {
                **harness_context,
                "final_answer": final_answer,
                "trajectory": result.get(
                    "messages", []
                ),
            },
        )
        await monitor.report_task_result(
            final_answer
        )
        return final_answer
    except asyncio.CancelledError:
        await monitor.report_error(
            "cancelled", "任务已取消"
        )
        raise
    except Exception as exc:
        await monitor.report_error(
            type(exc).__name__, str(exc)
        )
        raise


__all__ = [
    "build_main_agent",
    "run_main_agent",
]
