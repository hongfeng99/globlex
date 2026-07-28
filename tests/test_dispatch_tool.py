import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import app.agent.dispatch_tool as dispatch_module
from app.agent.dispatch_tool import dispatch_tool
from app.api.context import (
    require_session_dir,
    require_thread_id,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


class FakeAgent:
    def __init__(self) -> None:
        self.seen_thread_id: str | None = None
        self.seen_session_dir: Path | None = None
        self.config: dict[str, Any] | None = None

    async def ainvoke(
        self,
        payload: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.seen_thread_id = (
            require_thread_id()
        )
        self.seen_session_dir = (
            require_session_dir()
        )
        self.config = config

        return {
            "messages": [
                AIMessage(
                    content="子任务完成"
                )
            ]
        }


@pytest.mark.asyncio
async def test_dispatch_uses_child_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agent = FakeAgent()
    fork_events: list[
        tuple[str, str]
    ] = []

    async def fake_report_fork(
        sub_thread_id: str,
        demands: str,
    ) -> bool:
        fork_events.append(
            (sub_thread_id, demands)
        )
        return True

    def fake_create_agent(
        **kwargs: Any,
    ) -> FakeAgent:
        assert dispatch_tool in kwargs["tools"]

        assert "system_prompt" in kwargs
        assert "prompt" not in kwargs

        return fake_agent

    monkeypatch.setattr(
        dispatch_module.monitor,
        "report_fork",
        fake_report_fork,
    )
    monkeypatch.setattr(
        dispatch_module,
        "get_llm",
        lambda: object(),
    )
    monkeypatch.setattr(
        dispatch_module,
        "create_agent",
        fake_create_agent,
    )

    with bind_thread_context(
        "main-thread",
        tmp_path,
    ):
        result = await dispatch_tool.ainvoke(
            {
                "demands": (
                    "在 amazon 搜索旅行收纳袋"
                )
            }
        )

        assert require_thread_id() == (
            "main-thread"
        )

    assert result == "子任务完成"
    assert len(fork_events) == 1

    sub_thread_id = fork_events[0][0]
    assert sub_thread_id.startswith("sub-")
    assert fake_agent.seen_thread_id == (
        sub_thread_id
    )
    assert fake_agent.seen_session_dir == (
        tmp_path.resolve()
    )
    assert fake_agent.config == {
        "configurable": {
            "thread_id": sub_thread_id,
        }
    }

@pytest.mark.asyncio
async def test_four_dispatch_calls_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_count = 0

    all_started = asyncio.Event()
    release_agents = asyncio.Event()

    seen_thread_ids: set[str] = set()

    class BlockingAgent:
        async def ainvoke(
            self,
            payload: dict[str, Any],
            *,
            config: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal started_count

            seen_thread_ids.add(
                require_thread_id()
            )

            started_count += 1

            if started_count == 4:
                all_started.set()

            # 四个子 Agent 都进入这里后，
            # 测试才统一允许它们结束。
            await release_agents.wait()

            return {
                "messages": [
                    AIMessage(
                        content="子任务完成"
                    )
                ]
            }

    async def fake_report_fork(
        sub_thread_id: str,
        demands: str,
    ) -> bool:
        return True

    def fake_create_agent(
        **kwargs: Any,
    ) -> BlockingAgent:
        return BlockingAgent()

    monkeypatch.setattr(
        dispatch_module.monitor,
        "report_fork",
        fake_report_fork,
    )

    monkeypatch.setattr(
        dispatch_module,
        "get_llm",
        lambda: object(),
    )

    monkeypatch.setattr(
        dispatch_module,
        "create_agent",
        fake_create_agent,
    )

    platforms = [
        "amazon",
        "shopee",
        "aliexpress",
        "ebay",
    ]

    with bind_thread_context(
        "main-thread",
        tmp_path,
    ):
        tasks = [
            asyncio.create_task(
                dispatch_tool.ainvoke(
                    {
                        "demands": (
                            f"在 {platform} 搜索"
                            "旅行收纳袋"
                        )
                    }
                )
            )
            for platform in platforms
        ]

        try:
            # 只有四个 Agent 都已经开始，
            # all_started 才会被设置。
            await asyncio.wait_for(
                all_started.wait(),
                timeout=2.0,
            )
        finally:
            release_agents.set()

        results = await asyncio.gather(
            *tasks
        )

    assert results == [
        "子任务完成",
        "子任务完成",
        "子任务完成",
        "子任务完成",
    ]

    # 四次 fork 必须生成四个不同的子线程。
    assert len(seen_thread_ids) == 4

    assert all(
        thread_id.startswith("sub-")
        for thread_id in seen_thread_ids
    )