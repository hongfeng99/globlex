import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import app.agent.dispatch_tool as dispatch_module
from app.agent.request_context import bind_request_context
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
        assert dispatch_tool not in kwargs["tools"]
        assert any(
            tool.name == "item_search"
            for tool in kwargs["tools"]
        )

        assert "system_prompt" in kwargs
        assert "独立的平台检索子 AgentLoop" in kwargs["system_prompt"]
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
        },
        "recursion_limit": 30,
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


@pytest.mark.asyncio
async def test_one_multi_platform_dispatch_is_split_into_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = object()
    seen_demands: list[str] = []
    created_models: list[object] = []
    created_tool_names: list[set[str]] = []
    seen_configs: list[dict[str, Any]] = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    class LoopCapableAgent:
        async def ainvoke(
            self,
            payload: dict[str, Any],
            *,
            config: dict[str, Any],
        ) -> dict[str, Any]:
            demand = payload["messages"][0][1]
            seen_demands.append(demand)
            seen_configs.append(config)
            if len(seen_demands) == 4:
                all_started.set()
            await release.wait()
            return {
                "messages": [
                    AIMessage(content="第一轮召回后继续检查"),
                    AIMessage(content="完成"),
                ]
            }

    async def fake_report_fork(
        sub_thread_id: str,
        demands: str,
    ) -> bool:
        return True

    def fake_create_agent(**kwargs: Any) -> LoopCapableAgent:
        created_models.append(kwargs["model"])
        created_tool_names.append(
            {tool.name for tool in kwargs["tools"]}
        )
        assert "独立的平台检索子 AgentLoop" in kwargs["system_prompt"]
        return LoopCapableAgent()

    monkeypatch.setattr(
        dispatch_module.monitor,
        "report_fork",
        fake_report_fork,
    )
    monkeypatch.setattr(
        dispatch_module,
        "get_llm",
        lambda: fake_llm,
    )
    monkeypatch.setattr(
        dispatch_module,
        "create_agent",
        fake_create_agent,
    )

    with bind_request_context("四个平台搜索"):
        with bind_thread_context("main-thread", tmp_path):
            task = asyncio.create_task(
                dispatch_tool.ainvoke(
                    {
                        "demands": (
                            "分别搜索 Amazon、Shopee、"
                            "AliExpress 和 eBay"
                        )
                    }
                )
            )
            await asyncio.wait_for(all_started.wait(), timeout=1)
            release.set()
            result = await task

    assert len(seen_demands) == 4
    assert all(
        any(platform in demand.casefold() for demand in seen_demands)
        for platform in ("amazon", "shopee", "aliexpress", "ebay")
    )
    assert created_models == [fake_llm] * 4
    assert all(
        "item_search" in names and "dispatch_tool" not in names
        for names in created_tool_names
    )
    assert all(
        config["recursion_limit"] == 30
        for config in seen_configs
    )
    assert all(
        f"[{platform}] 完成" in result
        for platform in (
            "amazon",
            "shopee",
            "aliexpress",
            "ebay",
        )
    )


@pytest.mark.asyncio
async def test_unspecified_platform_dispatch_defaults_to_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_demands: list[str] = []

    async def fake_run_sub_agent(
        demands: str,
        parent_session_dir: Path,
    ) -> str:
        seen_demands.append(demands)
        return "完成"

    monkeypatch.setattr(
        dispatch_module,
        "_run_sub_agent",
        fake_run_sub_agent,
    )

    with bind_request_context("推荐一把机械键盘"):
        with bind_thread_context("main-thread", tmp_path):
            result = await dispatch_tool.ainvoke(
                {"demands": "搜索满足用户约束的机械键盘"}
            )

    assert len(seen_demands) == 4
    assert all(
        any(platform in demand.casefold() for demand in seen_demands)
        for platform in ("amazon", "shopee", "aliexpress", "ebay")
    )
    assert all(
        f"[{platform}] 完成" in result
        for platform in ("amazon", "shopee", "aliexpress", "ebay")
    )
