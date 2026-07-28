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

    def fake_create_react_agent(
        **kwargs: Any,
    ) -> FakeAgent:
        assert dispatch_tool in (
            kwargs["tools"]
        )
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
        "create_react_agent",
        fake_create_react_agent,
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
