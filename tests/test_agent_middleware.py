from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import (
    ModelRequest,
    ToolCallRequest,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent.middleware import (
    enforce_phase_tool_call,
    enforce_phase_tools,
    enforce_terminal_summary,
    is_transient_model_error,
    phase_tool_choice,
    resilient_model_call,
    should_force_shopping_summary,
    synchronize_phase,
)
from app.agent.tool_registry import FULL_TOOL_SET
from app.agent.request_context import (
    bind_request_context,
    record_category_insight,
)
from app.harness.phase_machine import (
    PHASE_TOOLS,
    Phase,
    phase_machine,
)
from app.tools.item_picker import item_picker
from app.tools.shopping_summary import shopping_summary


def test_summary_is_a_direct_terminal_tool() -> None:
    assert shopping_summary.return_direct is True


def test_force_summary_only_after_successful_picker() -> None:
    assert should_force_shopping_summary(
        [
            ToolMessage(
                content="picks",
                name="item_picker",
                tool_call_id="picker-1",
            )
        ]
    )
    assert not should_force_shopping_summary(
        [HumanMessage(content="推荐键盘")]
    )
    assert not should_force_shopping_summary(
        [
            ToolMessage(
                content="error",
                name="item_picker",
                tool_call_id="picker-2",
                status="error",
            )
        ]
    )


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.asyncio
async def test_phase_exposes_exact_tool_subset(phase: Phase) -> None:
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="购物请求")],
        system_message=SystemMessage(content="system"),
        tools=FULL_TOOL_SET,
    )
    captured: dict[str, Any] = {}

    async def handler(
        filtered_request: ModelRequest,
    ) -> AIMessage:
        captured["tool_names"] = {
            tool.name for tool in filtered_request.tools
        }
        captured["system_prompt"] = (
            filtered_request.system_message.text
            if filtered_request.system_message
            else ""
        )
        return AIMessage(content="continue")

    with phase_machine.bind(phase, fixed=True):
        await enforce_phase_tools.awrap_model_call(
            request,
            handler,
        )

    assert captured["tool_names"] == PHASE_TOOLS[phase]
    assert f"当前阶段：{phase.value}" in captured["system_prompt"]


def _tool_result(name: str, call_id: str) -> ToolMessage:
    return ToolMessage(
        content="result",
        name=name,
        tool_call_id=call_id,
    )


def test_phase_transitions_follow_successful_results() -> None:
    planner = _tool_result("planner", "planner-1")
    dispatch = _tool_result("dispatch_tool", "dispatch-1")
    picker = _tool_result("item_picker", "picker-1")

    with phase_machine.bind(Phase.PLANNING):
        assert synchronize_phase([planner]) == Phase.SEARCHING
        assert synchronize_phase([planner, dispatch]) == Phase.COMPARING
        assert synchronize_phase(
            [planner, dispatch, picker]
        ) == Phase.CONCLUDING


def test_sub_agent_fixed_search_phase_can_recall_again() -> None:
    item_result = _tool_result("item_search", "search-1")
    with phase_machine.bind(Phase.SEARCHING, fixed=True):
        assert synchronize_phase([item_result]) == Phase.SEARCHING
        assert phase_machine.is_allowed("item_search")
        assert not phase_machine.is_allowed("price_compare")


def test_root_phase_tool_choice_forces_progress() -> None:
    planner = _tool_result("planner", "planner-1")
    dispatch = _tool_result("dispatch_tool", "dispatch-1")
    price = _tool_result("price_compare", "price-1")
    shipping = _tool_result("shipping_calc", "shipping-1")

    with phase_machine.bind(Phase.PLANNING):
        assert phase_tool_choice(Phase.PLANNING, []) == "required"
        with bind_request_context(
            "比较 Amazon、Shopee、AliExpress 和 eBay 的键盘"
        ):
            assert phase_tool_choice(
                Phase.SEARCHING,
                [planner],
            ) == "category_insight"
            record_category_insight(
                effective=True,
                confidence=0.8,
            )
            assert phase_tool_choice(
                Phase.SEARCHING,
                [
                    planner,
                    _tool_result(
                        "category_insight",
                        "insight-1",
                    ),
                ],
            ) == "dispatch_tool"
        assert phase_tool_choice(
            Phase.COMPARING,
            [planner, dispatch],
        ) == "price_compare"
        assert phase_tool_choice(
            Phase.COMPARING,
            [planner, dispatch, price],
        ) == "shipping_calc"
        assert phase_tool_choice(
            Phase.COMPARING,
            [planner, dispatch, price, shipping],
        ) == "item_picker"
        assert phase_tool_choice(
            Phase.CONCLUDING,
            [planner, dispatch, price, shipping],
        ) == "shopping_summary"


def test_child_must_search_once_then_may_reflect() -> None:
    with phase_machine.bind(Phase.SEARCHING, fixed=True):
        assert phase_tool_choice(Phase.SEARCHING, []) == "item_search"
        assert phase_tool_choice(
            Phase.SEARCHING,
            [_tool_result("item_search", "search-1")],
        ) is None


def test_single_platform_search_uses_item_search() -> None:
    with bind_request_context("在 Amazon 搜索机械键盘"):
        with phase_machine.bind(Phase.SEARCHING):
            record_category_insight(
                effective=True,
                confidence=0.8,
            )
            assert phase_tool_choice(
                Phase.SEARCHING,
                [_tool_result("category_insight", "insight-1")],
            ) == "item_search"


def test_unspecified_platform_defaults_to_dispatch() -> None:
    with bind_request_context("推荐一把机械键盘"):
        with phase_machine.bind(Phase.SEARCHING):
            record_category_insight(
                effective=True,
                confidence=0.8,
            )
            assert phase_tool_choice(
                Phase.SEARCHING,
                [_tool_result("category_insight", "insight-1")],
            ) == "dispatch_tool"


def test_invalid_category_insight_forces_web_fallback() -> None:
    insight = _tool_result("category_insight", "insight-1")
    web = _tool_result("web_search", "web-1")

    with bind_request_context("推荐骑行三件套"):
        with phase_machine.bind(Phase.SEARCHING):
            assert phase_tool_choice(
                Phase.SEARCHING,
                [],
            ) == "category_insight"
            record_category_insight(
                effective=False,
                confidence=0.1,
                degraded_reason="没有有效知识卡",
            )
            assert phase_tool_choice(
                Phase.SEARCHING,
                [insight],
            ) == "web_search"
            assert phase_tool_choice(
                Phase.SEARCHING,
                [insight, web],
            ) == "dispatch_tool"


def test_chat_fallback_is_terminal() -> None:
    from app.tools.chat_fallback import chat_fallback

    assert chat_fallback.return_direct is True


@pytest.mark.asyncio
async def test_phase_guard_rejects_out_of_order_picker() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "item_picker",
            "args": {},
            "id": "picker-early",
            "type": "tool_call",
        },
        tool=item_picker,
        state={
            "messages": [
                _tool_result("dispatch_tool", "dispatch-1")
            ]
        },
        runtime=None,  # type: ignore[arg-type]
    )
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        raise AssertionError("handler should not run")

    with phase_machine.bind(Phase.COMPARING, fixed=True):
        response = await enforce_phase_tool_call.awrap_tool_call(
            request,
            handler,
        )

    assert called is False
    assert isinstance(response, ToolMessage)
    assert response.status == "error"
    assert "shipping_calc" in str(response.content)


@pytest.mark.asyncio
async def test_middleware_forces_only_summary_tool() -> None:
    request = ModelRequest(
        model=object(),  # handler is stubbed; no model call is made.
        messages=[
            HumanMessage(content="推荐机械键盘"),
            ToolMessage(
                content="picked items",
                name="item_picker",
                tool_call_id="picker-1",
            ),
        ],
        system_message=SystemMessage(
            content="system"
        ),
        tools=FULL_TOOL_SET,
    )
    captured: dict[str, Any] = {}

    async def handler(
        forced_request: ModelRequest,
    ) -> AIMessage:
        captured["tools"] = forced_request.tools
        captured["tool_choice"] = (
            forced_request.tool_choice
        )
        captured["system_prompt"] = (
            forced_request.system_message.text
            if forced_request.system_message
            else ""
        )
        return AIMessage(content="")

    await enforce_terminal_summary.awrap_model_call(
        request,
        handler,
    )

    assert captured["tools"] == [shopping_summary]
    assert captured["tool_choice"] == "shopping_summary"
    assert "必须且只能调用" in captured["system_prompt"]


def test_transient_model_error_detection() -> None:
    assert is_transient_model_error(
        RuntimeError("Error code: 429 rate limit")
    )
    assert is_transient_model_error(
        TypeError("Received null value for choices")
    )
    assert not is_transient_model_error(
        ValueError("invalid tool schema")
    )


@pytest.mark.asyncio
async def test_model_middleware_retries_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MIDDLEWARE_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SEC", "0")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")
    attempts = 0
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="test")],
        tools=[],
    )

    async def handler(_: ModelRequest) -> AIMessage:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("429 rate limit")
        return AIMessage(content="ok")

    response = await resilient_model_call.awrap_model_call(
        request,
        handler,
    )

    assert attempts == 3
    assert response.content == "ok"
