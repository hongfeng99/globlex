from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from app.tools.shopping_summary import shopping_summary
from app.config import env_float, env_int
from app.harness.phase_machine import (
    PHASE_TOOLS,
    Phase,
    phase_machine,
)
from app.harness.tool_filter import filter_tools_for_phase
from app.agent.request_context import (
    get_category_insight_state,
    get_original_request,
    platforms_in_text,
)


MAX_TOOL_RESULT_CHARS = 16_000


def truncate_tool_result(
    value: Any,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
) -> Any:
    """Limit oversized tool observations before they re-enter the loop."""

    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    removed = len(value) - max_chars
    return (
        value[:max_chars]
        + f"\n\n[工具结果已截断，省略 {removed} 个字符]"
    )


class LoopDetected(RuntimeError):
    pass


class LoopDetector:
    """Detect repeated identical tool calls in a rolling window."""

    def __init__(
        self,
        window: int = 6,
        repeat_threshold: int = 4,
    ) -> None:
        self._calls: deque[str] = deque(maxlen=window)
        self.repeat_threshold = repeat_threshold

    def observe(
        self,
        tool_name: str,
        args: Any,
    ) -> None:
        fingerprint = sha256(
            f"{tool_name}:{args!r}".encode()
        ).hexdigest()
        self._calls.append(fingerprint)
        if (
            sum(
                call == fingerprint
                for call in self._calls
            )
            >= self.repeat_threshold
        ):
            raise LoopDetected(
                f"检测到重复工具调用：{tool_name}"
            )


_model_gate: asyncio.Semaphore | None = None
_model_gate_loop: asyncio.AbstractEventLoop | None = None
_model_gate_limit: int | None = None


def _get_model_gate() -> asyncio.Semaphore:
    global _model_gate, _model_gate_loop, _model_gate_limit

    loop = asyncio.get_running_loop()
    limit = env_int("LLM_MAX_CONCURRENCY", 2, minimum=1)
    if (
        _model_gate is None
        or _model_gate_loop is not loop
        or _model_gate_limit != limit
    ):
        _model_gate = asyncio.Semaphore(limit)
        _model_gate_loop = loop
        _model_gate_limit = limit
    return _model_gate


def is_transient_model_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "rate limit",
            "429",
            "choices",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection error",
            "模型服务繁忙",
        )
    )


@wrap_model_call
async def resilient_model_call(
    request: ModelRequest,
    handler: Callable[
        [ModelRequest],
        Awaitable[ModelResponse | AIMessage],
    ],
) -> ModelResponse | AIMessage:
    """Throttle model calls and retry transient compatible-API errors."""

    max_attempts = env_int(
        "LLM_MIDDLEWARE_MAX_ATTEMPTS",
        3,
        minimum=1,
    )
    backoff = env_float(
        "LLM_RETRY_BACKOFF_SEC",
        2.0,
        minimum=0,
    )
    for attempt in range(max_attempts):
        try:
            async with _get_model_gate():
                return await handler(request)
        except Exception as exc:
            if (
                not is_transient_model_error(exc)
                or attempt + 1 >= max_attempts
            ):
                if is_transient_model_error(exc):
                    raise RuntimeError(
                        "模型服务繁忙或返回异常格式，"
                        f"已自动重试 {max_attempts} 次。"
                    ) from exc
                raise
            await asyncio.sleep(backoff * (2**attempt))

    raise AssertionError("unreachable")


def should_force_shopping_summary(
    messages: list[Any],
) -> bool:
    """Return true only immediately after a successful ItemPicker call."""

    if not messages:
        return False
    last_message = messages[-1]
    return (
        isinstance(last_message, ToolMessage)
        and last_message.name == "item_picker"
        and last_message.status == "success"
    )


def _completed_tool_names(messages: list[Any]) -> set[str]:
    return {
        message.name
        for message in messages
        if isinstance(message, ToolMessage)
        and message.status == "success"
    }


def _attempted_tool_names(messages: list[Any]) -> set[str]:
    return {
        message.name
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name is not None
    }


def _state_messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    return messages if isinstance(messages, list) else []


def synchronize_phase(messages: list[Any]) -> Phase:
    """Advance the request-local phase from successful tool observations."""

    if phase_machine.is_fixed():
        return phase_machine.get_current_phase()

    completed = _completed_tool_names(messages)
    current = phase_machine.get_current_phase()
    while True:
        if current == Phase.PLANNING and "planner" in completed:
            current = Phase.SEARCHING
        elif current == Phase.SEARCHING and completed & {
            "item_search",
            "dispatch_tool",
        }:
            current = Phase.COMPARING
        elif current == Phase.COMPARING and "item_picker" in completed:
            current = Phase.CONCLUDING
        else:
            break
        phase_machine.transition(current)
    return current


@wrap_model_call
async def enforce_phase_tools(
    request: ModelRequest,
    handler: Callable[
        [ModelRequest],
        Awaitable[ModelResponse | AIMessage],
    ],
) -> ModelResponse | AIMessage:
    """Expose only the tool schemas permitted in the current phase."""

    phase = synchronize_phase(request.messages)
    filtered_tools = filter_tools_for_phase(
        request.tools,
        phase,
    )
    existing_prompt = (
        request.system_message.text
        if request.system_message is not None
        else ""
    )
    phase_prompt = SystemMessage(
        content=(
            existing_prompt
            + "\n\n<phase_permissions>"
            f"当前阶段：{phase.value}。"
            "当前可见且可调用的工具仅限："
            f"{', '.join(sorted(PHASE_TOOLS[phase]))}。"
            "不得尝试调用未显示的工具。"
            "PLANNING 完成 Planner 后进入 SEARCHING；"
            "SEARCHING 获得候选后进入 COMPARING；"
            "COMPARING 必须依次完成 price_compare、shipping_calc、"
            "item_picker；之后进入 CONCLUDING 并调用 shopping_summary。"
            "</phase_permissions>"
        )
    )
    return await handler(
        request.override(
            tools=filtered_tools,
            tool_choice=phase_tool_choice(
                phase,
                request.messages,
            ),
            system_message=phase_prompt,
        )
    )


_TOOL_PREREQUISITES: dict[str, set[str]] = {
    "price_compare": {"item_search", "dispatch_tool"},
    "shipping_calc": {"price_compare"},
    "item_picker": {"shipping_calc"},
    "shopping_summary": {"item_picker"},
}


def _missing_prerequisites(
    tool_name: str,
    completed: set[str],
) -> set[str]:
    required = _TOOL_PREREQUISITES.get(tool_name, set())
    if tool_name == "price_compare":
        return set() if required & completed else required
    return required - completed


def phase_tool_choice(
    phase: Phase,
    messages: list[Any],
) -> str | None:
    """Require progress in root phases while allowing child recall reflection."""

    completed = _completed_tool_names(messages)
    if phase_machine.is_fixed():
        if phase == Phase.SEARCHING and "item_search" not in completed:
            return "item_search"
        return None
    if phase == Phase.COMPARING:
        if "price_compare" not in completed:
            return "price_compare"
        if "shipping_calc" not in completed:
            return "shipping_calc"
        return "item_picker"
    if phase == Phase.CONCLUDING:
        return "shopping_summary"
    if phase == Phase.SEARCHING:
        insight_state = get_category_insight_state()
        attempted = _attempted_tool_names(messages)
        if (
            "category_insight" not in attempted
            and (
                insight_state is None
                or not insight_state.attempted
            )
        ):
            return "category_insight"
        if (
            insight_state is None
            or not insight_state.effective
        ) and "web_search" not in attempted:
            return "web_search"
        mentioned = platforms_in_text(
            get_original_request()
        )
        return (
            "item_search"
            if len(mentioned) == 1
            else "dispatch_tool"
        )
    # PLANNING 需在 Planner/ChatFallback/背景工具中选择；
    # 由模型在结构化规划、闲聊和可选背景信息之间选择。
    return "required"


@wrap_tool_call
async def enforce_phase_tool_call(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        Awaitable[Any],
    ],
) -> Any:
    """Defensively reject hidden or out-of-order tool calls."""

    messages = _state_messages(request.state)
    phase = synchronize_phase(messages)
    tool_name = str(request.tool_call.get("name", ""))
    completed = _completed_tool_names(messages)
    missing = _missing_prerequisites(tool_name, completed)
    if tool_name not in PHASE_TOOLS[phase] or missing:
        reason = (
            f"工具 {tool_name} 在阶段 {phase.value} 不可用。"
            if tool_name not in PHASE_TOOLS[phase]
            else (
                f"工具 {tool_name} 缺少前置工具："
                + ", ".join(sorted(missing))
                + "。"
            )
        )
        return ToolMessage(
            content=(
                reason
                + " 当前可用工具："
                + ", ".join(sorted(PHASE_TOOLS[phase]))
            ),
            name=tool_name or None,
            tool_call_id=str(
                request.tool_call.get("id", "phase-guard")
            ),
            status="error",
        )
    return await handler(request)


@wrap_model_call
async def enforce_terminal_summary(
    request: ModelRequest,
    handler: Callable[
        [ModelRequest],
        Awaitable[ModelResponse | AIMessage],
    ],
) -> ModelResponse | AIMessage:
    """Force the deterministic ItemPicker -> ShoppingSummary transition."""

    if not should_force_shopping_summary(
        request.messages
    ):
        return await handler(request)

    existing_prompt = (
        request.system_message.text
        if request.system_message is not None
        else ""
    )
    forced_prompt = SystemMessage(
        content=(
            existing_prompt
            + "\n\n<forced_transition>"
            "ItemPicker 已完成。你现在必须且只能调用 "
            "shopping_summary；使用工具结果中的 picks，"
            "user_request 填写用户原始购物需求，并把本轮明确"
            "表达的可复用偏好写入 learned_preferences。"
            "不得调用其他工具或直接输出文字。"
            "</forced_transition>"
        )
    )
    forced_request = request.override(
        tools=[shopping_summary],
        tool_choice=shopping_summary.name,
        system_message=forced_prompt,
    )
    return await handler(forced_request)


AGENT_MIDDLEWARE = [
    resilient_model_call,
    enforce_phase_tools,
    enforce_phase_tool_call,
    enforce_terminal_summary,
]


__all__ = [
    "AGENT_MIDDLEWARE",
    "LoopDetected",
    "LoopDetector",
    "enforce_phase_tool_call",
    "enforce_phase_tools",
    "enforce_terminal_summary",
    "is_transient_model_error",
    "phase_tool_choice",
    "resilient_model_call",
    "should_force_shopping_summary",
    "synchronize_phase",
    "truncate_tool_result",
]
