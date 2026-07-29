from __future__ import annotations

from typing import Any

from app.agent.llm import (
    get_lite_llm,
    get_llm,
    get_minimal_llm,
)
from app.budget.token_budget import get_budget
from app.compress.breakpoint import (
    compression_boundary,
)
from app.compress.compressor import compress_messages


MINIMAL_HINT = """
[系统提示：当前请求 Token 预算紧张，请遵循以下约束]
- Think 阶段不要展开详细推理，直接给出结论。
- 不要再次解析全部工具说明。
- 基于已有 Observation 结果直接完成最终回答。
- 优先调用 ShoppingSummary 或 ChatFallback 收尾。
"""


def get_current_model() -> Any | None:
    budget = get_budget()
    tier = (
        budget.model_tier
        if budget is not None
        else "main"
    )
    if tier == "main":
        return get_llm()
    if tier == "lite":
        return get_lite_llm()
    if tier == "minimal":
        return get_minimal_llm()
    return None


def inject_budget_hint(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    budget = get_budget()
    if (
        budget is None
        or budget.model_tier != "minimal"
    ):
        return messages
    if (
        messages
        and messages[0].get("role")
        == "system"
    ):
        copied = [dict(message) for message in messages]
        copied[0]["content"] = (
            str(copied[0].get("content", ""))
            + MINIMAL_HINT
        )
        return copied
    return [
        {"role": "system", "content": MINIMAL_HINT},
        *messages,
    ]


async def budget_aware_compress(
    messages: list[Any],
) -> list[Any]:
    budget = get_budget()
    if budget is None:
        return messages
    keep_recent = (
        3
        if budget.remaining_ratio > 0.50
        else 1
    )
    boundary = compression_boundary(
        messages,
        max_chars=48_000,
        keep_recent=keep_recent,
    )
    if boundary or budget.remaining_ratio < 0.30:
        return await compress_messages(
            messages,
            max_chars=24_000,
            keep_recent=keep_recent,
        )
    return messages


__all__ = [
    "MINIMAL_HINT",
    "budget_aware_compress",
    "get_current_model",
    "inject_budget_hint",
]
