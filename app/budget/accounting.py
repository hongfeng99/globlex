from __future__ import annotations

from app.budget.token_budget import get_budget


async def account_llm_usage(
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    budget = get_budget()
    if budget is not None:
        budget.consume(
            prompt_tokens + completion_tokens
        )


def account_tool_result(
    result_text: str,
) -> None:
    budget = get_budget()
    if budget is not None:
        budget.consume(
            max(1, len(result_text) // 3)
        )


__all__ = [
    "account_llm_usage",
    "account_tool_result",
]
