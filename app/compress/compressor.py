from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.agent.llm import get_llm
from app.compress.breakpoint import (
    compression_boundary,
)


async def compress_messages(
    messages: list[Any],
    *,
    max_chars: int = 48_000,
    keep_recent: int = 8,
) -> list[Any]:
    boundary = compression_boundary(
        messages,
        max_chars=max_chars,
        keep_recent=keep_recent,
    )
    if boundary == 0:
        return messages

    old_text = "\n".join(
        f"{getattr(message, 'type', 'message')}: "
        f"{getattr(message, 'content', message)}"
        for message in messages[:boundary]
    )
    response = await get_llm().ainvoke(
        [
            SystemMessage(
                content=(
                    "把以下旧对话压缩成可供购物 Agent "
                    "继续执行的事实摘要。保留用户约束、"
                    "候选商品、工具结论和未完成事项。"
                )
            ),
            HumanMessage(content=old_text),
        ]
    )
    summary = response.content
    if not isinstance(summary, str):
        summary = str(summary)
    return [
        SystemMessage(
            content=f"[旧上下文摘要]\n{summary}"
        ),
        *messages[boundary:],
    ]


__all__ = ["compress_messages"]
