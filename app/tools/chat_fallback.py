from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.agent.llm import get_llm


@tool
async def chat_fallback(message: str) -> str:
    """非购物问题的闲聊兜底；调用后直接把回答交给用户。"""

    response = await get_llm().ainvoke(
        [HumanMessage(content=message)]
    )
    return (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )


__all__ = ["chat_fallback"]
