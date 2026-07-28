from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agent.dispatch_tool import (
    dispatch_tool,
)
from app.tools.item_search import item_search


# 当前章节已经落地的完整工具集。
# 后续章节实现其他核心工具后，只需在这里统一注册。
FULL_TOOL_SET: list[BaseTool] = [
    item_search,
    dispatch_tool,
]


__all__ = [
    "FULL_TOOL_SET",
]
