from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agent.dispatch_tool import dispatch_tool
from app.tools.category_insight import (
    category_insight,
)
from app.tools.chat_fallback import chat_fallback
from app.tools.item_picker import item_picker
from app.tools.item_search import item_search
from app.tools.planner import planner
from app.tools.price_compare import price_compare
from app.tools.shipping_calc import shipping_calc
from app.tools.shopping_summary import (
    shopping_summary,
)
from app.tools.web_search import web_search


CORE_TOOL_SET: list[BaseTool] = [
    planner,
    chat_fallback,
    web_search,
    category_insight,
    item_search,
    item_picker,
    price_compare,
    shipping_calc,
    shopping_summary,
]
FULL_TOOL_SET: list[BaseTool] = [
    *CORE_TOOL_SET,
    dispatch_tool,
]
TERMINAL_TOOLS = {
    shopping_summary.name,
    chat_fallback.name,
}


__all__ = [
    "CORE_TOOL_SET",
    "FULL_TOOL_SET",
    "TERMINAL_TOOLS",
]
