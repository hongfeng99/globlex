from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.agent.prompts import get_planner_prompt
from app.api.monitor import monitor


class ShoppingIntent(BaseModel):
    query: str
    category: str | None = None
    budget: float | None = None
    platforms: list[str] = Field(default_factory=list)
    material_pref: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "exclude": [],
            "prefer": [],
        }
    )
    style_pref: str | None = None
    hard_constraints: list[str] = Field(
        default_factory=list
    )
    soft_preferences: list[str] = Field(
        default_factory=list
    )


@tool
async def planner(user_request: str) -> ShoppingIntent:
    """将自然语言购物需求拆解为稳定的结构化意图字段。"""

    await monitor.report_tool_start(
        "planner",
        {"user_request": user_request[:200]},
    )
    started_at = time.perf_counter()
    model = get_llm().with_structured_output(
        ShoppingIntent
    )
    result = await model.ainvoke(
        [
            SystemMessage(
                content=get_planner_prompt()
            ),
            HumanMessage(content=user_request),
        ]
    )
    await monitor.report_tool_end(
        "planner",
        int((time.perf_counter() - started_at) * 1000),
    )
    return result


__all__ = ["ShoppingIntent", "planner"]
