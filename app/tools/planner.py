from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.agent.prompts import get_planner_prompt


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

    model = get_llm().with_structured_output(
        ShoppingIntent
    )
    return await model.ainvoke(
        [
            SystemMessage(
                content=get_planner_prompt()
            ),
            HumanMessage(content=user_request),
        ]
    )


__all__ = ["ShoppingIntent", "planner"]
