from __future__ import annotations

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.agent.prompts import get_shopping_summary_prompt
from app.api.monitor import monitor
from app.tools.item_picker import PickedItem


class ShoppingSummaryOutput(BaseModel):
    final_text: str
    picks: list[PickedItem]
    learned_preferences: list[str] = Field(
        default_factory=list
    )


def _content_as_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(
        content,
        ensure_ascii=False,
        default=str,
    )


@tool
async def shopping_summary(
    picks: list[PickedItem],
    user_request: str,
    learned_preferences: list[str] | None = None,
) -> ShoppingSummaryOutput:
    """终结性工具：把最多 3 件精挑结果整理成简洁、可执行的购物清单。"""

    picks = picks[:3]
    learned_preferences = (
        learned_preferences or []
    )
    await monitor.report_tool_start(
        "shopping_summary",
        {"picks_count": len(picks)},
    )
    started_at = time.perf_counter()
    payload = {
        "user_request": user_request,
        "picks": [
            pick.model_dump() for pick in picks
        ],
    }
    response = await get_llm().ainvoke(
        [
            SystemMessage(
                content=get_shopping_summary_prompt()
            ),
            HumanMessage(
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            ),
        ]
    )
    final_text = _content_as_text(
        response.content
    )
    await monitor.report_tool_end(
        "shopping_summary",
        int(
            (time.perf_counter() - started_at)
            * 1000
        ),
    )
    return ShoppingSummaryOutput(
        final_text=final_text,
        picks=picks,
        learned_preferences=(
            learned_preferences
        ),
    )


__all__ = [
    "ShoppingSummaryOutput",
    "shopping_summary",
]
