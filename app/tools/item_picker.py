from __future__ import annotations

import time
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.api.monitor import monitor
from app.tools.category_insight import CategoryInsightOutput
from app.tools.shipping_calc import LandedCost


class PickedItem(BaseModel):
    item_id: str
    platform: str
    landed_cny: float
    score: float
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ItemPickerOutput(BaseModel):
    picks: list[PickedItem]
    rejected_brief: list[str] = Field(default_factory=list)


def _check_preferences(
    item: LandedCost,
    preferences: list[str],
) -> tuple[bool, list[str]]:
    flags: list[str] = []
    joined = " ".join(preferences).lower()

    # 示例数据用后缀表达材质；生产环境应改为读取商品 attributes。
    if (
        ("不要塑料" in joined or "no plastic" in joined)
        and item.item_id.upper().endswith("-PLASTIC")
    ):
        flags.append("违反硬约束：不要塑料")
        return False, flags

    return True, flags


def _score(
    item: LandedCost,
    insight: CategoryInsightOutput | None,
    preferences: list[str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if insight is not None:
        mid_tiers = [
            tier
            for tier in insight.price_tiers
            if tier.tier == "mid"
        ]
        if any(
            low <= item.landed_cny <= high
            for low, high in (
                tier.range_cny for tier in mid_tiers
            )
        ):
            score += 0.4
            reasons.append("到手价位于主流价格带")

    if item.eta_days <= 12:
        score += 0.2
        reasons.append("预计到货较快")

    if item.duty_cny == 0:
        score += 0.2
        reasons.append("预计免税")

    joined = " ".join(preferences).lower()
    if (
        "小众" in joined or "small brand" in joined
    ) and item.platform in {"shopee", "aliexpress"}:
        score += 0.2
        reasons.append("更符合小众品牌偏好")

    # 同分时让低到手价略占优势，但不改变章节中的主要权重。
    score += 1 / (1 + item.landed_cny) / 100
    return round(score, 4), reasons[:3]


@tool
async def item_picker(
    items: list[LandedCost],
    preferences: list[str] | None = None,
    insight: CategoryInsightOutput | None = None,
    max_picks: Literal[1, 2, 3] = 3,
) -> ItemPickerOutput:
    """根据硬约束、品类价格带、时效和税费，从候选中精挑最多 3 件。"""

    preferences = preferences or []
    items = items[:30]
    await monitor.report_tool_start(
        "item_picker",
        {
            "items_count": len(items),
            "max_picks": max_picks,
        },
    )
    started_at = time.perf_counter()

    accepted: list[PickedItem] = []
    rejected: list[str] = []
    for item in items:
        allowed, flags = _check_preferences(
            item, preferences
        )
        if not allowed:
            rejected.append(
                f"{item.item_id}: {'；'.join(flags)}"
            )
            continue

        score, reasons = _score(
            item, insight, preferences
        )
        accepted.append(
            PickedItem(
                item_id=item.item_id,
                platform=item.platform,
                landed_cny=item.landed_cny,
                score=score,
                reasons=reasons,
                flags=flags,
            )
        )

    accepted.sort(
        key=lambda item: (
            -item.score,
            item.landed_cny,
        )
    )
    await monitor.report_tool_end(
        "item_picker",
        int(
            (time.perf_counter() - started_at)
            * 1000
        ),
    )
    return ItemPickerOutput(
        picks=accepted[:max_picks],
        rejected_brief=rejected[:10],
    )


__all__ = [
    "ItemPickerOutput",
    "PickedItem",
    "item_picker",
]
