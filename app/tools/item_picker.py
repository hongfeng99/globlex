from __future__ import annotations

import json
import time
from typing import Any, Literal

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
    title: str = ""
    price_cny: float = 0.0
    shipping_cny: float = 0.0
    duty_cny: float = 0.0
    eta_days: int | None = None
    rating: float | None = None
    sales: int | None = None
    attributes: dict[str, Any] = Field(
        default_factory=dict
    )
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ItemPickerOutput(BaseModel):
    picks: list[PickedItem]
    rejected_brief: list[str] = Field(default_factory=list)


def _decode_json_value(value: str, field_name: str) -> object:
    try:
        return json.loads(value.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{field_name} 必须是有效的 JSON 字符串。"
        ) from exc


def _normalize_items(
    value: list[LandedCost] | str,
) -> list[LandedCost]:
    raw_items: object = (
        _decode_json_value(value, "items")
        if isinstance(value, str)
        else value
    )
    if not isinstance(raw_items, list):
        raise ValueError("items 必须是商品数组。")
    return [
        item
        if isinstance(item, LandedCost)
        else LandedCost.model_validate(item)
        for item in raw_items
    ]


def _normalize_preferences(
    value: list[str] | str | None,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            raw_values: object = json.loads(value.strip())
        except json.JSONDecodeError:
            raw_values = value
    else:
        raw_values = value
    if not isinstance(raw_values, list):
        raw_values = [raw_values]
    return [
        text
        for item in raw_values
        if (text := str(item).strip())
    ]


def _normalize_insight(
    value: CategoryInsightOutput | dict[str, Any] | str | None,
) -> CategoryInsightOutput | None:
    if value is None:
        return None
    raw_value: object = (
        _decode_json_value(value, "insight")
        if isinstance(value, str)
        else value
    )
    return (
        raw_value
        if isinstance(raw_value, CategoryInsightOutput)
        else CategoryInsightOutput.model_validate(raw_value)
    )


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

    attributes = item.attributes
    material = str(attributes.get("material", "")).strip()
    raw_features = attributes.get("features", [])
    features = (
        [str(feature).strip() for feature in raw_features]
        if isinstance(raw_features, list)
        else []
    )
    product_details = "、".join(
        detail
        for detail in (
            material,
            "、".join(features[:2]),
        )
        if detail
    )
    if product_details:
        reasons.append(product_details)

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
    items: list[LandedCost] | str,
    preferences: list[str] | str | None = None,
    insight: CategoryInsightOutput | dict[str, Any] | str | None = None,
    max_picks: Literal[1, 2, 3] = 3,
) -> ItemPickerOutput:
    """根据硬约束、品类价格带、时效和税费，从候选中精挑最多 3 件。"""

    items = _normalize_items(items)
    preferences = _normalize_preferences(preferences)
    insight = _normalize_insight(insight)
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
                title=item.title,
                price_cny=item.price_cny,
                shipping_cny=item.shipping_cny,
                duty_cny=item.duty_cny,
                eta_days=item.eta_days,
                rating=item.rating,
                sales=item.sales,
                attributes=dict(item.attributes),
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
    "_normalize_insight",
    "_normalize_items",
    "_normalize_preferences",
    "item_picker",
]
