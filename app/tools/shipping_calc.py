from __future__ import annotations

import time

from langchain_core.tools import tool
from pydantic import BaseModel

from app.api.monitor import monitor
from app.recall.duty import (
    DutyTier,
    estimate_duty,
)
from app.recall.shipping import (
    estimate_shipping,
)
from app.tools.price_compare import PricePoint


class LandedCost(BaseModel):
    """
    单件候选商品的预估到手成本。
    """

    item_id: str
    platform: str
    price_cny: float
    shipping_cny: float
    duty_cny: float
    landed_cny: float
    eta_days: int
    duty_tier: DutyTier


class ShippingCalcOutput(BaseModel):
    """
    运费和关税计算结果。
    """

    destination: str
    items: list[LandedCost]


def _guess_weight_kg(
    point: PricePoint,
) -> float:
    """
    从 PricePoint 估计商品重量。

    第 13 章将通过品类洞察提供真实品类重量；当前章节
    使用 0.5 kg 占位值。
    """

    return 0.5


@tool
async def shipping_calc(
    points: list[PricePoint],
    destination: str = "CN",
) -> ShippingCalcOutput:
    """
    为已完成比价的候选估算到手价。

    Args:
        points: PriceCompare.ranked 的子集，最多处理
            30 件。
        destination: 收货国家 ISO 代码，默认中国大陆。

    Returns:
        items 按 landed_cny 从低到高排列。
    """

    normalized_destination = (
        destination.strip().upper()
    )

    if not normalized_destination:
        raise ValueError(
            "destination 不能为空字符串。"
        )

    points = points[:30]

    await monitor.report_tool_start(
        "shipping_calc",
        {
            "items_count": len(points),
            "destination": (
                normalized_destination
            ),
        },
    )

    started_at = time.perf_counter()
    landed_items: list[LandedCost] = []

    for point in points:
        weight = _guess_weight_kg(point)
        shipping_cny, eta = (
            estimate_shipping(
                weight,
                point.platform,
            )
        )
        duty_cny, duty_tier = (
            estimate_duty(
                point.price_cny,
                point.platform,
            )
        )
        total = round(
            point.price_cny
            + shipping_cny
            + duty_cny,
            2,
        )

        landed_items.append(
            LandedCost(
                item_id=point.item_id,
                platform=point.platform,
                price_cny=point.price_cny,
                shipping_cny=shipping_cny,
                duty_cny=duty_cny,
                landed_cny=total,
                eta_days=eta,
                duty_tier=duty_tier,
            )
        )

    landed_items.sort(
        key=lambda item: item.landed_cny
    )

    duration_ms = int(
        (
            time.perf_counter()
            - started_at
        )
        * 1000
    )
    await monitor.report_tool_end(
        "shipping_calc",
        duration_ms,
    )

    return ShippingCalcOutput(
        destination=normalized_destination,
        items=landed_items,
    )


__all__ = [
    "LandedCost",
    "ShippingCalcOutput",
    "shipping_calc",
]
