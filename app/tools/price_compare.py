from __future__ import annotations

import time

from langchain_core.tools import tool
from pydantic import BaseModel

from app.api.monitor import monitor
from app.recall.fx import to_base
from app.tools.item_search import Candidate


class PricePoint(BaseModel):
    """
    币种归一后的商品价格点。
    """

    item_id: str
    platform: str
    title: str
    price_local: float
    currency_local: str
    price_cny: float
    rating: float | None = None
    sales: int | None = None
    note: str | None = None


class PriceCompareOutput(BaseModel):
    """
    跨平台比价结果。
    """

    base_currency: str = "CNY"
    ranked: list[PricePoint]
    cheapest_per_platform: dict[str, str]


def _pack_note(
    candidate: Candidate,
) -> str | None:
    """
    从 attributes 中识别“一套 N 件”的特殊计价信息。
    """

    pack_size = candidate.attributes.get(
        "pack_size"
    )

    if (
        isinstance(pack_size, int)
        and not isinstance(pack_size, bool)
        and pack_size > 1
    ):
        unit_price = round(
            candidate.price / pack_size,
            2,
        )
        return (
            f"一套 {pack_size} 件，"
            f"等价单件 {unit_price} "
            f"{candidate.currency}"
        )

    return None


@tool
async def price_compare(
    candidates: list[Candidate],
    base_currency: str = "CNY",
    top_n: int = 12,
) -> PriceCompareOutput:
    """
    跨平台候选商品比价，输出币种归一后的排序。

    Args:
        candidates: ItemSearch 合流后的候选集，最多处理
            100 件。
        base_currency: 归一目标币种，默认人民币。
        top_n: 返回排序后的前 N 件，默认 12，最大 30。

    Returns:
        ranked 为按归一价格升序排列的 PricePoint；
        cheapest_per_platform 保存各平台最低价 item_id。
    """

    if top_n <= 0:
        raise ValueError(
            "top_n 必须大于 0。"
        )

    normalized_base = (
        base_currency.strip().upper()
    )

    if not normalized_base:
        raise ValueError(
            "base_currency 不能为空字符串。"
        )

    top_n = min(top_n, 30)
    candidates = candidates[:100]

    await monitor.report_tool_start(
        "price_compare",
        {
            "candidates_count": len(
                candidates
            ),
            "base_currency": (
                normalized_base
            ),
        },
    )

    started_at = time.perf_counter()
    points: list[PricePoint] = []

    for candidate in candidates:
        try:
            normalized_price = to_base(
                candidate.price,
                candidate.currency,
                normalized_base,
            )
        except ValueError:
            # 单个未知币种不阻断整批候选的比价。
            continue

        points.append(
            PricePoint(
                item_id=candidate.item_id,
                platform=candidate.platform,
                title=candidate.title,
                price_local=candidate.price,
                currency_local=(
                    candidate.currency.upper()
                ),
                price_cny=round(
                    normalized_price,
                    2,
                ),
                rating=candidate.rating,
                sales=candidate.sales,
                note=_pack_note(candidate),
            )
        )

    points.sort(
        key=lambda point: point.price_cny
    )
    ranked = points[:top_n]

    cheapest: dict[str, str] = {}

    # points 已按价格升序，因此每个平台首次出现的候选
    # 就是该平台最低价。
    for point in points:
        if point.platform not in cheapest:
            cheapest[point.platform] = (
                point.item_id
            )

    duration_ms = int(
        (
            time.perf_counter()
            - started_at
        )
        * 1000
    )
    await monitor.report_tool_end(
        "price_compare",
        duration_ms,
    )

    return PriceCompareOutput(
        base_currency=normalized_base,
        ranked=ranked,
        cheapest_per_platform=cheapest,
    )


__all__ = [
    "PriceCompareOutput",
    "PricePoint",
    "price_compare",
]
