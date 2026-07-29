from pathlib import Path
from typing import Any

import pytest

import app.tools.shipping_calc as shipping_module
from app.recall.duty import estimate_duty
from app.recall.shipping import (
    estimate_shipping,
)
from app.tools.price_compare import PricePoint
from app.tools.shipping_calc import shipping_calc
from app.utils.thread_ctx import (
    bind_thread_context,
)


def _price_point(
    item_id: str,
    platform: str,
    price_cny: float,
) -> PricePoint:
    return PricePoint(
        item_id=item_id,
        platform=platform,
        title=f"商品 {item_id}",
        price_local=price_cny,
        currency_local="CNY",
        price_cny=price_cny,
    )


def test_duty_and_shipping_tables() -> None:
    assert estimate_duty(
        240.0,
        "aliexpress",
    ) == (31.2, "标准")
    assert estimate_duty(
        100.0,
        "unknown",
    ) == (13.0, "标准")

    assert estimate_shipping(
        0.5,
        "aliexpress",
    ) == (40.0, 22)
    assert estimate_shipping(
        9.0,
        "ebay",
    ) == (300.0, 10)


@pytest.mark.asyncio
async def test_shipping_calc_landed_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def fake_start(
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        events.append(f"start:{tool_name}")
        return True

    async def fake_end(
        tool_name: str,
        duration_ms: int,
    ) -> bool:
        events.append(f"end:{tool_name}")
        return True

    monkeypatch.setattr(
        shipping_module.monitor,
        "report_tool_start",
        fake_start,
    )
    monkeypatch.setattr(
        shipping_module.monitor,
        "report_tool_end",
        fake_end,
    )

    points = [
        _price_point(
            "amazon-1",
            "amazon",
            200.0,
        ),
        _price_point(
            "ali-1",
            "aliexpress",
            240.0,
        ),
    ]

    with bind_thread_context(
        "thread-shipping",
        tmp_path,
    ):
        output = await shipping_calc.ainvoke(
            {
                "points": points,
                "destination": "cn",
            }
        )

    assert output.destination == "CN"
    assert [
        item.item_id
        for item in output.items
    ] == ["ali-1", "amazon-1"]

    aliexpress = output.items[0]
    assert aliexpress.landed_cny == 311.2

    amazon = output.items[1]
    assert amazon.shipping_cny == 130.0
    assert amazon.duty_cny == 26.0
    assert amazon.landed_cny == 356.0
    assert amazon.eta_days == 10
    assert amazon.duty_tier == "标准"

    assert events == [
        "start:shipping_calc",
        "end:shipping_calc",
    ]
