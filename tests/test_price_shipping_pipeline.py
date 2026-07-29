import pytest

from app.tools.item_search import Candidate
from app.tools.price_compare import (
    price_compare,
)
from app.tools.shipping_calc import (
    shipping_calc,
)


@pytest.mark.asyncio
async def test_price_compare_to_shipping_pipeline() -> None:
    candidates = [
        Candidate(
            item_id="amazon-1",
            platform="amazon",
            title="Amazon 收纳袋",
            price=30.0,
            currency="USD",
        ),
        Candidate(
            item_id="ali-1",
            platform="aliexpress",
            title="AliExpress 收纳袋",
            price=240.0,
            currency="CNY",
        ),
    ]

    compared = await price_compare.ainvoke(
        {
            "candidates": candidates,
            "top_n": 12,
        }
    )
    landed = await shipping_calc.ainvoke(
        {
            "points": compared.ranked,
            "destination": "CN",
        }
    )

    assert len(compared.ranked) == 2
    assert len(landed.items) == 2
    assert {
        item.item_id
        for item in landed.items
    } == {
        "amazon-1",
        "ali-1",
    }
