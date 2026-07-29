from pathlib import Path
from typing import Any

import pytest

import app.tools.price_compare as price_module
from app.tools.item_search import Candidate
from app.tools.price_compare import (
    price_compare,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


def _candidate(
    item_id: str,
    platform: str,
    price: float,
    currency: str,
    *,
    pack_size: int | None = None,
) -> Candidate:
    attributes: dict[str, Any] = {}

    if pack_size is not None:
        attributes["pack_size"] = pack_size

    return Candidate(
        item_id=item_id,
        platform=platform,
        title=f"商品 {item_id}",
        price=price,
        currency=currency,
        attributes=attributes,
    )


@pytest.mark.asyncio
async def test_price_compare_ranks_and_explains(
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
        price_module.monitor,
        "report_tool_start",
        fake_start,
    )
    monkeypatch.setattr(
        price_module.monitor,
        "report_tool_end",
        fake_end,
    )

    candidates = [
        _candidate(
            "amazon-1",
            "amazon",
            39.9,
            "USD",
        ),
        _candidate(
            "amazon-2",
            "amazon",
            30.0,
            "USD",
        ),
        _candidate(
            "shopee-1",
            "shopee",
            158.0,
            "SGD",
        ),
        _candidate(
            "ali-1",
            "aliexpress",
            240.0,
            "CNY",
            pack_size=3,
        ),
        _candidate(
            "ignored",
            "ebay",
            1.0,
            "UNKNOWN",
        ),
    ]

    with bind_thread_context(
        "thread-price",
        tmp_path,
    ):
        output = await price_compare.ainvoke(
            {
                "candidates": candidates,
                "base_currency": "cny",
                "top_n": 3,
            }
        )

    assert output.base_currency == "CNY"
    assert [
        point.item_id
        for point in output.ranked
    ] == [
        "amazon-2",
        "ali-1",
        "amazon-1",
    ]
    assert output.cheapest_per_platform == {
        "amazon": "amazon-2",
        "aliexpress": "ali-1",
        "shopee": "shopee-1",
    }
    assert output.ranked[1].note == (
        "一套 3 件，等价单件 80.0 CNY"
    )
    assert events == [
        "start:price_compare",
        "end:price_compare",
    ]


@pytest.mark.asyncio
async def test_price_compare_clamps_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_event(
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        return True

    monkeypatch.setattr(
        price_module.monitor,
        "report_tool_start",
        no_event,
    )
    monkeypatch.setattr(
        price_module.monitor,
        "report_tool_end",
        no_event,
    )

    candidates = [
        _candidate(
            f"item-{index}",
            "amazon",
            float(index + 1),
            "CNY",
        )
        for index in range(40)
    ]

    output = await price_compare.ainvoke(
        {
            "candidates": candidates,
            "top_n": 100,
        }
    )

    assert len(output.ranked) == 30
