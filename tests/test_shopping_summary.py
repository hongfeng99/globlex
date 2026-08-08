from __future__ import annotations

from typing import Any

import pytest

import app.tools.shopping_summary as summary_module
from app.agent.request_context import bind_request_context
from app.memory.context import bind_user_context
from app.memory.store import preference_store
from app.tools.shopping_summary import shopping_summary
from app.tools.shopping_summary import (
    _normalize_learned_preferences,
    _normalize_picks,
)


class FakeLlm:
    async def ainvoke(self, messages: list[Any]) -> Any:
        return type("Response", (), {"content": "已生成购物清单"})()


class LeakyLlm:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        return type(
            "Response",
            (),
            {
                "content": (
                    "Shopee 商品 (ID: shp_004)，¥281.54，可直邮"
                )
            },
        )()


def test_summary_accepts_json_encoded_preferences() -> None:
    assert _normalize_learned_preferences(
        '["偏好静音轴", "不要 RGB 灯"]'
    ) == ["偏好静音轴", "不要 RGB 灯"]
    assert _normalize_learned_preferences("偏好三模") == ["偏好三模"]


def test_summary_accepts_json_encoded_picks() -> None:
    picks = _normalize_picks(
        '[{"item_id":"item-1","platform":"shopee",'
        '"landed_cny":188.2,"score":0.8}]'
    )
    assert len(picks) == 1
    assert picks[0].item_id == "item-1"
    assert picks[0].landed_cny == 188.2


@pytest.mark.asyncio
async def test_summary_persists_learned_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "summary-preference-user"

    async def noop(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(summary_module, "get_llm", lambda: FakeLlm())
    monkeypatch.setattr(
        summary_module.monitor, "report_tool_start", noop
    )
    monkeypatch.setattr(
        summary_module.monitor, "report_tool_end", noop
    )

    with bind_user_context(user_id):
        output = await shopping_summary.ainvoke(
            {
                "picks": [],
                "user_request": "推荐键盘",
                "learned_preferences": [
                    "偏好静音轴",
                    "不要 RGB 灯",
                ],
            }
        )

    entries = await preference_store.get(user_id)
    assert output.learned_preferences == [
        "偏好静音轴",
        "不要 RGB 灯",
    ]
    assert {entry.preference for entry in entries} == {
        "偏好静音轴",
        "不要 RGB 灯",
    }


@pytest.mark.asyncio
async def test_summary_never_relaxes_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def noop(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(summary_module, "get_llm", lambda: FakeLlm())
    monkeypatch.setattr(
        summary_module.monitor, "report_tool_start", noop
    )
    monkeypatch.setattr(
        summary_module.monitor, "report_tool_end", noop
    )

    with bind_request_context("200元以内的机械键盘"):
        output = await shopping_summary.ainvoke(
            {
                "picks": [
                    {
                        "item_id": "over-budget",
                        "platform": "amazon",
                        "landed_cny": 410,
                        "score": 0.9,
                    }
                ],
                "user_request": "200元以内的机械键盘",
            }
        )

    assert output.picks == []
    assert "未自动放宽预算" in output.final_text


@pytest.mark.asyncio
async def test_summary_uses_real_title_and_hides_internal_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = LeakyLlm()

    async def noop(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(summary_module, "get_llm", lambda: llm)
    monkeypatch.setattr(
        summary_module.monitor, "report_tool_start", noop
    )
    monkeypatch.setattr(
        summary_module.monitor, "report_tool_end", noop
    )
    pick = {
        "item_id": "offline-shopee-cycling-kit-004",
        "platform": "shopee",
        "title": "原野实验室 耐力骑行三件套",
        "price_cny": 220,
        "shipping_cny": 35,
        "duty_cny": 20,
        "landed_cny": 275,
        "eta_days": 12,
        "rating": 4.8,
        "sales": 2600,
        "attributes": {
            "components": ["骑行上衣", "背带骑行裤", "骑行手套"],
            "material": "速干聚酯纤维",
            "features": ["吸湿排汗", "反光条"],
        },
        "score": 0.8,
        "reasons": ["速干聚酯纤维、吸湿排汗"],
    }

    with bind_request_context("骑行三件套，500元以下"):
        output = await shopping_summary.ainvoke(
            {
                "picks": [pick],
                "user_request": "骑行三件套，500元以下",
            }
        )

    payload = str(llm.messages[-1].content)
    assert "item_id" not in payload
    assert "offline-shopee" not in payload
    assert "原野实验室 耐力骑行三件套" in output.final_text
    assert "骑行上衣、背带骑行裤、骑行手套" in output.final_text
    assert "Shopee" in output.final_text
    assert "ID:" not in output.final_text
    assert "shp_004" not in output.final_text
    assert "可直邮" not in output.final_text
