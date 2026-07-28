from pathlib import Path
from typing import Any

import pytest

import app.tools.item_search as item_module
from app.tools.item_search import (
    _dedupe_and_rerank,
    item_search,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


def _raw_candidate(
    item_id: str,
    score: float,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "platform": "amazon",
        "title": f"商品 {item_id}",
        "price": 99.0,
        "currency": "CNY",
        "attributes": {
            "material": "nylon",
        },
        "score": score,
    }


def test_dedupe_and_rerank() -> None:
    merged = _dedupe_and_rerank(
        [
            _raw_candidate("a", 0.8),
            _raw_candidate("b", 0.7),
        ],
        [
            _raw_candidate("b", 0.9),
            _raw_candidate("c", 0.6),
        ],
    )

    assert [
        item["item_id"]
        for item in merged
    ] == ["b", "a", "c"]
    assert merged[0]["boost"] == pytest.approx(
        1.15
    )


@pytest.mark.asyncio
async def test_item_search_output_and_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []

    async def fake_recall(
        query: str,
        platform: str,
        top_k: int,
        user_id: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
        assert query == "旅行收纳袋"
        assert platform == "amazon"
        assert top_k == 50
        assert user_id == "user-1"
        return [
            _raw_candidate("a", 0.9)
        ], 60

    async def fake_start(
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        events.append(("start", args))
        return True

    async def fake_end(
        tool_name: str,
        duration_ms: int,
    ) -> bool:
        events.append(("end", duration_ms))
        return True

    monkeypatch.setattr(
        item_module,
        "_recall",
        fake_recall,
    )
    monkeypatch.setattr(
        item_module.monitor,
        "report_tool_start",
        fake_start,
    )
    monkeypatch.setattr(
        item_module.monitor,
        "report_tool_end",
        fake_end,
    )

    with bind_thread_context(
        "thread-A",
        tmp_path,
    ):
        output = await item_search.ainvoke(
            {
                "query": " 旅行收纳袋 ",
                "platform": "amazon",
                "top_k": 100,
                "user_id": " user-1 ",
            }
        )

    assert output.platform == "amazon"
    assert output.total_recall == 60
    assert output.truncated is True
    assert output.candidates[0].item_id == "a"
    assert events[0][0] == "start"
    assert events[1][0] == "end"


@pytest.mark.asyncio
async def test_recall_without_user_is_semantic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_semantic(
        query: str,
        platform: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        return [_raw_candidate("a", 0.9)]

    async def unexpected_personalized(
        query: str,
        platform: str,
        top_k: int,
        user_id: str,
    ) -> list[dict[str, Any]]:
        raise AssertionError(
            "没有 user_id 时不应调用个性化召回。"
        )

    monkeypatch.setattr(
        item_module,
        "_semantic_recall",
        fake_semantic,
    )
    monkeypatch.setattr(
        item_module,
        "_personalized_recall",
        unexpected_personalized,
    )

    raw, total = await item_module._recall(
        "query",
        "amazon",
        20,
        None,
    )

    assert total == 1
    assert raw[0]["item_id"] == "a"



@pytest.mark.asyncio
async def test_recall_overfetches_one_for_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_top_k: list[int] = []

    async def fake_semantic(
        query: str,
        platform: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        requested_top_k.append(top_k)

        return [
            _raw_candidate(
                item_id=f"item-{index}",
                score=1.0 - index / 100,
            )
            for index in range(top_k)
        ]

    monkeypatch.setattr(
        item_module,
        "_semantic_recall",
        fake_semantic,
    )

    raw_candidates, total_recall = (
        await item_module._recall(
            query="旅行收纳袋",
            platform="amazon",
            top_k=20,
            user_id=None,
        )
    )

    # 内部应多取一条，用于判断是否发生截断。
    assert requested_top_k == [21]

    # 对外仍然只返回 20 条。
    assert len(raw_candidates) == 20

    # 截断前实际获得了 21 条。
    assert total_recall == 21