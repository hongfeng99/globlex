from app.agent.request_context import (
    bind_request_context,
    get_original_request,
    get_search_candidates,
    has_search_observations,
    record_search_candidates,
    reserve_dispatch,
)


def test_request_context_limits_and_deduplicates_platforms() -> None:
    with bind_request_context("四个平台搜索机械键盘"):
        assert get_original_request() == "四个平台搜索机械键盘"
        assert reserve_dispatch("搜索 Amazon")[0]
        assert reserve_dispatch("搜索 Shopee")[0]
        assert reserve_dispatch("再次搜索 Amazon") == (
            False,
            "平台 amazon 已经派发，拒绝重复子任务。",
        )
        assert reserve_dispatch("搜索 AliExpress")[0]
        assert reserve_dispatch("搜索 eBay")[0]
        allowed, reason = reserve_dispatch("额外汇总任务")
        assert allowed is False
        assert reason == "本轮最多派发四个平台子任务。"

    assert get_original_request() == ""


def test_request_context_collects_structured_candidates() -> None:
    with bind_request_context("四个平台搜索骑行三件套"):
        assert not has_search_observations()
        record_search_candidates(
            [
                {
                    "item_id": "real-1",
                    "platform": "shopee",
                    "title": "真实离线商品名",
                }
            ]
        )
        record_search_candidates([])

        assert has_search_observations()
        assert get_search_candidates() == [
            {
                "item_id": "real-1",
                "platform": "shopee",
                "title": "真实离线商品名",
            }
        ]

    assert get_search_candidates() == []
