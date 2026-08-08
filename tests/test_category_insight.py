from typing import Any

import pytest

import app.tools.category_insight as module
from app.recall.category_kb import CategoryCard
from app.tools.category_insight import (
    Bestseller,
    CategoryRecallResult,
    CategoryInsightOutput,
    _build_hybrid_body,
    category_insight,
    is_effective_category_insight,
)
from app.recall.category_norm import (
    find_category_alias,
    normalize_category,
)


def _card(
    card_id: str,
    card_type: str,
    summary: str,
    *,
    evidence: list[str] | None = None,
    confidence: float = 0.8,
) -> CategoryCard:
    return CategoryCard(
        card_id=card_id,
        category="旅行收纳",
        card_type=card_type,
        summary=summary,
        raw_evidence=evidence or [],
        last_updated="2026-07-29",
        confidence=confidence,
    )


def test_hybrid_body_contains_knn_and_bm25() -> None:
    body = _build_hybrid_body(
        "旅行收纳",
        [0.1, 0.2],
    )
    queries = body["query"]["hybrid"][
        "queries"
    ]

    assert "knn" in queries[0]
    assert queries[0]["knn"]["content_vector"][
        "filter"
    ] == {"term": {"category.keyword": "旅行收纳"}}
    multi_match = queries[1]["bool"]["must"][0][
        "multi_match"
    ]
    assert multi_match["analyzer"] == (
        "standard"
    )
    assert body["_source"] == {
        "excludes": ["content_vector"]
    }


def test_cycling_bundle_alias_is_normalized() -> None:
    assert normalize_category("骑行三件套") == "骑行套装"
    assert normalize_category("骑行服三件套") == "骑行套装"
    assert find_category_alias(
        "我想购买骑行三件套，预算 800 元"
    ) == "骑行套装"


def test_effective_insight_requires_structure_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CATEGORY_INSIGHT_MIN_CONFIDENCE", "0.45")
    result = CategoryInsightOutput(
        category="骑行套装",
        components=["骑行上衣", "骑行裤"],
        bestsellers=[
            Bestseller(
                name="示例套装",
                typical_price_cny=299,
                why_popular="透气",
            )
        ],
        attributes=[],
        price_tiers=[],
        confidence=0.86,
    )

    assert is_effective_category_insight(result)
    assert not is_effective_category_insight(
        result.model_copy(update={"components": []})
    )
    assert not is_effective_category_insight(
        result.model_copy(update={"confidence": 0.2})
    )


@pytest.mark.asyncio
async def test_category_insight_extracts_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_recall(
        category: str,
        top_k: int,
    ) -> CategoryRecallResult:
        assert category == "旅行收纳"
        assert top_k == 15
        return CategoryRecallResult(cards=[
            _card(
                "b1",
                "bestseller",
                "洗漱包、鞋包、数码收纳",
                evidence=[
                    "多功能收纳包|89.8|容量大"
                ],
            ),
            _card(
                "a1",
                "attribute",
                "材质：尼龙 60% / 帆布 25% / 牛津布 15%",
            ),
            _card(
                "p1",
                "price_range",
                "便宜款 60-150 元",
            ),
        ])

    async def no_event(
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        return True

    monkeypatch.setattr(
        module,
        "_recall_cards",
        fake_recall,
    )
    monkeypatch.setattr(
        module.monitor,
        "report_tool_start",
        no_event,
    )
    monkeypatch.setattr(
        module.monitor,
        "report_tool_end",
        no_event,
    )

    result = await category_insight.ainvoke(
        {
            "category": "旅行收纳",
            "depth": "deep",
        }
    )

    assert result.category == "旅行收纳"
    assert result.bestsellers[0].name == (
        "多功能收纳包"
    )
    assert result.attributes[0].distribution[
        "尼龙"
    ] == 0.6
    assert result.price_tiers[0].range_cny == (
        60.0,
        150.0,
    )
    assert result.confidence == 0.8
    assert result.knowledge_base_available is True
    assert result.degraded_reason is None
