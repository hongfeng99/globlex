from __future__ import annotations

import asyncio
import os
import re
import time
import logging
from functools import lru_cache
from typing import Any, Literal, Protocol

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.api.monitor import monitor
from app.recall.category_kb import CategoryCard
from app.recall.category_norm import (
    normalize_category,
)
from app.recall.reranker import reranker_client
from app.recall.towers import tower_client
from app.config import env_bool


INDEX_NAME = "globex_category_kb"
COARSE_K = 30
FINE_K_QUICK = 8
FINE_K_DEEP = 15
RERANK_BYPASS_TOP_SCORE = 0.92
SEMANTIC_TOKENS = {
    "气质",
    "感觉",
    "风格",
    "适合",
    "送",
    "氛围",
}
logger = logging.getLogger(__name__)


def should_disable_bm25(
    category: str,
) -> bool:
    return any(
        token in category
        for token in SEMANTIC_TOKENS
    )


class SearchClient(Protocol):
    def search(
        self,
        *,
        index: str,
        body: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class Bestseller(BaseModel):
    name: str
    typical_price_cny: float
    why_popular: str


class AttributeDist(BaseModel):
    name: str
    distribution: dict[str, float] = Field(
        default_factory=dict
    )


class PriceTier(BaseModel):
    tier: Literal[
        "budget",
        "mid",
        "premium",
    ]
    range_cny: tuple[float, float]
    notes: str


class CategoryInsightOutput(BaseModel):
    category: str
    components: list[str]
    bestsellers: list[Bestseller]
    attributes: list[AttributeDist]
    price_tiers: list[PriceTier]
    confidence: float


@lru_cache(maxsize=1)
def _get_kb_client() -> SearchClient:
    """
    懒加载 OpenSearch，避免无知识库环境在导入时失败。
    """

    from opensearchpy import OpenSearch

    host = os.getenv(
        "OPENSEARCH_HOST",
        "localhost",
    )
    port = int(
        os.getenv(
            "OPENSEARCH_PORT",
            "9200",
        )
    )
    user = os.getenv(
        "OPENSEARCH_USER",
        "admin",
    )
    password = os.getenv(
        "OPENSEARCH_PASS",
        "admin",
    )

    return OpenSearch(
        hosts=[
            {
                "host": host,
                "port": port,
            }
        ],
        http_auth=(user, password),
        use_ssl=False,
    )


def _build_hybrid_body(
    category: str,
    embedding: list[float],
    *,
    coarse_k: int = COARSE_K,
) -> dict[str, Any]:
    queries: list[dict[str, Any]] = [
        {
            "knn": {
                "content_vector": {
                    "vector": embedding,
                    "k": coarse_k,
                }
            }
        }
    ]
    if not should_disable_bm25(category):
        queries.append(
            {
                "multi_match": {
                    "query": category,
                    "fields": [
                        "category^2",
                        "summary",
                    ],
                    "analyzer": "ik_max_word",
                }
            }
        )
    return {
        "size": coarse_k,
        "query": {
            "hybrid": {
                "queries": queries
            }
        },
        "_source": {
            "excludes": [
                "content_vector"
            ]
        },
    }


async def _recall_cards(
    category: str,
    top_k: int,
) -> list[CategoryCard]:
    try:
        embedding = await tower_client.encode_query(
            category
        )
        body = _build_hybrid_body(
            category,
            embedding,
        )

        response = await asyncio.to_thread(
            _get_kb_client().search,
            index=INDEX_NAME,
            body=body,
            params={
                "search_pipeline": (
                    "globex_hybrid_pipeline"
                )
            },
        )
    except Exception as exc:
        if env_bool(
            "CATEGORY_KB_REQUIRED",
            False,
        ):
            raise
        logger.warning(
            "OpenSearch 品类知识库不可用，返回空洞察：%s",
            exc,
        )
        return []
    hits = response.get(
        "hits",
        {},
    ).get("hits", [])

    if not hits:
        return []

    coarse_cards = [
        CategoryCard.model_validate(
            hit["_source"]
        )
        for hit in hits
    ]

    top_score = float(
        hits[0].get("_score", 0.0)
    )

    if (
        top_score >= RERANK_BYPASS_TOP_SCORE
        or len(coarse_cards) <= top_k
    ):
        return coarse_cards[:top_k]

    try:
        scores = await reranker_client.score(
            category,
            [
                card.summary
                for card in coarse_cards
            ],
        )
    except Exception:
        return coarse_cards[:top_k]

    ranked = [
        card
        for _, card in sorted(
            zip(scores, coarse_cards),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ]
    return ranked[:top_k]


def _split_by_type(
    cards: list[CategoryCard],
) -> dict[str, list[CategoryCard]]:
    grouped: dict[
        str,
        list[CategoryCard],
    ] = {
        "bestseller": [],
        "attribute": [],
        "price_range": [],
    }

    for card in cards:
        grouped[card.card_type].append(card)

    return grouped


def _extract_components(
    cards: list[CategoryCard],
) -> list[str]:
    found: set[str] = set()

    for card in cards:
        text = card.summary.replace(
            "，",
            "、",
        )

        for token in re.split(
            r"[、/|]",
            text,
        ):
            token = token.strip()

            if token and len(token) <= 20:
                found.add(token)

    return sorted(found)[:12]


def _extract_bestsellers(
    cards: list[CategoryCard],
) -> list[Bestseller]:
    output: list[Bestseller] = []

    for card in cards:
        for evidence in card.raw_evidence:
            try:
                name, price, reason = [
                    value.strip()
                    for value in evidence.split(
                        "|",
                        maxsplit=2,
                    )
                ]
                output.append(
                    Bestseller(
                        name=name,
                        typical_price_cny=float(
                            price
                        ),
                        why_popular=reason,
                    )
                )
            except (TypeError, ValueError):
                continue

    return output[:5]


def _extract_attributes(
    cards: list[CategoryCard],
) -> list[AttributeDist]:
    output: list[AttributeDist] = []

    for card in cards:
        if "：" not in card.summary:
            continue

        name, distribution_text = (
            card.summary.split(
                "：",
                maxsplit=1,
            )
        )
        distribution: dict[str, float] = {}

        for token in distribution_text.split(
            "/"
        ):
            parts = token.strip().rsplit(
                " ",
                maxsplit=1,
            )

            if (
                len(parts) == 2
                and parts[1].endswith("%")
            ):
                try:
                    distribution[parts[0]] = (
                        float(
                            parts[1].rstrip(
                                "%"
                            )
                        )
                        / 100
                    )
                except ValueError:
                    continue

        if distribution:
            output.append(
                AttributeDist(
                    name=name.strip(),
                    distribution=distribution,
                )
            )

    return output


def _extract_price_tiers(
    cards: list[CategoryCard],
) -> list[PriceTier]:
    tiers: list[PriceTier] = []
    labels = {
        "budget": "便宜款",
        "mid": "中档",
        "premium": "高端",
    }

    for tier, label in labels.items():
        for card in cards:
            if label not in card.summary:
                continue

            match = re.search(
                r"(\d+(?:\.\d+)?)\s*"
                r"[-—]\s*"
                r"(\d+(?:\.\d+)?)",
                card.summary,
            )

            if match:
                tiers.append(
                    PriceTier(
                        tier=tier,
                        range_cny=(
                            float(match.group(1)),
                            float(match.group(2)),
                        ),
                        notes=card.summary,
                    )
                )
                break

    return tiers


@tool
async def category_insight(
    category: str,
    depth: Literal[
        "quick",
        "deep",
    ] = "quick",
) -> CategoryInsightOutput:
    """
    获取品类结构化常识：典型组件、爆款、属性分布和
    价格档位。返回内容不包含 RAG 原文。
    """

    normalized_category = normalize_category(
        category
    )
    await monitor.report_tool_start(
        "category_insight",
        {
            "category": (
                normalized_category
            ),
            "depth": depth,
        },
    )
    started_at = time.perf_counter()

    top_k = (
        FINE_K_QUICK
        if depth == "quick"
        else FINE_K_DEEP
    )
    cards = await _recall_cards(
        normalized_category,
        top_k,
    )
    grouped = _split_by_type(cards)

    confidence = (
        sum(
            card.confidence
            for card in cards
        )
        / len(cards)
        if cards
        else 0.0
    )
    result = CategoryInsightOutput(
        category=normalized_category,
        components=_extract_components(
            grouped["bestseller"]
        ),
        bestsellers=_extract_bestsellers(
            grouped["bestseller"]
        ),
        attributes=(
            _extract_attributes(
                grouped["attribute"]
            )
            if depth == "deep"
            else []
        ),
        price_tiers=_extract_price_tiers(
            grouped["price_range"]
        ),
        confidence=round(
            confidence,
            2,
        ),
    )

    await monitor.report_tool_end(
        "category_insight",
        int(
            (
                time.perf_counter()
                - started_at
            )
            * 1000
        ),
    )
    return result


__all__ = [
    "AttributeDist",
    "Bestseller",
    "CategoryInsightOutput",
    "PriceTier",
    "category_insight",
]
