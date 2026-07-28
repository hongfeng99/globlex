from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.api.monitor import monitor
from app.recall.ann import ann_client
from app.recall.towers import tower_client


Platform = Literal[
    "amazon",
    "shopee",
    "aliexpress",
    "ebay",
]


class Candidate(BaseModel):
    """
    单个候选商品的稳定结构。
    """

    item_id: str
    platform: str
    title: str
    price: float
    currency: str
    rating: float | None = None
    sales: int | None = None
    image_url: str | None = None
    attributes: dict[str, Any] = Field(
        default_factory=dict
    )


class ItemSearchOutput(BaseModel):
    """
    ItemSearch 对后续工具暴露的固定输出结构。
    """

    platform: str
    candidates: list[Candidate]
    total_recall: int
    truncated: bool


async def _semantic_recall(
    query: str,
    platform: str,
    top_k: int,
) -> list[dict[str, Any]]:
    embedding = await tower_client.encode_query(
        query
    )

    return ann_client.search(
        embedding,
        top_k,
        platform,
    )


async def _personalized_recall(
    query: str,
    platform: str,
    top_k: int,
    user_id: str,
) -> list[dict[str, Any]]:
    user_embedding, query_embedding = (
        await asyncio.gather(
            tower_client.encode_user(user_id),
            tower_client.encode_query(query),
        )
    )

    if len(user_embedding) != len(
        query_embedding
    ):
        raise ValueError(
            "User 塔与 Query 塔向量维度不一致。"
        )

    # 当前章节使用固定权重；后续可替换成训练得到的融合层。
    fused_embedding = [
        0.6 * user_value
        + 0.4 * query_value
        for user_value, query_value in zip(
            user_embedding,
            query_embedding,
        )
    ]

    return ann_client.search(
        fused_embedding,
        top_k,
        platform,
    )


def _dedupe_and_rerank(
    semantic: list[dict[str, Any]],
    personalized: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    合并两路召回、按 item_id 去重，再按融合分数排序。
    """

    candidates: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in semantic:
        item_id = str(item["item_id"])
        score = float(item["score"])
        candidates[item_id] = {
            **item,
            "boost": score,
        }

    for item in personalized:
        item_id = str(item["item_id"])
        score = float(item["score"])
        existing = candidates.get(item_id)

        if existing is not None:
            existing["boost"] = (
                float(existing["boost"])
                + 0.5 * score
            )
        else:
            candidates[item_id] = {
                **item,
                "boost": 0.8 * score,
            }

    return sorted(
        candidates.values(),
        key=lambda item: float(
            item["boost"]
        ),
        reverse=True,
    )


async def _recall(
    query: str,
    platform: str,
    top_k: int,
    user_id: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """
    并发执行语义召回和可选的个性化召回。

    内部额外召回一条，用于判断是否因为 top_k
    而截断了候选集。
    """

    recall_k = top_k + 1

    semantic_task = asyncio.create_task(
        _semantic_recall(
            query,
            platform,
            recall_k,
        )
    )

    personalized_task = (
        asyncio.create_task(
            _personalized_recall(
                query,
                platform,
                recall_k,
                user_id,
            )
        )
        if user_id
        else None
    )

    try:
        semantic = await semantic_task

        personalized = (
            await personalized_task
            if personalized_task is not None
            else []
        )
    except BaseException:
        if not semantic_task.done():
            semantic_task.cancel()

        if (
            personalized_task is not None
            and not personalized_task.done()
        ):
            personalized_task.cancel()

        await asyncio.gather(
            *(
                task
                for task in (
                    semantic_task,
                    personalized_task,
                )
                if task is not None
            ),
            return_exceptions=True,
        )

        raise

    merged = _dedupe_and_rerank(
        semantic,
        personalized,
    )

    total_recall = len(merged)

    return (
        merged[:top_k],
        total_recall,
    )


@tool
async def item_search(
    query: str,
    platform: Platform,
    top_k: int = 20,
    user_id: str | None = None,
) -> ItemSearchOutput:
    """
    在指定平台检索商品候选集。

    Args:
        query: Planner 拆解后的具体检索词。
        platform: amazon、shopee、aliexpress 或 ebay。
        top_k: 返回候选数量，默认 20，最大 50。
        user_id: 可选；传入后启用个性化召回通道。

    Returns:
        platform / candidates / total_recall / truncated
        四字段固定结构。
    """

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "query 不能为空字符串。"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    top_k = min(top_k, 50)
    normalized_user_id = (
        user_id.strip()
        if user_id is not None
        else None
    )
    normalized_user_id = (
        normalized_user_id or None
    )

    await monitor.report_tool_start(
        "item_search",
        {
            "query": normalized_query,
            "platform": platform,
            "top_k": top_k,
        },
    )

    started_at = time.perf_counter()

    raw_candidates, total_recall = (
        await _recall(
            normalized_query,
            platform,
            top_k,
            normalized_user_id,
        )
    )

    candidates = [
        Candidate(
            item_id=str(raw["item_id"]),
            platform=platform,
            title=str(raw["title"]),
            price=float(raw["price"]),
            currency=str(raw["currency"]),
            rating=raw.get("rating"),
            sales=raw.get("sales"),
            image_url=raw.get("image_url"),
            attributes=raw.get(
                "attributes",
                {},
            ),
        )
        for raw in raw_candidates
    ]

    duration_ms = int(
        (
            time.perf_counter()
            - started_at
        )
        * 1000
    )

    await monitor.report_tool_end(
        "item_search",
        duration_ms,
    )

    return ItemSearchOutput(
        platform=platform,
        candidates=candidates,
        total_recall=total_recall,
        truncated=(
            total_recall > len(candidates)
        ),
    )


__all__ = [
    "Candidate",
    "ItemSearchOutput",
    "Platform",
    "item_search",
]
