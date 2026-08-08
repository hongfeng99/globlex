from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.api.monitor import monitor
from app.agent.request_context import (
    get_original_request,
    record_search_candidates,
)
from app.config import env_float
from app.memory.context import get_current_user_id
from app.memory.store import preference_store
from app.recall.ann import ann_client
from app.recall.search_constraints import (
    candidate_rejection_reasons,
    estimate_landed_cny,
    parse_search_constraints,
)
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
    estimated_landed_cny: float | None = None


class ItemSearchOutput(BaseModel):
    """
    ItemSearch 对后续工具暴露的固定输出结构。
    """

    platform: str
    candidates: list[Candidate]
    total_recall: int
    truncated: bool
    matched_total: int | None = None
    rejected_count: int = 0
    applied_constraints: list[str] = Field(
        default_factory=list
    )
    no_match_reason: str | None = None


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
    preferences: list[str],
) -> list[dict[str, Any]]:
    user_embedding, query_embedding = (
        await asyncio.gather(
            tower_client.encode_user(
                user_id,
                preferences,
            ),
            tower_client.encode_query(query),
        )
    )

    fused_embedding = _fuse_embeddings(
        query_embedding,
        user_embedding,
    )

    return ann_client.search(
        fused_embedding,
        top_k,
        platform,
    )


def _fuse_embeddings(
    query_embedding: list[float],
    user_embedding: list[float],
    *,
    query_weight: float | None = None,
    user_weight: float | None = None,
) -> list[float]:
    """Weighted fusion that keeps the item-index dimension unchanged."""

    if len(user_embedding) != len(query_embedding):
        raise ValueError(
            "User 塔与 Query 塔向量维度不一致。"
        )
    effective_query_weight = (
        query_weight
        if query_weight is not None
        else env_float(
            "TOWER_QUERY_WEIGHT", 0.75, minimum=0
        )
    )
    effective_user_weight = (
        user_weight
        if user_weight is not None
        else env_float(
            "TOWER_USER_WEIGHT", 0.25, minimum=0
        )
    )
    if effective_query_weight + effective_user_weight <= 0:
        raise ValueError("Query 与 User 融合权重不能同时为 0。")

    fused = [
        effective_query_weight * query_value
        + effective_user_weight * user_value
        for query_value, user_value in zip(
            query_embedding,
            user_embedding,
        )
    ]
    norm = math.sqrt(sum(value * value for value in fused))
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError("融合后的向量无法归一化。")
    return [value / norm for value in fused]


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

    preferences: list[str] = []
    if user_id:
        entries = await preference_store.read_relevant(
            user_id,
            query,
        )
        preferences = [
            entry.preference for entry in entries
        ]

    personalized_task = (
        asyncio.create_task(
            _personalized_recall(
                query,
                platform,
                recall_k,
                user_id,
                preferences,
            )
        )
        if user_id and preferences
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
    constraints = parse_search_constraints(
        get_original_request(),
        normalized_query,
    )
    context_user_id = get_current_user_id()
    normalized_user_id = (
        context_user_id
        if context_user_id is not None
        else (
            user_id.strip()
            if user_id is not None
            else None
        )
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

    # Hard constraints need a wider semantic pool. Otherwise a top_k=5
    # request could filter five near-matches and miss a valid sixth item.
    recall_limit = 50 if constraints.active else top_k
    raw_candidates, total_recall = (
        await _recall(
            normalized_query,
            platform,
            recall_limit,
            normalized_user_id,
        )
    )

    recalled_count = len(raw_candidates)
    filtered_candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if candidate_rejection_reasons(raw, constraints):
            continue
        candidate = dict(raw)
        candidate["estimated_landed_cny"] = (
            estimate_landed_cny(raw)
        )
        filtered_candidates.append(candidate)

    matched_total = len(filtered_candidates)
    raw_candidates = filtered_candidates[:top_k]
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
            estimated_landed_cny=raw.get(
                "estimated_landed_cny"
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

    result = ItemSearchOutput(
        platform=platform,
        candidates=candidates,
        total_recall=total_recall,
        truncated=(
            total_recall > recalled_count
            or matched_total > len(candidates)
        ),
        matched_total=matched_total,
        rejected_count=recalled_count - matched_total,
        applied_constraints=constraints.labels(),
        no_match_reason=(
            None
            if candidates
            else (
                (
                    "未找到同时满足以下硬条件的离线模拟商品："
                    + "、".join(constraints.labels())
                    if constraints.labels()
                    else "当前离线模拟库未找到候选商品"
                )
                + "。系统没有自动放宽条件。"
            )
        ),
    )
    record_search_candidates(result.candidates)
    return result


__all__ = [
    "Candidate",
    "ItemSearchOutput",
    "Platform",
    "_fuse_embeddings",
    "item_search",
]
