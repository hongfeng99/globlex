from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CardType = Literal[
    "bestseller",
    "attribute",
    "price_range",
]


class CategoryCard(BaseModel):
    """
    RAG 商品知识库中的最小结构化卡片。
    """

    card_id: str
    category: str
    card_type: CardType
    summary: str
    raw_evidence: list[str] = Field(
        default_factory=list
    )
    last_updated: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


__all__ = [
    "CardType",
    "CategoryCard",
]
