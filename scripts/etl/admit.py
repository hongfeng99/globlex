from __future__ import annotations

import random

from pydantic import ValidationError

from app.recall.category_kb import CategoryCard


MIN_CONFIDENCE = 0.5
MAX_SUMMARY_LEN = 200
SAMPLE_AUDIT_RATIO = 0.1


def admit(raw: dict) -> tuple[bool, str]:
    try:
        card = CategoryCard(**raw)
    except ValidationError as exc:
        return False, f"schema 校验失败：{exc}"
    if card.confidence < MIN_CONFIDENCE:
        return False, "confidence 低于门槛"
    if len(card.summary) > MAX_SUMMARY_LEN:
        return False, "summary 过长"
    if (
        card.card_type == "bestseller"
        and "|" not in card.summary
    ):
        return False, "bestseller 缺少分隔字段"
    if random.random() < SAMPLE_AUDIT_RATIO:
        return True, "通过，进入人工抽审"
    return True, "通过"
