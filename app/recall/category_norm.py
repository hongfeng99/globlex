from __future__ import annotations


CATEGORY_ALIASES: dict[str, str] = {
    "旅行收纳": "旅行三件套",
    "便携收纳包": "旅行三件套",
    "出差三件套": "旅行三件套",
    "咖啡杯": "咖啡杯",
    "马克杯": "咖啡杯",
}


def normalize_category(raw: str) -> str:
    """
    对业务侧常见品类别名做稳定归一。
    """

    normalized = raw.strip().lower()

    if not normalized:
        raise ValueError(
            "category 不能为空字符串。"
        )

    return CATEGORY_ALIASES.get(
        normalized,
        normalized,
    )


__all__ = [
    "CATEGORY_ALIASES",
    "normalize_category",
]
