from __future__ import annotations


CATEGORY_ALIASES: dict[str, str] = {
    "旅行收纳": "旅行收纳",
    "便携收纳包": "旅行收纳",
    "出差三件套": "旅行收纳",
    "骑行三件套": "骑行套装",
    "骑行服三件套": "骑行套装",
    "骑行套装": "骑行套装",
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


def find_category_alias(text: str) -> str | None:
    """Find the most specific known category phrase in free-form text."""

    normalized = text.strip().lower()
    matches = [
        (alias, category)
        for alias, category in CATEGORY_ALIASES.items()
        if alias in normalized
    ]
    if not matches:
        return None
    _, category = max(
        matches,
        key=lambda pair: len(pair[0]),
    )
    return category


__all__ = [
    "CATEGORY_ALIASES",
    "find_category_alias",
    "normalize_category",
]
