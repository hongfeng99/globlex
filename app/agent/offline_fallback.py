from __future__ import annotations

from app.recall.search_constraints import parse_search_constraints


def build_clarification(user_request: str) -> str | None:
    """Return a deterministic clarification for a bare shopping intent."""

    constraints = parse_search_constraints(user_request)
    if constraints.category_key is None:
        return None

    normalized = user_request.casefold()
    has_usage = any(
        marker in normalized
        for marker in ("办公", "游戏", "编程", "打字", "送礼", "通勤")
    )
    has_any_detail = any(
        (
            constraints.max_landed_cny is not None,
            constraints.switch_type is not None,
            constraints.connection is not None,
            constraints.layout is not None,
            has_usage,
        )
    )
    if has_any_detail:
        return None

    if constraints.category_key == "mechanical-keyboard":
        return (
            "为了帮您精准筛选机械键盘，需要了解以下信息：\n\n"
            "1. 预算上限（例如 200 元、500 元或不限）\n"
            "2. 使用场景（办公打字、游戏或兼顾）\n"
            "3. 轴体偏好（青轴、红轴、茶轴、静音红轴或不限）\n"
            "4. 配列大小（75%、87 键、98 键或不限）\n"
            "5. 连接方式（有线、无线、蓝牙、2.4G 或三模）\n"
            "6. 收货国家或地区（未说明时默认中国大陆）\n\n"
            "请直接逐行补充即可；下一轮会自动与本需求合并。"
        )

    return (
        f"为了筛选{constraints.category_name}，请补充预算上限、"
        "主要使用场景和收货国家或地区。"
    )


__all__ = ["build_clarification"]
