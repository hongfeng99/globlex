from __future__ import annotations

from app.memory.store import (
    PreferenceStore,
    preference_store,
)
from app.memory.strategy import StrategyEntry


async def inject_preferences(
    system_prompt: str,
    user_id: str | None,
    query: str = "",
    *,
    store: PreferenceStore = preference_store,
) -> str:
    if not user_id:
        return system_prompt
    rendered = await store.render(user_id, query)
    if not rendered:
        rendered = "（暂无长期偏好）"
    return (
        f"{system_prompt}\n\n"
        "## 当前用户长期偏好\n"
        f"{rendered}"
    )


def inject_strategies(
    prompt: str,
    strategies: list[StrategyEntry],
) -> str:
    if not strategies:
        return prompt
    lines = "\n".join(
        f"- {item.query_pattern}: {item.summary} "
        f"(推荐顺序：{' → '.join(item.tool_hints)})"
        for item in strategies[:3]
    )
    return (
        f"{prompt}\n\n"
        "## 成功策略（来自历史高分轨迹，可选择性参考）\n"
        f"{lines}\n"
        "注意：以上策略仅供参考，请根据当前请求调整。"
    )


__all__ = [
    "inject_preferences",
    "inject_strategies",
]
