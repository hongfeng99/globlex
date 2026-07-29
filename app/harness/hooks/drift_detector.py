from __future__ import annotations

from collections import deque

from app.harness.middleware import harness_hook


CHECK_INTERVAL = 3
_round_counter = 0
_recent_actions: deque[str] = deque(maxlen=3)


@harness_hook(
    "post_reflect",
    name="drift_detector",
    priority=20,
)
async def detect_drift(
    context: dict,
) -> dict | None:
    global _round_counter
    _round_counter += 1
    action = context.get(
        "last_tool_name", ""
    )
    if action:
        _recent_actions.append(action)
    if _round_counter % CHECK_INTERVAL:
        return None

    original_query = context.get(
        "original_query", ""
    )
    if not original_query:
        return None

    severe = (
        len(_recent_actions) == 3
        and len(set(_recent_actions)) == 1
        and context.get("progress_count", 0)
        == 0
    )
    if severe:
        count = (
            context.get(
                "consecutive_severe_drift", 0
            )
            + 1
        )
        context[
            "consecutive_severe_drift"
        ] = count
        context.setdefault(
            "inject_messages", []
        ).append(
            {
                "role": "system",
                "content": (
                    "漂移校正：回到用户原始需求，"
                    "停止重复无进展动作，并选择能直接"
                    "产生候选或收尾的工具。"
                ),
            }
        )
    return context
