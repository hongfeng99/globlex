from collections import deque

from app.harness.middleware import harness_hook


_recent_tools: deque[str] = deque(maxlen=6)
REPEAT_THRESHOLD = 4


@harness_hook(
    "post_reflect",
    name="loop_detector",
    priority=10,
)
async def detect_loop(
    context: dict,
) -> dict | None:
    tool_name = context.get("last_tool_name")
    if not tool_name:
        return None
    _recent_tools.append(tool_name)
    if (
        _recent_tools.count(tool_name)
        < REPEAT_THRESHOLD
    ):
        return None
    context.setdefault(
        "inject_messages", []
    ).append(
        {
            "role": "system",
            "content": (
                f"你已重复调用 {tool_name} "
                f"{REPEAT_THRESHOLD} 次，请基于已有"
                "结果换路径或尽快收尾。"
            ),
        }
    )
    return context
