from app.agent.middleware import (
    truncate_tool_result,
)
from app.harness.middleware import harness_hook


@harness_hook(
    "post_tool_call",
    name="truncate",
    priority=20,
)
async def truncate_result(
    context: dict,
) -> dict | None:
    result = context.get("tool_result")
    truncated = truncate_tool_result(result)
    if truncated == result:
        return None
    return {
        "tool_result": truncated,
        "truncated": True,
    }
