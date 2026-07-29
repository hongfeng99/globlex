from app.harness.middleware import harness_hook
from app.security.content_filter import (
    sanitize_tool_output,
)


@harness_hook(
    "post_tool_call",
    name="content_filter",
    priority=10,
)
async def filter_tool_output(
    context: dict,
) -> dict | None:
    result = context.get("tool_result")
    if not isinstance(result, str):
        return None
    return {
        "tool_result": sanitize_tool_output(
            result
        )
    }
