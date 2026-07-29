from app.harness.middleware import (
    HookRejectSignal,
    harness_hook,
)
from app.security.tool_whitelist import (
    validate_tool_call,
)


@harness_hook(
    "pre_tool_call",
    name="tool_whitelist",
    priority=10,
)
async def check_tool_whitelist(
    context: dict,
) -> None:
    tool_name = context.get("tool_name", "")
    if not validate_tool_call(tool_name):
        raise HookRejectSignal(
            f"工具 {tool_name} 不在白名单中"
        )
