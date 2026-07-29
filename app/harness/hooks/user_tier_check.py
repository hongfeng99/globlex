from app.harness.middleware import (
    HookRejectSignal,
    harness_hook,
)
from app.harness.user_tool_filter import (
    get_user_filtered_tools,
)


@harness_hook(
    "pre_tool_call",
    name="user_tier_check",
    priority=22,
)
async def check_user_tier(
    context: dict,
) -> None:
    tool_name = context.get("tool_name", "")
    tier = context.get("user_tier", "free")
    if tool_name not in get_user_filtered_tools(tier):
        raise HookRejectSignal(
            f"工具 {tool_name} 对 {tier} 用户不可用"
        )
