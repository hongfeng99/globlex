from app.harness.middleware import (
    HookRejectSignal,
    harness_hook,
)
from app.harness.phase_machine import (
    phase_machine,
)


@harness_hook(
    "pre_tool_call",
    name="phase_check",
    priority=20,
)
async def check_phase_permission(
    context: dict,
) -> None:
    tool_name = context.get("tool_name", "")
    if not phase_machine.is_allowed(tool_name):
        raise HookRejectSignal(
            f"工具 {tool_name} 在当前阶段 "
            f"{phase_machine.get_current_phase().value} "
            "不可用"
        )
