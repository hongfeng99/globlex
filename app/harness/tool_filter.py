from app.harness.phase_machine import (
    PHASE_TOOLS,
    Phase,
    phase_machine,
)


def _tool_name(tool: object) -> str | None:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(tool, dict):
        direct_name = tool.get("name")
        if isinstance(direct_name, str):
            return direct_name
        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str):
                return function_name
    return None


def filter_tools_for_phase(
    tools: list,
    phase: Phase | None = None,
) -> list:
    allowed = (
        phase_machine.get_allowed_tools()
        if phase is None
        else PHASE_TOOLS[phase]
    )
    return [
        tool for tool in tools if _tool_name(tool) in allowed
    ]


def get_filtered_tool_set() -> list:
    # 延迟导入，避免 tool_registry -> dispatch_tool -> middleware 循环。
    from app.agent.tool_registry import FULL_TOOL_SET

    return filter_tools_for_phase(FULL_TOOL_SET)


__all__ = [
    "filter_tools_for_phase",
    "get_filtered_tool_set",
]
