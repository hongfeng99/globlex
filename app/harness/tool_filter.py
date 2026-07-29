from app.agent.tool_registry import FULL_TOOL_SET
from app.harness.phase_machine import (
    phase_machine,
)


def get_filtered_tool_set() -> list:
    allowed = phase_machine.get_allowed_tools()
    return [
        tool
        for tool in FULL_TOOL_SET
        if tool.name in allowed
    ]


__all__ = ["get_filtered_tool_set"]
