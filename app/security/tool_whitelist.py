from app.agent.tool_registry import FULL_TOOL_SET


ALLOWED_TOOLS = {
    tool.name for tool in FULL_TOOL_SET
}


def validate_tool_call(tool_name: str) -> bool:
    return tool_name in ALLOWED_TOOLS


__all__ = [
    "ALLOWED_TOOLS",
    "validate_tool_call",
]
