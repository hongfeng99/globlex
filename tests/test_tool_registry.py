from app.agent.dispatch_tool import (
    dispatch_tool,
)
from app.agent.registry import FULL_TOOL_SET
from app.tools.item_search import item_search


def test_full_tool_set_contains_chapter_tools() -> None:
    assert item_search in FULL_TOOL_SET
    assert dispatch_tool in FULL_TOOL_SET
