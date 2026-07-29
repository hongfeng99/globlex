from app.agent.dispatch_tool import (
    dispatch_tool,
)
from app.agent.registry import FULL_TOOL_SET
from app.tools.item_search import item_search
from app.tools.price_compare import (
    price_compare,
)
from app.tools.shipping_calc import (
    shipping_calc,
)


def test_full_tool_set_contains_chapter_tools() -> None:
    assert item_search in FULL_TOOL_SET
    assert price_compare in FULL_TOOL_SET
    assert shipping_calc in FULL_TOOL_SET
    assert dispatch_tool in FULL_TOOL_SET
