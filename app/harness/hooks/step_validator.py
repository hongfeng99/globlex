from __future__ import annotations

from pydantic import BaseModel, ValidationError

from app.harness.middleware import harness_hook
from app.tools.category_insight import (
    CategoryInsightOutput,
)
from app.tools.item_picker import ItemPickerOutput
from app.tools.item_search import ItemSearchOutput
from app.tools.price_compare import (
    PriceCompareOutput,
)
from app.tools.shipping_calc import (
    ShippingCalcOutput,
)
from app.tools.shopping_summary import (
    ShoppingSummaryOutput,
)


TOOL_SCHEMAS: dict[
    str, type[BaseModel]
] = {
    "item_search": ItemSearchOutput,
    "price_compare": PriceCompareOutput,
    "shipping_calc": ShippingCalcOutput,
    "category_insight": CategoryInsightOutput,
    "item_picker": ItemPickerOutput,
    "shopping_summary": ShoppingSummaryOutput,
}


@harness_hook(
    "post_tool_call",
    name="schema_assertion",
    priority=40,
)
async def check_schema(
    context: dict,
) -> None:
    schema = TOOL_SCHEMAS.get(
        context.get("tool_name", "")
    )
    if schema is None:
        return
    result = context.get("tool_result")
    try:
        if isinstance(result, schema):
            return
        schema.model_validate(result)
    except ValidationError as exc:
        context.setdefault(
            "assertions_failed", []
        ).append(
            {
                "type": "schema",
                "tool": context.get(
                    "tool_name", ""
                ),
                "reason": str(exc),
            }
        )
