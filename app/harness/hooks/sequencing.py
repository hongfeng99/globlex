from app.harness.middleware import harness_hook


PREREQUISITES = {
    "shopping_summary": ["item_picker"],
    "price_compare": ["item_search"],
    "shipping_calc": ["price_compare"],
    "item_picker": ["shipping_calc"],
}


@harness_hook(
    "pre_tool_call",
    name="sequencing_assertion",
    priority=25,
)
async def check_sequencing(
    context: dict,
) -> dict | None:
    tool_name = context.get("tool_name", "")
    called = context.setdefault(
        "called_tools", []
    )
    missing = [
        item
        for item in PREREQUISITES.get(
            tool_name, []
        )
        if item not in called
    ]
    if missing:
        context.setdefault(
            "assertions_failed", []
        ).append(
            {
                "type": "sequencing",
                "tool": tool_name,
                "reason": (
                    "缺少前置工具："
                    + ", ".join(missing)
                ),
            }
        )
    called.append(tool_name)
    return context
