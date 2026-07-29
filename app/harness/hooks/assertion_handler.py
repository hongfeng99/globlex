from app.harness.middleware import harness_hook


@harness_hook(
    "post_reflect",
    name="assertion_handler",
    priority=15,
)
async def handle_failed_assertions(
    context: dict,
) -> dict | None:
    failures = context.get(
        "assertions_failed", []
    )
    if not failures:
        return None
    messages = []
    for failure in failures:
        messages.append(
            f"{failure.get('type')} 校验失败："
            f"{failure.get('reason')}"
        )
    context.setdefault(
        "inject_messages", []
    ).append(
        {
            "role": "system",
            "content": "；".join(messages),
        }
    )
    context["assertions_failed"] = []
    return context
