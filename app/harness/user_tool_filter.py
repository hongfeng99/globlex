from app.harness.phase_machine import (
    phase_machine,
)


USER_TIER_RESTRICTIONS = {
    "free": {"dispatch_tool"},
    "standard": set(),
    "premium": set(),
}


def get_user_filtered_tools(
    user_tier: str = "free",
) -> set[str]:
    return (
        phase_machine.get_allowed_tools()
        - USER_TIER_RESTRICTIONS.get(
            user_tier, set()
        )
    )
