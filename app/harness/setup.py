from app.harness.middleware import harness

# Import modules for decorator side effects.
from app.harness.hooks import (  # noqa: F401
    content_filter,
    assertion_handler,
    drift_detector,
    loop_detector,
    phase_check,
    phase_transition,
    sequencing,
    step_validator,
    tool_whitelist,
    truncate,
    user_tier_check,
)


def setup_harness():
    return harness


__all__ = ["setup_harness"]
