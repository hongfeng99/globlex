from app.harness.middleware import harness_hook
from app.harness.phase_machine import (
    Phase,
    phase_machine,
)


@harness_hook(
    "post_reflect",
    name="phase_transition",
    priority=40,
)
async def try_phase_transition(
    context: dict,
) -> dict:
    current = phase_machine.get_current_phase()
    if (
        current == Phase.PLANNING
        and context.get("planner_output_ready")
    ):
        phase_machine.transition(
            Phase.SEARCHING
        )
    elif (
        current == Phase.SEARCHING
        and context.get("total_candidates", 0)
        > 0
    ):
        phase_machine.transition(
            Phase.COMPARING
        )
    elif (
        current == Phase.COMPARING
        and context.get("picks_ready")
    ):
        phase_machine.transition(
            Phase.CONCLUDING
        )
    return context
