from __future__ import annotations

from contextvars import ContextVar
from enum import Enum


class Phase(str, Enum):
    PLANNING = "planning"
    SEARCHING = "searching"
    COMPARING = "comparing"
    CONCLUDING = "concluding"


PHASE_TOOLS = {
    Phase.PLANNING: {
        "planner",
        "chat_fallback",
        "category_insight",
        "web_search",
    },
    Phase.SEARCHING: {
        "item_search",
        "dispatch_tool",
        "web_search",
        "category_insight",
        "chat_fallback",
    },
    Phase.COMPARING: {
        "price_compare",
        "shipping_calc",
        "item_picker",
        "chat_fallback",
    },
    Phase.CONCLUDING: {
        "shopping_summary",
        "chat_fallback",
    },
}


class PhaseStateMachine:
    def __init__(self) -> None:
        self._phase: ContextVar[Phase] = (
            ContextVar(
                "globex_phase",
                default=Phase.PLANNING,
            )
        )

    def get_current_phase(self) -> Phase:
        return self._phase.get()

    def transition(
        self, phase: Phase
    ) -> None:
        self._phase.set(phase)

    def is_allowed(
        self, tool_name: str
    ) -> bool:
        return (
            tool_name
            in PHASE_TOOLS[
                self.get_current_phase()
            ]
        )

    def get_allowed_tools(self) -> set[str]:
        return set(
            PHASE_TOOLS[
                self.get_current_phase()
            ]
        )


phase_machine = PhaseStateMachine()


__all__ = [
    "PHASE_TOOLS",
    "Phase",
    "PhaseStateMachine",
    "phase_machine",
]
