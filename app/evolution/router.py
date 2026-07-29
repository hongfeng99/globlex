from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaseSeverity(str, Enum):
    P0 = "p0"
    P1 = "p1"
    P2 = "p2"


@dataclass
class RoutedCase:
    severity: CaseSeverity
    trace_id: str
    query: str
    trajectory: list[dict]
    rubric_detail: dict
    repair_suggestion: str | None = None


def route_bad_case(
    rubric_detail: dict,
) -> CaseSeverity:
    if rubric_detail.get("p0_pass") is False:
        return CaseSeverity.P0
    if (
        rubric_detail.get("p1_score", 5)
        < 3
    ):
        return CaseSeverity.P1
    return CaseSeverity.P2


__all__ = [
    "CaseSeverity",
    "RoutedCase",
    "route_bad_case",
]
