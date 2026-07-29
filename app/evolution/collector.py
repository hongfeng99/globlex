from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


COLLECTION_THRESHOLD = 0.65
bad_case_pool: list[dict[str, Any]] = []


async def should_collect(
    trace_id: str,
    rubric_score: float,
) -> bool:
    return rubric_score < COLLECTION_THRESHOLD


async def collect_bad_case(
    *,
    trace_id: str,
    query: str,
    trajectory: list[dict],
    rubric_score: float,
    rubric_comment: str,
    tool_calls: list[dict],
    token_consumed: int,
) -> dict:
    case = {
        "trace_id": trace_id,
        "query": query,
        "trajectory": trajectory,
        "rubric_score": rubric_score,
        "rubric_comment": rubric_comment,
        "tool_calls": tool_calls,
        "token_consumed": token_consumed,
        "timestamp": datetime.now(
            UTC
        ).isoformat(),
    }
    bad_case_pool.append(case)
    return case


__all__ = [
    "COLLECTION_THRESHOLD",
    "bad_case_pool",
    "collect_bad_case",
    "should_collect",
]
