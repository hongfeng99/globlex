from __future__ import annotations

import re

from app.evolution.router import RoutedCase
from app.security.output_guard import (
    SENSITIVE_PATTERNS,
)


_compiled = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in SENSITIVE_PATTERNS
]
blacklisted_trace_ids: set[str] = set()


async def auto_fix(
    case: RoutedCase,
) -> list[str]:
    comment = case.rubric_detail.get(
        "p0_comment", ""
    )
    added: list[str] = []
    if any(
        word in comment.lower()
        for word in ["泄露", "leak"]
    ):
        for text in _extract_leaked_patterns(
            case.trajectory
        ):
            pattern = re.escape(text)
            if pattern not in SENSITIVE_PATTERNS:
                SENSITIVE_PATTERNS.append(pattern)
                _compiled.append(
                    re.compile(
                        pattern, re.IGNORECASE
                    )
                )
                added.append(pattern)
    if "越权" in comment:
        blacklisted_trace_ids.add(case.trace_id)
    return added


def _extract_leaked_patterns(
    trajectory: list[dict],
) -> list[str]:
    leaked: list[str] = []
    for step in trajectory:
        text = str(step.get("content", ""))
        for token in re.findall(
            r"(?:sk-[\w-]{12,}|"
            r"(?:item|thread)_id\s*[:=]\s*[\w-]+)",
            text,
            re.IGNORECASE,
        ):
            leaked.append(token)
    return leaked
