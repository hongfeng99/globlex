from __future__ import annotations

import re


DANGEROUS_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
    r"(?i)system\s*prompt",
    r"(?i)you\s+are\s+now",
    r"(?i)(reveal|show).*(api|secret)\s*key",
    r"(?i)output\s+(all|every).*(user|system)",
]
_compiled = [
    re.compile(pattern)
    for pattern in DANGEROUS_PATTERNS
]


def sanitize_tool_output(text: str) -> str:
    for pattern in _compiled:
        if pattern.search(text):
            text = pattern.sub(
                "[内容已过滤：疑似注入]", text
            )
    return text


__all__ = [
    "DANGEROUS_PATTERNS",
    "sanitize_tool_output",
]
