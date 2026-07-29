from __future__ import annotations

import re


SENSITIVE_PATTERNS = [
    r"item_id\s*[:=]\s*[\w-]+",
    r"thread_id\s*[:=]\s*[\w-]+",
    r"sk-[A-Za-z0-9_-]{12,}",
    r"https?://(?:vllm|reranker|opensearch)[^\s]*",
    r"(?:dispatch_tool|task_tool)",
]
_compiled = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in SENSITIVE_PATTERNS
]


def audit_output(
    text: str,
) -> tuple[bool, str]:
    violations: list[str] = []
    sanitized = text
    for pattern in _compiled:
        matches = pattern.findall(sanitized)
        if matches:
            violations.extend(
                str(match) for match in matches
            )
            sanitized = pattern.sub(
                "[已脱敏]", sanitized
            )
    return len(violations) == 0, sanitized


__all__ = [
    "SENSITIVE_PATTERNS",
    "audit_output",
]
