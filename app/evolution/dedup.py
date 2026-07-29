from __future__ import annotations

import hashlib
from collections import defaultdict


_daily_counts: dict[str, int] = defaultdict(int)
MAX_PER_PATTERN_PER_DAY = 3


def should_keep(query: str) -> bool:
    pattern = " ".join(query.lower().split())
    key = hashlib.md5(
        pattern[:200].encode(),
        usedforsecurity=False,
    ).hexdigest()[:8]
    _daily_counts[key] += 1
    return (
        _daily_counts[key]
        <= MAX_PER_PATTERN_PER_DAY
    )
