from __future__ import annotations

import hashlib
import time


_recent_requests: dict[str, float] = {}
DEDUP_WINDOW = 5.0


def is_duplicate(
    user_id: str | None,
    query: str,
) -> bool:
    fingerprint = hashlib.md5(
        f"{user_id}:{query}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    now = time.time()
    for key, timestamp in list(
        _recent_requests.items()
    ):
        if now - timestamp > DEDUP_WINDOW:
            del _recent_requests[key]
    if fingerprint in _recent_requests:
        return True
    _recent_requests[fingerprint] = now
    return False
