from __future__ import annotations

import hashlib
import re
from typing import Any


def sanitize_for_log(
    data: dict[str, Any],
) -> dict[str, Any]:
    result = dict(data)
    query = result.get("query")
    if isinstance(query, str) and len(query) > 15:
        result["query"] = (
            query[:10] + "*****" + query[-5:]
        )
    user_id = result.get("user_id")
    if isinstance(user_id, str):
        result["user_id"] = hashlib.md5(
            user_id.encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]
    for key in list(result):
        if re.search(
            r"(api[_-]?key|secret|password)",
            key,
            re.IGNORECASE,
        ):
            result[key] = "********"
    preferences = result.get("preferences")
    if isinstance(preferences, list):
        result["preferences"] = [
            re.sub(r"\d", "*", str(item))
            for item in preferences
        ]
    return result


__all__ = ["sanitize_for_log"]
