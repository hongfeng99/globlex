from __future__ import annotations


USER_TIER_LIMITS = {
    "free": 30_000,
    "standard": 50_000,
    "premium": 100_000,
}


def get_user_token_limit(
    user_id: str | None,
    tier: str | None = None,
) -> int:
    return USER_TIER_LIMITS.get(
        tier or "free",
        USER_TIER_LIMITS["free"],
    )


__all__ = [
    "USER_TIER_LIMITS",
    "get_user_token_limit",
]
