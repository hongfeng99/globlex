from app.evolution.fork_optimizer import (
    platform_tracker,
)


DEFAULT_PLATFORMS = [
    "amazon",
    "shopee",
    "aliexpress",
    "ebay",
]
MIN_SUCCESS_RATE = 0.3


def get_fork_candidates() -> list[str]:
    candidates = [
        platform
        for platform in DEFAULT_PLATFORMS
        if platform_tracker.get_success_rate(
            platform
        )
        >= MIN_SUCCESS_RATE
    ]
    if len(candidates) < 2:
        ranked = (
            platform_tracker
            .get_ranked_platforms()
        )
        candidates = [
            platform
            for platform, _ in ranked[:2]
        ] or DEFAULT_PLATFORMS[:2]
    return candidates
