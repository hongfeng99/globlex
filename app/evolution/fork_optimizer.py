from collections import defaultdict


class PlatformSuccessTracker:
    def __init__(self) -> None:
        self._stats = defaultdict(
            lambda: {"success": 0, "total": 0}
        )

    def record(
        self, platform: str, success: bool
    ) -> None:
        stat = self._stats[platform]
        stat["total"] += 1
        stat["success"] += int(success)

    def get_success_rate(
        self, platform: str
    ) -> float:
        stat = self._stats[platform]
        if stat["total"] == 0:
            return 0.5
        return stat["success"] / stat["total"]

    def get_ranked_platforms(
        self,
    ) -> list[tuple[str, float]]:
        return sorted(
            (
                (
                    platform,
                    self.get_success_rate(platform),
                )
                for platform in self._stats
            ),
            key=lambda item: item[1],
            reverse=True,
        )


platform_tracker = PlatformSuccessTracker()
