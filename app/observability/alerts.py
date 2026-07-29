from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import httpx
import numpy as np


@dataclass(frozen=True)
class AlertRule:
    tool_name: str
    p99_threshold_ms: int
    window_minutes: int = 5
    min_samples: int = 10


ALERT_RULES = [
    AlertRule("item_search", 3000),
    AlertRule("price_compare", 500),
    AlertRule("shipping_calc", 200),
    AlertRule("category_insight", 2000),
    AlertRule("shopping_summary", 4000),
    AlertRule("dispatch_tool", 5000),
]


class ToolRTMonitor:
    def __init__(self) -> None:
        self._windows = {
            rule.tool_name: deque(maxlen=200)
            for rule in ALERT_RULES
        }

    def record(
        self, tool_name: str, duration_ms: int
    ) -> None:
        if tool_name in self._windows:
            self._windows[tool_name].append(
                duration_ms
            )

    def check_alerts(self) -> list[str]:
        alerts: list[str] = []
        for rule in ALERT_RULES:
            values = self._windows[rule.tool_name]
            if len(values) < rule.min_samples:
                continue
            p99 = float(
                np.percentile(list(values), 99)
            )
            if p99 > rule.p99_threshold_ms:
                alerts.append(
                    f"{rule.tool_name} P99 "
                    f"{p99:.0f}ms 超过阈值"
                )
        return alerts


async def send_alert(message: str) -> bool:
    url = os.getenv(
        "ALERT_WEBHOOK_URL", ""
    ).strip()
    if not url:
        return False
    async with httpx.AsyncClient() as client:
        await client.post(
            url,
            json={
                "msgtype": "text",
                "text": {
                    "content": (
                        f"[Globex Agent] {message}"
                    )
                },
            },
        )
    return True


tool_rt_monitor = ToolRTMonitor()


__all__ = [
    "ALERT_RULES",
    "AlertRule",
    "ToolRTMonitor",
    "send_alert",
    "tool_rt_monitor",
]
