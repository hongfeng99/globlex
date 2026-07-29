from __future__ import annotations

from typing import Literal


DutyTier = Literal[
    "免征",
    "标准",
    "高税",
]


# 极简通用税率表。生产环境应根据 HS Code、原产地、
# 目的地及实时政策计算。
DUTY_TABLE: dict[
    str,
    tuple[float, DutyTier],
] = {
    "amazon": (0.13, "标准"),
    "shopee": (0.06, "免征"),
    "aliexpress": (0.13, "标准"),
    "ebay": (0.20, "高税"),
}


def estimate_duty(
    price_cny: float,
    platform: str,
) -> tuple[float, DutyTier]:
    """
    根据平台估算关税；未知平台按标准 13% 兜底。
    """

    if price_cny < 0:
        raise ValueError(
            "price_cny 不能小于 0。"
        )

    rate, tier = DUTY_TABLE.get(
        platform.strip().lower(),
        (0.13, "标准"),
    )

    return round(
        price_cny * rate,
        2,
    ), tier


__all__ = [
    "DUTY_TABLE",
    "DutyTier",
    "estimate_duty",
]
