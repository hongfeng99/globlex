from __future__ import annotations


# 按平台和最低重量分档的简化运费表：
# (min_weight_kg, fee_cny, eta_days)
SHIPPING_TABLE: dict[
    str,
    list[tuple[float, float, int]],
] = {
    "amazon": [
        (0.0, 85.0, 12),
        (0.5, 130.0, 10),
        (2.0, 240.0, 8),
    ],
    "shopee": [
        (0.0, 35.0, 9),
        (0.5, 60.0, 9),
        (2.0, 120.0, 7),
    ],
    "aliexpress": [
        (0.0, 20.0, 25),
        (0.5, 40.0, 22),
        (2.0, 90.0, 18),
    ],
    "ebay": [
        (0.0, 90.0, 14),
        (0.5, 150.0, 12),
        (2.0, 300.0, 10),
    ],
}


def estimate_shipping(
    weight_kg: float,
    platform: str,
) -> tuple[float, int]:
    """
    根据重量和平台返回预估运费与物流天数。

    未知平台使用 Amazon 档位；超过最高重量档时使用
    最高档估算。
    """

    if weight_kg < 0:
        raise ValueError(
            "weight_kg 不能小于 0。"
        )

    table = SHIPPING_TABLE.get(
        platform.strip().lower(),
        SHIPPING_TABLE["amazon"],
    )

    fee, eta = table[0][1], table[0][2]

    for minimum_weight, tier_fee, days in (
        table
    ):
        if weight_kg >= minimum_weight:
            fee, eta = tier_fee, days

    return fee, eta


__all__ = [
    "SHIPPING_TABLE",
    "estimate_shipping",
]
