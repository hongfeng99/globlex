from __future__ import annotations

from typing import Final


# 静态演示汇率：每单位外币对应的人民币金额。
# 生产环境应替换成带缓存的实时汇率服务。
FX_RATES: Final[dict[str, float]] = {
    "CNY": 1.0,
    "USD": 7.18,
    "SGD": 5.32,
    "GBP": 9.05,
    "EUR": 7.78,
    "JPY": 0.046,
}


def to_base(
    amount: float,
    currency: str,
    base: str = "CNY",
) -> float:
    """
    把指定币种金额换算为目标基础币种。
    """

    normalized_currency = (
        currency.strip().upper()
    )
    normalized_base = base.strip().upper()

    if amount < 0:
        raise ValueError(
            "amount 不能小于 0。"
        )

    if (
        normalized_currency not in FX_RATES
        or normalized_base not in FX_RATES
    ):
        raise ValueError(
            "未知币种："
            f"{normalized_currency} "
            f"或 {normalized_base}"
        )

    return (
        amount
        * FX_RATES[normalized_currency]
        / FX_RATES[normalized_base]
    )


__all__ = [
    "FX_RATES",
    "to_base",
]
