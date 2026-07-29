import pytest

from app.recall.fx import to_base


def test_to_base_currency() -> None:
    assert to_base(
        39.9,
        "usd",
    ) == pytest.approx(286.482)
    assert to_base(
        286.482,
        "CNY",
        "USD",
    ) == pytest.approx(39.9)


def test_unknown_currency_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="未知币种",
    ):
        to_base(
            10,
            "UNKNOWN",
        )
