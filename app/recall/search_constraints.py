from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from app.recall.duty import estimate_duty
from app.recall.fx import to_base
from app.recall.offline_catalog import CATEGORY_SPECS
from app.recall.category_norm import find_category_alias
from app.recall.shipping import estimate_shipping


_BUDGET_PATTERNS = (
    re.compile(
        r"^\s*(?P<amount>\d+(?:\.\d+)?)\s*"
        r"(?:元|人民币|cny)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"(?P<amount>\d+(?:\.\d+)?)\s*"
        r"(?:元|人民币|cny)\s*"
        r"(?:以内|以下|之内|内|封顶|上限)",
        re.IGNORECASE,
    ),
    re.compile(
        r"预算\s*(?:不超过|上限(?:为)?|为|是|[:：])?\s*"
        r"(?P<amount>\d+(?:\.\d+)?)\s*"
        r"(?:元|人民币|cny)?",
        re.IGNORECASE,
    ),
)

_SWITCH_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("静音红轴", ("静音红轴", "silent red switch")),
    ("青轴", ("青轴", "蓝轴", "blue switch")),
    ("茶轴", ("茶轴", "brown switch")),
    ("红轴", ("红轴", "red switch")),
    ("黑轴", ("黑轴", "black switch")),
)


@dataclass(frozen=True, slots=True)
class SearchConstraints:
    category_key: str | None = None
    category_name: str | None = None
    max_landed_cny: float | None = None
    switch_type: str | None = None
    connection: str | None = None
    layout: str | None = None
    quiet_required: bool = False

    @property
    def active(self) -> bool:
        return any(
            value is not None and value is not False
            for value in (
                self.category_key,
                self.max_landed_cny,
                self.switch_type,
                self.connection,
                self.layout,
                self.quiet_required,
            )
        )

    def labels(self) -> list[str]:
        labels: list[str] = []
        if self.category_name:
            labels.append(f"品类={self.category_name}")
        if self.max_landed_cny is not None:
            labels.append(
                f"预估到手价≤¥{self.max_landed_cny:g}"
            )
        if self.switch_type:
            labels.append(f"轴体={self.switch_type}")
        if self.connection:
            labels.append(f"连接={self.connection}")
        if self.layout:
            labels.append(f"配列={self.layout}")
        if self.quiet_required:
            labels.append("静音")
        return labels


def _combined_text(*texts: str) -> str:
    return "\n".join(text.strip() for text in texts if text).casefold()


def _extract_budget(text: str) -> float | None:
    values: list[float] = []
    for pattern in _BUDGET_PATTERNS:
        for match in pattern.finditer(text):
            amount = float(match.group("amount"))
            if amount > 0:
                values.append(amount)
    return min(values) if values else None


def _extract_category(text: str) -> tuple[str | None, str | None]:
    for spec in sorted(
        CATEGORY_SPECS,
        key=lambda item: len(item.name),
        reverse=True,
    ):
        if spec.name.casefold() in text:
            return spec.key, spec.name
        if any(keyword.casefold() in text for keyword in spec.keywords):
            return spec.key, spec.name
    return None, None


def parse_search_constraints(*texts: str) -> SearchConstraints:
    text = _combined_text(*texts)
    normalized_alias = find_category_alias(text)
    category_key, category_name = _extract_category(
        text
        + (f"\n{normalized_alias}" if normalized_alias else "")
    )

    switch_type = next(
        (
            canonical
            for canonical, aliases in _SWITCH_ALIASES
            if any(alias.casefold() in text for alias in aliases)
        ),
        None,
    )

    connection: str | None = None
    if "三模" in text or "tri-mode" in text or "trimode" in text:
        connection = "三模"
    elif "蓝牙" in text or "bluetooth" in text:
        connection = "蓝牙"
    elif re.search(r"(?:2[.]4\s*g|2[.]4g)", text):
        connection = "2.4G"
    elif "无线" in text or "wireless" in text:
        connection = "无线"
    elif "有线" in text or "wired" in text:
        connection = "有线"

    layout = next(
        (
            value
            for value in ("75%", "87键", "98键", "104键", "紧凑型")
            if value.casefold() in text
        ),
        None,
    )
    quiet_required = any(
        marker in text
        for marker in ("静音", "安静", "低噪", "quiet", "silent")
    )

    return SearchConstraints(
        category_key=category_key,
        category_name=category_name,
        max_landed_cny=_extract_budget(text),
        switch_type=switch_type,
        connection=connection,
        layout=layout,
        quiet_required=quiet_required,
    )


def estimate_landed_cny(item: Mapping[str, Any]) -> float:
    price_cny = to_base(
        float(item["price"]),
        str(item["currency"]),
        "CNY",
    )
    attributes = item.get("attributes", {})
    weight = 0.5
    if isinstance(attributes, Mapping):
        raw_weight = attributes.get("weight_kg")
        if isinstance(raw_weight, (int, float)) and not isinstance(
            raw_weight, bool
        ):
            weight = max(0.0, float(raw_weight))
    platform = str(item["platform"])
    shipping_cny, _ = estimate_shipping(weight, platform)
    duty_cny, _ = estimate_duty(price_cny, platform)
    return round(price_cny + shipping_cny + duty_cny, 2)


def candidate_rejection_reasons(
    item: Mapping[str, Any],
    constraints: SearchConstraints,
) -> list[str]:
    reasons: list[str] = []
    attributes = item.get("attributes", {})
    if not isinstance(attributes, Mapping):
        attributes = {}

    if constraints.category_key and str(
        item.get("category_key", "")
    ) != constraints.category_key:
        reasons.append("品类不符")

    if constraints.switch_type and str(
        attributes.get("switch_type", "")
    ) != constraints.switch_type:
        reasons.append("轴体不符")

    modes = {
        str(mode).casefold()
        for mode in attributes.get("connection_modes", [])
    }
    if constraints.connection == "三模" and not {
        "usb-c",
        "2.4g",
        "蓝牙",
    }.issubset(modes):
        reasons.append("连接方式不符")
    elif constraints.connection == "无线" and not (
        {"2.4g", "蓝牙"} & modes
    ):
        reasons.append("连接方式不符")
    elif constraints.connection == "蓝牙" and "蓝牙" not in modes:
        reasons.append("连接方式不符")
    elif constraints.connection == "2.4G" and "2.4g" not in modes:
        reasons.append("连接方式不符")
    elif constraints.connection == "有线" and "usb-c" not in modes:
        reasons.append("连接方式不符")

    if constraints.layout and constraints.layout not in str(
        attributes.get("layout", "")
    ):
        reasons.append("配列不符")

    if constraints.quiet_required and str(
        attributes.get("noise_level", "")
    ) != "安静":
        reasons.append("静音要求不符")

    if constraints.max_landed_cny is not None:
        landed_cny = estimate_landed_cny(item)
        if landed_cny > constraints.max_landed_cny:
            reasons.append("预估到手价超预算")

    return reasons


__all__ = [
    "SearchConstraints",
    "candidate_rejection_reasons",
    "estimate_landed_cny",
    "parse_search_constraints",
]
