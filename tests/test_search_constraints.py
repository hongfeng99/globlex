from __future__ import annotations

from app.recall.offline_catalog import generate_offline_catalog
from app.recall.search_constraints import (
    candidate_rejection_reasons,
    estimate_landed_cny,
    parse_search_constraints,
)


def test_parse_keyboard_hard_constraints() -> None:
    constraints = parse_search_constraints(
        "200 元以内，青轴，无线，办公机械键盘"
    )

    assert constraints.category_key == "mechanical-keyboard"
    assert constraints.max_landed_cny == 200
    assert constraints.switch_type == "青轴"
    assert constraints.connection == "无线"
    assert constraints.quiet_required is False


def test_blue_switch_is_canonical_chinese_green_switch() -> None:
    constraints = parse_search_constraints(
        "mechanical keyboard blue switch"
    )

    assert constraints.switch_type == "青轴"


def test_standalone_supplemental_amount_is_budget() -> None:
    constraints = parse_search_constraints(
        "我要买机械键盘\n\n用户补充信息：\n"
        "500元\n办公\n青轴\n尺寸无要求\n无线"
    )

    assert constraints.max_landed_cny == 500


def test_cycling_bundle_alias_is_a_hard_category() -> None:
    constraints = parse_search_constraints(
        "骑行三件套，500元以下"
    )

    assert constraints.category_key == "cycling-kit"
    assert constraints.category_name == "骑行套装"
    assert constraints.max_landed_cny == 500


def test_generated_catalog_has_exact_budget_match() -> None:
    catalog = generate_offline_catalog()
    candidate = next(
        item
        for item in catalog
        if item["platform"] == "aliexpress"
        and item["category_key"] == "mechanical-keyboard"
        and item["attributes"]["switch_type"] == "青轴"
        and "蓝牙" in item["attributes"]["connection_modes"]
        and estimate_landed_cny(item) <= 200
    )
    constraints = parse_search_constraints(
        "200元以内，青轴，无线，办公机械键盘"
    )

    assert candidate_rejection_reasons(candidate, constraints) == []

    wrong = dict(candidate)
    wrong["attributes"] = {
        **candidate["attributes"],
        "switch_type": "红轴",
    }
    assert "轴体不符" in candidate_rejection_reasons(
        wrong, constraints
    )
