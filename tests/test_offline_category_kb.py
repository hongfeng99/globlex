from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from app.recall.offline_catalog import (
    CATEGORY_SPECS,
    DEFAULT_OFFLINE_CATALOG_PATH,
)
from app.recall.offline_category_kb import (
    generate_category_cards,
    write_category_cards,
)
from app.tools.category_insight import (
    _extract_attributes,
    _extract_bestsellers,
    _extract_price_tiers,
)


def _catalog() -> list[dict]:
    value = json.loads(
        DEFAULT_OFFLINE_CATALOG_PATH.read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, list)
    return value


def test_category_card_generation_is_deterministic() -> None:
    first = generate_category_cards(_catalog())
    second = generate_category_cards(_catalog())
    assert first == second


def test_category_card_contract_and_extractors() -> None:
    cards = generate_category_cards(_catalog())
    counts = Counter(card.card_type for card in cards)
    categories = {card.category for card in cards}

    assert len(cards) == len(CATEGORY_SPECS) * 6
    assert counts == {
        "bestseller": len(CATEGORY_SPECS),
        "attribute": len(CATEGORY_SPECS) * 2,
        "price_range": len(CATEGORY_SPECS) * 3,
    }
    assert categories == {
        spec.name for spec in CATEGORY_SPECS
    }

    keyboard_cards = [
        card
        for card in cards
        if card.category == "机械键盘"
    ]
    assert _extract_bestsellers(keyboard_cards)
    assert len(_extract_attributes(keyboard_cards)) == 2
    assert len(_extract_price_tiers(keyboard_cards)) == 3


def test_write_category_cards(
    tmp_path: Path,
) -> None:
    cards_path = tmp_path / "cards.json"
    manifest_path = tmp_path / "cards.manifest.json"
    _, _, count = write_category_cards(
        cards_path=cards_path,
        manifest_path=manifest_path,
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert count == len(CATEGORY_SPECS) * 6
    assert cards_path.is_file()
    assert manifest["card_count"] == count
    assert manifest["data_mode"] == "synthetic"

