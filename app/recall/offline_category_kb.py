from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.recall.category_kb import CategoryCard
from app.recall.fx import to_base
from app.recall.offline_catalog import (
    CATEGORY_SPECS,
    DEFAULT_OFFLINE_CATALOG_PATH,
    SYNTHETIC_NOTICE,
)
from app.utils.path_utils import PROJECT_ROOT


CATEGORY_KB_VERSION = "offline-category-kb-v1"
DEFAULT_CATEGORY_CARDS_PATH = (
    PROJECT_ROOT / "data" / "offline_category_cards.json"
)
DEFAULT_CATEGORY_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "offline_category_cards.manifest.json"
)


def _cny_price(item: dict[str, Any]) -> float:
    return round(
        to_base(
            float(item["price"]),
            str(item["currency"]),
        ),
        2,
    )


def _distribution_summary(
    name: str,
    values: list[str],
    *,
    limit: int = 6,
) -> str:
    counts = Counter(value for value in values if value)
    total = sum(counts.values())
    if total <= 0:
        return f"{name}：暂无统计"
    tokens = [
        f"{value} {count / total:.0%}"
        for value, count in counts.most_common(limit)
    ]
    return f"{name}：" + " / ".join(tokens)


def _percentile(
    values: list[float],
    ratio: float,
) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return round(ordered[index], 2)


def generate_category_cards(
    catalog: list[dict[str, Any]],
) -> list[CategoryCard]:
    """Aggregate the synthetic catalog into deterministic category cards."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in catalog:
        grouped[str(item["category_key"])].append(item)

    cards: list[CategoryCard] = []
    updated_at = "2026-08-08"
    for spec in CATEGORY_SPECS:
        items = grouped.get(spec.key, [])
        if not items:
            raise ValueError(f"离线商品库缺少品类：{spec.key}")

        ranked = sorted(
            items,
            key=lambda item: (
                float(item.get("rating", 0)),
                int(item.get("sales", 0)),
            ),
            reverse=True,
        )
        evidence = [
            "|".join(
                (
                    str(item["title"]),
                    f"{_cny_price(item):.2f}",
                    (
                        "离线模拟数据中评分 "
                        f"{float(item.get('rating', 0)):.1f}，"
                        f"销量 {int(item.get('sales', 0))}"
                    ),
                )
            )
            for item in ranked[:5]
        ]
        focus = (
            list(spec.components)
            if spec.components
            else list(spec.features[:5])
        )
        focus_label = (
            "典型组成" if spec.components else "热销关注点"
        )
        cards.append(
            CategoryCard(
                card_id=f"{spec.key}-bestseller",
                category=spec.name,
                card_type="bestseller",
                summary=(
                    f"{focus_label}：" + "、".join(focus)
                ),
                raw_evidence=evidence,
                last_updated=updated_at,
                confidence=0.86,
            )
        )

        materials: list[str] = []
        features: list[str] = []
        for item in items:
            attributes = item.get("attributes", {})
            if not isinstance(attributes, dict):
                continue
            material = str(attributes.get("material", "")).strip()
            if material:
                materials.append(material)
            raw_features = attributes.get("features", [])
            if isinstance(raw_features, list):
                features.extend(
                    str(value).strip()
                    for value in raw_features
                    if str(value).strip()
                )

        cards.extend(
            (
                CategoryCard(
                    card_id=f"{spec.key}-attribute-material",
                    category=spec.name,
                    card_type="attribute",
                    summary=_distribution_summary(
                        "材质", materials
                    ),
                    last_updated=updated_at,
                    confidence=0.9,
                ),
                CategoryCard(
                    card_id=f"{spec.key}-attribute-feature",
                    category=spec.name,
                    card_type="attribute",
                    summary=_distribution_summary(
                        "功能", features
                    ),
                    last_updated=updated_at,
                    confidence=0.9,
                ),
            )
        )

        prices = [_cny_price(item) for item in items]
        minimum = round(min(prices), 2)
        first_cut = _percentile(prices, 1 / 3)
        second_cut = _percentile(prices, 2 / 3)
        maximum = round(max(prices), 2)
        tiers = (
            ("budget", "便宜款", minimum, first_cut),
            ("mid", "中档", first_cut, second_cut),
            ("premium", "高端", second_cut, maximum),
        )
        for tier, label, low, high in tiers:
            cards.append(
                CategoryCard(
                    card_id=f"{spec.key}-price-{tier}",
                    category=spec.name,
                    card_type="price_range",
                    summary=(
                        f"{label} {low:.2f}-{high:.2f} 元；"
                        "基于四个平台离线模拟标价换算，不含运费关税"
                    ),
                    last_updated=updated_at,
                    confidence=0.88,
                )
            )

    return cards


def render_category_card_text(card: CategoryCard) -> str:
    evidence = "；".join(card.raw_evidence)
    return "\n".join(
        value
        for value in (
            f"品类：{card.category}",
            f"知识类型：{card.card_type}",
            f"摘要：{card.summary}",
            f"证据：{evidence}" if evidence else "",
        )
        if value
    )


def write_category_cards(
    catalog_path: Path = DEFAULT_OFFLINE_CATALOG_PATH,
    cards_path: Path = DEFAULT_CATEGORY_CARDS_PATH,
    manifest_path: Path = DEFAULT_CATEGORY_MANIFEST_PATH,
) -> tuple[Path, Path, int]:
    raw_catalog = json.loads(
        catalog_path.read_text(encoding="utf-8")
    )
    if not isinstance(raw_catalog, list) or not raw_catalog:
        raise ValueError("离线商品目录必须是非空数组。")
    cards = generate_category_cards(raw_catalog)
    payload = [card.model_dump() for card in cards]
    cards_path.parent.mkdir(parents=True, exist_ok=True)
    cards_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    type_counts = Counter(card.card_type for card in cards)
    category_counts = Counter(card.category for card in cards)
    manifest = {
        "category_kb_version": CATEGORY_KB_VERSION,
        "schema_version": "1.0",
        "data_mode": "synthetic",
        "notice": SYNTHETIC_NOTICE,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": sha256(
            catalog_path.read_bytes()
        ).hexdigest(),
        "card_count": len(cards),
        "category_counts": dict(sorted(category_counts.items())),
        "card_type_counts": dict(sorted(type_counts.items())),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cards_path, manifest_path, len(cards)


__all__ = [
    "CATEGORY_KB_VERSION",
    "DEFAULT_CATEGORY_CARDS_PATH",
    "DEFAULT_CATEGORY_MANIFEST_PATH",
    "generate_category_cards",
    "render_category_card_text",
    "write_category_cards",
]
