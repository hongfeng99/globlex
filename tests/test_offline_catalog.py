from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from app.recall.ann import AnnClient
from app.recall.demo_index import (
    DEFAULT_CATALOG_PATH,
    build_demo_index,
)
from app.recall.local_embeddings import embed_text
from app.recall.offline_catalog import (
    CATEGORY_SPECS,
    DEFAULT_OFFLINE_CATALOG_PATH,
    DEFAULT_OFFLINE_MANIFEST_PATH,
    DEFAULT_SEED,
    DEFAULT_VARIANTS_PER_PLATFORM,
    PLATFORMS,
    generate_offline_catalog,
)


def test_offline_catalog_generation_is_deterministic(
) -> None:
    first = generate_offline_catalog(
        variants_per_platform=2,
        seed=1234,
    )
    second = generate_offline_catalog(
        variants_per_platform=2,
        seed=1234,
    )

    assert first == second


def test_default_offline_catalog_contract() -> None:
    catalog = json.loads(
        DEFAULT_OFFLINE_CATALOG_PATH.read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        DEFAULT_OFFLINE_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    expected_count = (
        len(CATEGORY_SPECS)
        * len(PLATFORMS)
        * DEFAULT_VARIANTS_PER_PLATFORM
    )

    assert DEFAULT_CATALOG_PATH == (
        DEFAULT_OFFLINE_CATALOG_PATH
    )
    assert len(catalog) == expected_count
    assert manifest["item_count"] == expected_count
    assert manifest["seed"] == DEFAULT_SEED

    item_ids = {
        item["item_id"] for item in catalog
    }
    assert len(item_ids) == expected_count
    assert {
        item["platform"] for item in catalog
    } == {platform.name for platform in PLATFORMS}
    assert {
        item["category"] for item in catalog
    } == {category.name for category in CATEGORY_SPECS}
    assert {
        item["category_key"] for item in catalog
    } == {category.key for category in CATEGORY_SPECS}
    cell_counts = Counter(
        (
            item["category_key"],
            item["platform"],
        )
        for item in catalog
    )
    assert set(cell_counts.values()) == {
        DEFAULT_VARIANTS_PER_PLATFORM
    }

    for item in catalog:
        assert item["data_mode"] == "synthetic"
        assert item["source"] == "offline_catalog"
        assert item["price"] > 0
        assert item["currency"] in {"CNY", "USD"}
        assert 0 <= item["rating"] <= 5
        assert item["sales"] >= 0
        assert item["product_url"] is None
        assert item["is_purchasable"] is False
        assert item["availability"] == "in_stock"
        assert item["stock"] > 0
        assert item["image_url"] is None
        assert isinstance(item["attributes"], dict)

    keyboards = [
        item
        for item in catalog
        if item["category_key"] == "mechanical-keyboard"
    ]
    assert keyboards
    for item in keyboards:
        attributes = item["attributes"]
        assert attributes["switch_type"] in {
            "青轴",
            "红轴",
            "茶轴",
            "静音红轴",
        }
        assert attributes["connection_modes"]
        assert attributes["layout"]

    cycling_kits = [
        item
        for item in catalog
        if item["category_key"] == "cycling-kit"
    ]
    assert cycling_kits
    for item in cycling_kits:
        attributes = item["attributes"]
        assert attributes["components"] == [
            "骑行上衣",
            "背带骑行裤",
            "骑行手套",
        ]
        assert attributes["pack_size"] == 3
        assert "3件套" in item["title"]


def test_offline_catalog_cycling_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOWER_BACKEND", "hash")
    index_path = tmp_path / "offline.faiss"
    build_demo_index(
        DEFAULT_OFFLINE_CATALOG_PATH,
        index_path,
        backend_name="hash",
    )
    client = AnnClient(index_path)

    for platform in PLATFORMS:
        results = client.search(
            embed_text(
                "男士 XL 公路车骑行套装 "
                "专业速干面料"
            ),
            top_k=5,
            platform=platform.name,
        )
        assert results
        assert any(
            str(item["category"]).startswith(
                "骑行"
            )
            for item in results
        )
