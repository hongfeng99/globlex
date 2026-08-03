from __future__ import annotations

import json
from pathlib import Path

from app.recall.ann import AnnClient
from app.recall.demo_index import build_demo_index
from app.recall.local_embeddings import embed_text


def test_build_demo_index_can_be_searched(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "items.json"
    index_path = tmp_path / "items.faiss"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "item_id": "travel-1",
                    "platform": "amazon",
                    "title": "防水旅行收纳袋",
                    "category": "旅行收纳",
                    "price": 10,
                    "currency": "USD",
                    "attributes": {},
                },
                {
                    "item_id": "mug-1",
                    "platform": "amazon",
                    "title": "陶瓷咖啡杯",
                    "category": "咖啡杯",
                    "price": 8,
                    "currency": "USD",
                    "attributes": {},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _, metadata_path, count = build_demo_index(
        catalog_path,
        index_path,
    )
    results = AnnClient(index_path).search(
        embed_text("旅行收纳袋"),
        top_k=1,
        platform="amazon",
    )

    assert count == 2
    assert metadata_path.is_file()
    assert results[0]["item_id"] == "travel-1"
