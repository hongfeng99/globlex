from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.recall.local_embeddings import embed_item
from app.utils.path_utils import PROJECT_ROOT


DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "demo_items.json"
)
DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "data" / "item_index.faiss"
)


def build_demo_index(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> tuple[Path, Path, int]:
    import faiss

    raw_items = json.loads(
        catalog_path.read_text(encoding="utf-8")
    )
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("演示商品目录必须是非空数组。")

    items: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    required_fields = {
        "item_id",
        "platform",
        "title",
        "price",
        "currency",
    }

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("演示商品必须是 JSON 对象。")
        missing = required_fields - raw_item.keys()
        if missing:
            raise ValueError(
                "演示商品缺少字段："
                + ", ".join(sorted(missing))
            )
        item = dict(raw_item)
        embeddings.append(embed_item(item))
        items.append(item)

    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    faiss.write_index(index, str(index_path))

    metadata_path = index_path.with_suffix(
        ".meta.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                str(index): item
                for index, item in enumerate(items)
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index_path, metadata_path, len(items)


__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_INDEX_PATH",
    "build_demo_index",
]
