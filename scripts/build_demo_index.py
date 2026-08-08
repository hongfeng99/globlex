from __future__ import annotations

import argparse
from pathlib import Path

from app.recall.demo_index import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INDEX_PATH,
    build_demo_index,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "从离线模拟商品目录构建本地 FAISS 索引。"
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    parser.add_argument(
        "--backend",
        choices=("sentence_transformers", "hash", "local"),
        default=None,
        help="覆盖 TOWER_BACKEND；local 是 hash 的兼容别名。",
    )
    arguments = parser.parse_args()
    index_path, metadata_path, count = (
        build_demo_index(
            arguments.catalog,
            arguments.output,
            backend_name=arguments.backend,
        )
    )
    print(
        f"已写入 {count} 件离线模拟商品："
        f"{index_path} / {metadata_path}"
    )
