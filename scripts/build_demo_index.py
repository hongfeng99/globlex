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
        description="构建本地演示商品 FAISS 索引。"
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
    arguments = parser.parse_args()
    index_path, metadata_path, count = (
        build_demo_index(
            arguments.catalog,
            arguments.output,
        )
    )
    print(
        f"已写入 {count} 件演示商品："
        f"{index_path} / {metadata_path}"
    )
