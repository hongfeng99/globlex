from __future__ import annotations

import argparse
from pathlib import Path

from app.recall.offline_catalog import (
    DEFAULT_OFFLINE_CATALOG_PATH,
    DEFAULT_OFFLINE_MANIFEST_PATH,
    DEFAULT_SEED,
    DEFAULT_VARIANTS_PER_PLATFORM,
    write_offline_catalog,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "生成确定性的离线模拟商品目录和清单。"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OFFLINE_CATALOG_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OFFLINE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--variants-per-platform",
        type=int,
        default=DEFAULT_VARIANTS_PER_PLATFORM,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    arguments = parser.parse_args()
    catalog_path, manifest_path, count = (
        write_offline_catalog(
            arguments.output,
            arguments.manifest,
            variants_per_platform=(
                arguments.variants_per_platform
            ),
            seed=arguments.seed,
        )
    )
    print(
        f"已生成 {count} 件离线模拟商品："
        f"{catalog_path} / {manifest_path}"
    )
