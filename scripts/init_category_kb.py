from __future__ import annotations

import argparse
from pathlib import Path

from app.recall.category_kb_admin import seed_category_kb
from app.recall.offline_category_kb import (
    DEFAULT_CATEGORY_CARDS_PATH,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="创建并导入 Globex 离线模拟品类知识库。"
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=DEFAULT_CATEGORY_CARDS_PATH,
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除并重建 globex_category_kb。",
    )
    arguments = parser.parse_args()
    indexed, total = seed_category_kb(
        arguments.cards,
        recreate=arguments.recreate,
    )
    print(
        f"Category KB 初始化完成：本次写入 {indexed}，"
        f"索引总数 {total}。"
    )
