from __future__ import annotations

import asyncio
import json

from app.recall.opensearch_client import get_opensearch_client
from app.tools.category_insight import INDEX_NAME, category_insight


async def main() -> None:
    client = get_opensearch_client()
    count = int(client.count(index=INDEX_NAME)["count"])
    result = await category_insight.ainvoke(
        {"category": "机械键盘", "depth": "deep"}
    )
    if count != 108:
        raise RuntimeError(f"Category KB 文档数异常：{count}")
    if (
        not result.bestsellers
        or len(result.attributes) < 2
        or len(result.price_tiers) != 3
        or result.confidence <= 0
    ):
        raise RuntimeError(
            "CategoryInsight 未返回完整的离线知识。"
        )
    print(
        json.dumps(
            {
                "index": INDEX_NAME,
                "document_count": count,
                "category": result.category,
                "components": result.components,
                "bestseller_count": len(result.bestsellers),
                "attribute_count": len(result.attributes),
                "price_tier_count": len(result.price_tiers),
                "confidence": result.confidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

