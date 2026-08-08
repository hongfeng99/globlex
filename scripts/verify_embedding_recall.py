from __future__ import annotations

import asyncio
import json

from app.recall.ann import ann_client
from app.recall.towers import tower_client
from app.tools.item_search import _fuse_embeddings


PLATFORMS = ("amazon", "shopee", "aliexpress", "ebay")


async def main() -> None:
    query = "办公室静音机械键盘"
    preferences = ["偏好消音填充", "不喜欢 RGB 灯"]
    query_embedding, user_embedding = await asyncio.gather(
        tower_client.encode_query(query),
        tower_client.encode_user("smoke-user", preferences),
    )
    fused_embedding = _fuse_embeddings(
        query_embedding,
        user_embedding,
    )

    report: dict[str, object] = {
        "dimension": len(query_embedding),
        "platforms": {},
    }
    platforms = report["platforms"]
    assert isinstance(platforms, dict)

    for platform in PLATFORMS:
        semantic = ann_client.search(
            query_embedding,
            top_k=20,
            platform=platform,
        )
        personalized = ann_client.search(
            fused_embedding,
            top_k=20,
            platform=platform,
        )
        if len(semantic) != 20 or len(personalized) != 20:
            raise RuntimeError(
                f"{platform} 未返回完整 20 条候选。"
            )
        platforms[platform] = {
            "semantic_count": len(semantic),
            "personalized_count": len(personalized),
            "semantic_top": semantic[0]["item_id"],
            "personalized_top": personalized[0]["item_id"],
            "ranking_changed": [
                item["item_id"] for item in semantic
            ]
            != [item["item_id"] for item in personalized],
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    await tower_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

