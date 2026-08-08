from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from opensearchpy import helpers

from app.recall.category_kb import CategoryCard
from app.recall.embedding_backend import get_embedding_backend
from app.recall.offline_category_kb import (
    DEFAULT_CATEGORY_CARDS_PATH,
    render_category_card_text,
)
from app.recall.opensearch_client import get_opensearch_client
from app.tools.category_insight import INDEX_NAME
from app.utils.path_utils import PROJECT_ROOT


PIPELINE_NAME = "globex_hybrid_pipeline"
DEFAULT_INDEX_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "category_kb_index.manifest.json"
)


def build_index_body(
    dimension: int,
    *,
    analyzer: str = "standard",
) -> dict[str, Any]:
    if dimension <= 0:
        raise ValueError("dimension 必须大于 0。")
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "card_id": {"type": "keyword"},
                "category": {
                    "type": "text",
                    "analyzer": analyzer,
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                },
                "card_type": {"type": "keyword"},
                "summary": {
                    "type": "text",
                    "analyzer": analyzer,
                },
                "raw_evidence": {
                    "type": "text",
                    "analyzer": analyzer,
                },
                "last_updated": {"type": "date"},
                "confidence": {"type": "float"},
                "content_vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "innerproduct",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16,
                        },
                    },
                },
            },
        },
    }


def build_pipeline_body() -> dict[str, Any]:
    return {
        "description": "KNN + BM25 双路召回归一与加权融合",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [0.7, 0.3]
                        },
                    },
                }
            }
        ],
    }


def _load_cards(path: Path) -> list[CategoryCard]:
    raw_cards = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cards, list) or not raw_cards:
        raise ValueError("离线品类知识卡必须是非空数组。")
    return [CategoryCard.model_validate(card) for card in raw_cards]


def _existing_dimension(client: Any) -> int | None:
    mapping = client.indices.get_mapping(index=INDEX_NAME)
    index_mapping = mapping.get(INDEX_NAME, mapping)
    try:
        return int(
            index_mapping["mappings"]["properties"]
            ["content_vector"]["dimension"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def seed_category_kb(
    cards_path: Path = DEFAULT_CATEGORY_CARDS_PATH,
    *,
    recreate: bool = False,
    manifest_path: Path = DEFAULT_INDEX_MANIFEST_PATH,
) -> tuple[int, int]:
    """Create/update the category index and return indexed/total counts."""

    cards = _load_cards(cards_path)
    backend = get_embedding_backend()
    spec = backend.spec
    vectors = backend.encode_documents(
        [render_category_card_text(card) for card in cards]
    )
    if len(vectors) != len(cards):
        raise RuntimeError("品类卡数量与向量数量不一致。")

    client = get_opensearch_client()
    if not client.ping():
        raise RuntimeError(
            "无法连接 OpenSearch；请先启动 localhost:9200。"
        )

    client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{PIPELINE_NAME}",
        body=build_pipeline_body(),
    )
    exists = bool(client.indices.exists(index=INDEX_NAME))
    if exists and recreate:
        client.indices.delete(index=INDEX_NAME)
        exists = False
    analyzer = os.getenv(
        "CATEGORY_KB_ANALYZER", "standard"
    ).strip() or "standard"
    if not exists:
        client.indices.create(
            index=INDEX_NAME,
            body=build_index_body(
                spec.dimension,
                analyzer=analyzer,
            ),
        )
    else:
        existing_dimension = _existing_dimension(client)
        if existing_dimension != spec.dimension:
            raise RuntimeError(
                "现有 Category KB 向量维度与当前模型不一致："
                f"{existing_dimension} != {spec.dimension}；"
                "请使用 --recreate 重建。"
            )

    actions = [
        {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": card.card_id,
            "_source": {
                **card.model_dump(mode="json"),
                "content_vector": vector,
            },
        }
        for card, vector in zip(cards, vectors)
    ]
    indexed, errors = helpers.bulk(
        client,
        actions,
        refresh=True,
        raise_on_error=False,
    )
    if errors:
        raise RuntimeError(
            f"Category KB 批量写入失败：{errors[:3]}"
        )
    total = int(client.count(index=INDEX_NAME)["count"])
    manifest_path.write_text(
        json.dumps(
            {
                "index_name": INDEX_NAME,
                "pipeline_name": PIPELINE_NAME,
                "card_count": total,
                "indexed_in_run": indexed,
                "analyzer": analyzer,
                "embedding": spec.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return indexed, total


def get_category_kb_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "index_name": INDEX_NAME,
        "pipeline_name": PIPELINE_NAME,
        "document_count": 0,
    }
    try:
        client = get_opensearch_client()
        if not client.ping():
            status["reason"] = "opensearch_unreachable"
            return status
        if not client.indices.exists(index=INDEX_NAME):
            status["reason"] = "index_missing"
            return status
        client.transport.perform_request(
            "GET",
            f"/_search/pipeline/{PIPELINE_NAME}",
        )
        count = int(client.count(index=INDEX_NAME)["count"])
        status["document_count"] = count
        status["available"] = count > 0
        if count <= 0:
            status["reason"] = "index_empty"
    except Exception as exc:
        status["reason"] = type(exc).__name__
    return status


__all__ = [
    "DEFAULT_INDEX_MANIFEST_PATH",
    "PIPELINE_NAME",
    "build_index_body",
    "build_pipeline_body",
    "get_category_kb_status",
    "seed_category_kb",
]
