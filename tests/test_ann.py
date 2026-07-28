import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.recall.ann import AnnClient


class FakeIndex:
    def __init__(
        self,
        scores: list[float],
        indexes: list[int],
    ) -> None:
        self.scores = scores
        self.indexes = indexes
        self.requested_top_k: int | None = None

    def search(
        self,
        vectors: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.requested_top_k = top_k
        assert vectors.dtype == np.float32
        assert vectors.shape == (1, 2)

        return (
            np.asarray(
                [self.scores],
                dtype=np.float32,
            ),
            np.asarray(
                [self.indexes],
                dtype=np.int64,
            ),
        )


def test_ann_filters_platform_and_skips_missing() -> None:
    index = FakeIndex(
        scores=[0.9, 0.8, 0.7, 0.0],
        indexes=[0, 1, 2, -1],
    )
    metadata: dict[
        int,
        dict[str, Any],
    ] = {
        0: {
            "item_id": "a",
            "platform": "amazon",
        },
        1: {
            "item_id": "b",
            "platform": "ebay",
        },
        2: {
            "item_id": "c",
            "platform": "amazon",
        },
    }
    client = AnnClient(
        index=index,
        metadata=metadata,
    )

    results = client.search(
        [0.1, 0.2],
        top_k=2,
        platform="amazon",
    )

    assert index.requested_top_k == 6
    assert [
        result["item_id"]
        for result in results
    ] == ["a", "c"]
    assert results[0]["score"] > 0.89


def test_ann_lazy_loads_faiss_and_metadata(
    tmp_path: Path,
) -> None:
    index_path = (
        tmp_path
        / "item_index.faiss"
    )
    metadata_path = (
        tmp_path
        / "item_index.meta.json"
    )

    index = faiss.IndexFlatIP(2)
    index.add(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    faiss.write_index(
        index,
        str(index_path),
    )

    metadata_path.write_text(
        json.dumps(
            {
                "0": {
                    "item_id": "amazon-1",
                    "platform": "amazon",
                },
                "1": {
                    "item_id": "ebay-1",
                    "platform": "ebay",
                },
            }
        ),
        encoding="utf-8",
    )

    client = AnnClient(index_path)
    results = client.search(
        [1.0, 0.0],
        top_k=1,
        platform="amazon",
    )

    assert len(results) == 1
    assert results[0]["item_id"] == (
        "amazon-1"
    )
