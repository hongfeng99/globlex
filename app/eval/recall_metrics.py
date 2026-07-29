from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k 必须大于 0。")

    top_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    return len(
        top_k & relevant_set
    ) / len(relevant_set)


def mrr(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    relevant_set = set(relevant)

    for rank, item_id in enumerate(
        retrieved,
        start=1,
    ):
        if item_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k 必须大于 0。")

    relevance = {
        item_id: len(relevant) - index
        for index, item_id in enumerate(
            relevant
        )
    }

    def dcg(items: Sequence[str]) -> float:
        return sum(
            relevance.get(item_id, 0)
            / math.log2(rank + 1)
            for rank, item_id in enumerate(
                items[:k],
                start=1,
            )
        )

    ideal = dcg(list(relevant))
    return (
        dcg(retrieved) / ideal
        if ideal > 0
        else 0.0
    )


__all__ = [
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
