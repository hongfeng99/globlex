from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.eval.recall_metrics import (
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from app.tools.category_insight import (
    _recall_cards,
)


async def run(path: Path, k: int) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    values = []
    for row in rows:
        cards = await _recall_cards(
            row["query"], k
        )
        retrieved = [
            card.card_id for card in cards
        ]
        relevant = row["relevant"]
        values.append(
            (
                recall_at_k(
                    retrieved, relevant, k
                ),
                mrr(retrieved, relevant),
                ndcg_at_k(
                    retrieved, relevant, k
                ),
            )
        )
    count = len(values) or 1
    return {
        "recall": sum(x[0] for x in values)
        / count,
        "mrr": sum(x[1] for x in values)
        / count,
        "ndcg": sum(x[2] for x in values)
        / count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                run(args.dataset, args.k)
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
