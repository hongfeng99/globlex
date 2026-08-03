from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from app.config import env_int


def embedding_dimension() -> int:
    return env_int(
        "TOWER_EMBEDDING_DIM",
        256,
        minimum=32,
    )


def _tokens(text: str) -> list[str]:
    normalized = " ".join(text.lower().split())
    words = re.findall(
        r"[a-z0-9]+|[\u4e00-\u9fff]",
        normalized,
    )
    compact = "".join(words)
    grams = [
        compact[index : index + size]
        for size in (2, 3)
        for index in range(
            max(0, len(compact) - size + 1)
        )
    ]
    return [*words, *grams]


def embed_text(text: str) -> list[float]:
    dimension = embedding_dimension()
    vector = [0.0] * dimension

    for token in _tokens(text):
        digest = sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(
            digest[:4], "big"
        ) % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(
        sum(value * value for value in vector)
    )
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def embed_item(item: Mapping[str, Any]) -> list[float]:
    attributes = json.dumps(
        item.get("attributes", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    text = " ".join(
        str(value)
        for value in (
            item.get("title", ""),
            item.get("category", ""),
            attributes,
        )
        if value
    )
    return embed_text(text)


__all__ = [
    "embed_item",
    "embed_text",
    "embedding_dimension",
]
