from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Protocol

import numpy as np
from dotenv import load_dotenv

from app.config import env_bool, env_int
from app.recall.local_embeddings import (
    embed_item,
    embed_text,
    embedding_dimension,
    render_item_text,
)
from app.utils.path_utils import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    backend: str
    model_name: str
    model_revision: str
    dimension: int
    normalized: bool
    query_prefix: str
    item_prefix: str
    user_prefix: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingBackend(Protocol):
    @property
    def spec(self) -> EmbeddingSpec:
        ...

    def encode_query(self, query: str) -> list[float]:
        ...

    def encode_preferences(
        self,
        preferences: Sequence[str],
    ) -> list[float]:
        ...

    def encode_item(
        self,
        item: Mapping[str, Any],
    ) -> list[float]:
        ...

    def encode_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]]:
        ...

    def encode_items(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> list[list[float]]:
        ...


def resolve_embedding_backend_name(
    backend_name: str | None = None,
) -> str:
    value = (
        backend_name
        if backend_name is not None
        else os.getenv("TOWER_BACKEND", "hash")
    ).strip().lower().replace("-", "_")
    aliases = {
        "local": "hash",
        "sentence_transformer": "sentence_transformers",
        "sentence_transformers": "sentence_transformers",
        "st": "sentence_transformers",
    }
    value = aliases.get(value, value)
    if value not in {"hash", "sentence_transformers", "http"}:
        raise RuntimeError(
            "TOWER_BACKEND 必须是 sentence_transformers、hash/local 或 http。"
        )
    return value


def configured_embedding_identity(
    backend_name: str | None = None,
) -> dict[str, str]:
    backend = resolve_embedding_backend_name(backend_name)
    if backend == "sentence_transformers":
        return {
            "backend": backend,
            "model_name": os.getenv(
                "TOWER_MODEL_NAME", DEFAULT_MODEL_NAME
            ).strip(),
            "model_revision": os.getenv(
                "TOWER_MODEL_REVISION", ""
            ).strip(),
            "query_prefix": os.getenv(
                "TOWER_QUERY_PREFIX", "query: "
            ),
            "item_prefix": os.getenv(
                "TOWER_ITEM_PREFIX", "passage: "
            ),
            "user_prefix": os.getenv(
                "TOWER_USER_PREFIX", "query: "
            ),
        }
    if backend == "hash":
        return {
            "backend": backend,
            "model_name": "globex-hash-v1",
            "model_revision": "",
            "query_prefix": "",
            "item_prefix": "",
            "user_prefix": "",
        }
    return {
        "backend": "http",
        "model_name": "",
        "model_revision": "",
        "query_prefix": "",
        "item_prefix": "",
        "user_prefix": "",
    }


class HashEmbeddingBackend:
    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            **configured_embedding_identity("hash"),
            dimension=embedding_dimension(),
            normalized=True,
        )

    def encode_query(self, query: str) -> list[float]:
        return embed_text(query)

    def encode_preferences(
        self,
        preferences: Sequence[str],
    ) -> list[float]:
        return embed_text("；".join(preferences))

    def encode_item(
        self,
        item: Mapping[str, Any],
    ) -> list[float]:
        return embed_item(item)

    def encode_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]]:
        return [embed_text(document) for document in documents]

    def encode_items(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> list[list[float]]:
        return [self.encode_item(item) for item in items]


@lru_cache(maxsize=4)
def _load_sentence_transformer_cached(
    model_name: str,
    model_revision: str,
    device: str,
    local_files_only: bool,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers 未安装；请运行 "
            'python -m pip install -e ".[embeddings]"，'
            "或将 TOWER_BACKEND=hash 作为离线回退。"
        ) from exc

    kwargs: dict[str, Any] = {
        "device": device,
        "local_files_only": local_files_only,
    }
    if model_revision:
        kwargs["revision"] = model_revision
    return SentenceTransformer(model_name, **kwargs)


_MODEL_LOAD_LOCK = Lock()


def _load_sentence_transformer(
    model_name: str,
    model_revision: str,
    device: str,
    local_files_only: bool,
) -> Any:
    # lru_cache may compute the same missing key concurrently. Serializing
    # first load avoids loading a several-hundred-MB model twice at startup.
    with _MODEL_LOAD_LOCK:
        return _load_sentence_transformer_cached(
            model_name,
            model_revision,
            device,
            local_files_only,
        )


class SentenceTransformerEmbeddingBackend:
    def __init__(self) -> None:
        identity = configured_embedding_identity(
            "sentence_transformers"
        )
        self.model_name = identity["model_name"]
        self.model_revision = identity["model_revision"]
        self.query_prefix = identity["query_prefix"]
        self.item_prefix = identity["item_prefix"]
        self.user_prefix = identity["user_prefix"]
        self.device = os.getenv("TOWER_DEVICE", "cpu").strip() or "cpu"
        self.local_files_only = env_bool(
            "TOWER_LOCAL_FILES_ONLY", False
        )
        self.batch_size = env_int(
            "TOWER_BATCH_SIZE", 32, minimum=1
        )

    @property
    def _model(self) -> Any:
        return _load_sentence_transformer(
            self.model_name,
            self.model_revision,
            self.device,
            self.local_files_only,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        dimension_getter = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        )
        if dimension_getter is None:
            dimension_getter = (
                self._model.get_sentence_embedding_dimension
            )
        dimension = dimension_getter()
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("Embedding 模型没有返回有效维度。")
        return EmbeddingSpec(
            backend="sentence_transformers",
            model_name=self.model_name,
            model_revision=self.model_revision,
            dimension=dimension,
            normalized=True,
            query_prefix=self.query_prefix,
            item_prefix=self.item_prefix,
            user_prefix=self.user_prefix,
        )

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] != len(texts):
            raise RuntimeError("Embedding 模型返回了无效矩阵。")
        if not np.isfinite(values).all():
            raise RuntimeError("Embedding 模型返回了非有限数值。")
        return values.tolist()

    def encode_query(self, query: str) -> list[float]:
        return self._encode([self.query_prefix + query])[0]

    def encode_preferences(
        self,
        preferences: Sequence[str],
    ) -> list[float]:
        text = "用户长期偏好：" + "；".join(preferences)
        return self._encode([self.user_prefix + text])[0]

    def encode_item(
        self,
        item: Mapping[str, Any],
    ) -> list[float]:
        return self._encode(
            [self.item_prefix + render_item_text(item)]
        )[0]

    def encode_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]]:
        return self._encode(
            [self.item_prefix + document for document in documents]
        )

    def encode_items(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> list[list[float]]:
        return self._encode(
            [
                self.item_prefix + render_item_text(item)
                for item in items
            ]
        )


def get_embedding_backend(
    backend_name: str | None = None,
) -> EmbeddingBackend:
    backend = resolve_embedding_backend_name(backend_name)
    if backend == "hash":
        return HashEmbeddingBackend()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbeddingBackend()
    raise RuntimeError(
        "HTTP 三塔只能通过 TowerClient 在线调用，不能用于本地索引构建。"
    )


__all__ = [
    "DEFAULT_MODEL_NAME",
    "EmbeddingBackend",
    "EmbeddingSpec",
    "configured_embedding_identity",
    "get_embedding_backend",
    "resolve_embedding_backend_name",
]
