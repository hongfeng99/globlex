from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from dotenv import load_dotenv

from app.utils.path_utils import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


class FaissIndex(Protocol):
    """
    AnnClient 测试与运行时所需的最小 Faiss 接口。
    """

    def search(
        self,
        vectors: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class AnnClient:
    """
    商品 Faiss 索引及其元数据的访问封装。

    索引采用懒加载，导入模块时不要求本地索引已经存在。
    """

    def __init__(
        self,
        index_path: Path | None = None,
        *,
        index: FaissIndex | None = None,
        metadata: Mapping[
            int,
            Mapping[str, Any],
        ]
        | None = None,
    ) -> None:
        self._index_path = index_path
        self._index = index
        self._meta = (
            {
                int(key): dict(value)
                for key, value in metadata.items()
            }
            if metadata is not None
            else None
        )

    def _resolve_index_path(self) -> Path:
        if self._index_path is not None:
            return self._index_path.resolve()

        configured_path = os.getenv(
            "ANN_INDEX_PATH",
            "",
        ).strip()

        if not configured_path:
            raise RuntimeError(
                "缺少环境变量：ANN_INDEX_PATH。"
            )

        index_path = Path(configured_path)

        if not index_path.is_absolute():
            index_path = (
                PROJECT_ROOT
                / index_path
            )

        self._index_path = index_path.resolve()
        return self._index_path

    def _ensure_loaded(self) -> None:
        if (
            self._index is not None
            and self._meta is not None
        ):
            return

        index_path = self._resolve_index_path()

        if not index_path.is_file():
            raise FileNotFoundError(
                f"ANN 索引不存在：{index_path}"
            )

        metadata_path = index_path.with_suffix(
            ".meta.json"
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                "ANN 元数据不存在："
                f"{metadata_path}"
            )

        # Faiss 只在真正检索时导入，方便没有本地索引的
        # API 进程完成模块加载和健康检查。
        import faiss

        self._index = faiss.read_index(
            str(index_path)
        )
        self._meta = self._load_meta(
            metadata_path
        )

    def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        platform: str,
    ) -> list[dict[str, Any]]:
        """
        在 ANN 中检索并筛选指定平台的候选商品。
        """

        if not embedding:
            raise ValueError(
                "embedding 不能为空。"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        normalized_platform = platform.strip().lower()

        if not normalized_platform:
            raise ValueError(
                "platform 不能为空字符串。"
            )

        self._ensure_loaded()

        if self._index is None or self._meta is None:
            raise RuntimeError(
                "ANN 索引初始化失败。"
            )

        vector = np.asarray(
            [embedding],
            dtype=np.float32,
        )

        if vector.ndim != 2:
            raise ValueError(
                "embedding 必须是一维向量。"
            )

        scores, indexes = self._index.search(
            vector,
            top_k * 3,
        )

        results: list[dict[str, Any]] = []

        for score, index in zip(
            scores[0],
            indexes[0],
        ):
            if int(index) < 0:
                continue

            metadata = self._meta.get(
                int(index)
            )

            if (
                metadata
                and str(
                    metadata.get(
                        "platform",
                        "",
                    )
                ).lower()
                == normalized_platform
            ):
                results.append(
                    {
                        **metadata,
                        "score": float(score),
                    }
                )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _load_meta(
        path: Path,
    ) -> dict[int, dict[str, Any]]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            raw_metadata = json.load(
                metadata_file
            )

        if not isinstance(raw_metadata, dict):
            raise ValueError(
                "ANN 元数据顶层必须是对象。"
            )

        metadata: dict[int, dict[str, Any]] = {}

        for raw_index, raw_item in (
            raw_metadata.items()
        ):
            if not isinstance(raw_item, dict):
                raise ValueError(
                    "ANN 商品元数据必须是对象。"
                )

            metadata[int(raw_index)] = raw_item

        return metadata


# 全项目共享懒加载 ANN 客户端。
ann_client = AnnClient()


__all__ = [
    "AnnClient",
    "ann_client",
]
