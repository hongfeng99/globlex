from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from dotenv import load_dotenv

from app.recall.embedding_backend import (
    get_embedding_backend,
    resolve_embedding_backend_name,
)
from app.utils.path_utils import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


class TowerClient:
    """User、Query、Item 三类向量的统一异步客户端。"""

    def __init__(
        self,
        *,
        user_endpoint: str | None = None,
        query_endpoint: str | None = None,
        item_endpoint: str | None = None,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
        backend_name: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0。")

        self.user_endpoint = user_endpoint
        self.query_endpoint = query_endpoint
        self.item_endpoint = item_endpoint
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._backend_name = backend_name

    def _get_endpoint(
        self,
        configured_endpoint: str | None,
        environment_name: str,
    ) -> str:
        endpoint = (
            configured_endpoint or os.getenv(environment_name, "")
        ).strip()
        if not endpoint:
            raise RuntimeError(
                f"缺少召回服务环境变量：{environment_name}。"
            )
        return endpoint

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _uses_http(self) -> bool:
        if self._client is not None or any(
            endpoint is not None
            for endpoint in (
                self.user_endpoint,
                self.query_endpoint,
                self.item_endpoint,
            )
        ):
            return True
        return resolve_embedding_backend_name(
            self._backend_name
        ) == "http"

    def _local_backend(self) -> Any:
        return get_embedding_backend(self._backend_name)

    async def _encode(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> list[float]:
        response = await self._get_client().post(
            endpoint,
            json=dict(payload),
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("召回服务响应必须是 JSON 对象。")

        embedding = body.get("embedding")
        if (
            not isinstance(embedding, list)
            or not embedding
            or any(
                not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in embedding
            )
        ):
            raise ValueError("召回服务响应缺少有效 embedding。")
        return [float(value) for value in embedding]

    async def encode_user(
        self,
        user_id: str,
        preferences: Sequence[str] | None = None,
    ) -> list[float]:
        """将用户的自然语言偏好编码；user_id 只用于身份关联。"""

        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            raise ValueError("user_id 不能为空字符串。")
        normalized_preferences = [
            value.strip()
            for value in (preferences or [])
            if value.strip()
        ]

        if not self._uses_http():
            backend = self._local_backend()
            if not normalized_preferences:
                return [0.0] * backend.spec.dimension
            return await asyncio.to_thread(
                backend.encode_preferences,
                normalized_preferences,
            )

        payload: dict[str, Any] = {
            "user_id": normalized_user_id,
        }
        if normalized_preferences:
            payload["preferences"] = normalized_preferences
        return await self._encode(
            endpoint=self._get_endpoint(
                self.user_endpoint,
                "TOWER_USER_ENDPOINT",
            ),
            payload=payload,
        )

    async def encode_query(self, query: str) -> list[float]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query 不能为空字符串。")
        if not self._uses_http():
            backend = self._local_backend()
            return await asyncio.to_thread(
                backend.encode_query,
                normalized_query,
            )
        return await self._encode(
            endpoint=self._get_endpoint(
                self.query_endpoint,
                "TOWER_QUERY_ENDPOINT",
            ),
            payload={"query": normalized_query},
        )

    async def encode_item(
        self,
        item: Mapping[str, Any],
    ) -> list[float]:
        if not item:
            raise ValueError("item 不能为空。")
        if not self._uses_http():
            backend = self._local_backend()
            return await asyncio.to_thread(
                backend.encode_item,
                item,
            )
        return await self._encode(
            endpoint=self._get_endpoint(
                self.item_endpoint,
                "TOWER_ITEM_ENDPOINT",
            ),
            payload={"item": dict(item)},
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


tower_client = TowerClient()


__all__ = ["TowerClient", "tower_client"]
