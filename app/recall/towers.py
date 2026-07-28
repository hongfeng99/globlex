from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx
from dotenv import load_dotenv

from app.utils.path_utils import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


class TowerClient:
    """
    LLM 三塔向量召回服务的异步客户端。

    User 塔和 Query 塔用于在线召回，Item 塔主要用于
    离线构建或增量更新 ANN 商品索引。
    """

    def __init__(
        self,
        *,
        user_endpoint: str | None = None,
        query_endpoint: str | None = None,
        item_endpoint: str | None = None,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError(
                "timeout 必须大于 0。"
            )

        self.user_endpoint = user_endpoint
        self.query_endpoint = query_endpoint
        self.item_endpoint = item_endpoint
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_endpoint(
        self,
        configured_endpoint: str | None,
        environment_name: str,
    ) -> str:
        endpoint = (
            configured_endpoint
            or os.getenv(environment_name, "")
        ).strip()

        if not endpoint:
            raise RuntimeError(
                "缺少召回服务环境变量："
                f"{environment_name}。"
            )

        return endpoint

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout
            )

        return self._client

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
            raise ValueError(
                "召回服务响应必须是 JSON 对象。"
            )

        embedding = body.get("embedding")

        if (
            not isinstance(embedding, list)
            or not embedding
            or any(
                not isinstance(value, int | float)
                for value in embedding
            )
        ):
            raise ValueError(
                "召回服务响应缺少有效 embedding。"
            )

        return [
            float(value)
            for value in embedding
        ]

    async def encode_user(
        self,
        user_id: str,
    ) -> list[float]:
        """
        调用 User 塔，把用户标识编码成向量。
        """

        normalized_user_id = user_id.strip()

        if not normalized_user_id:
            raise ValueError(
                "user_id 不能为空字符串。"
            )

        return await self._encode(
            endpoint=self._get_endpoint(
                self.user_endpoint,
                "TOWER_USER_ENDPOINT",
            ),
            payload={
                "user_id": normalized_user_id,
            },
        )

    async def encode_query(
        self,
        query: str,
    ) -> list[float]:
        """
        调用 Query 塔，把检索词编码成向量。
        """

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query 不能为空字符串。"
            )

        return await self._encode(
            endpoint=self._get_endpoint(
                self.query_endpoint,
                "TOWER_QUERY_ENDPOINT",
            ),
            payload={
                "query": normalized_query,
            },
        )

    async def encode_item(
        self,
        item: Mapping[str, Any],
    ) -> list[float]:
        """
        调用 Item 塔，把商品结构编码成向量。
        """

        if not item:
            raise ValueError(
                "item 不能为空。"
            )

        return await self._encode(
            endpoint=self._get_endpoint(
                self.item_endpoint,
                "TOWER_ITEM_ENDPOINT",
            ),
            payload={
                "item": dict(item),
            },
        )

    async def aclose(self) -> None:
        """
        关闭由客户端自身创建的 HTTP 连接池。
        """

        if (
            self._owns_client
            and self._client is not None
        ):
            await self._client.aclose()
            self._client = None


# 全项目复用连接池；真正的网络客户端在首次调用时创建。
tower_client = TowerClient()


__all__ = [
    "TowerClient",
    "tower_client",
]
