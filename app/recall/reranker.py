from __future__ import annotations

import os
from collections.abc import Sequence

import httpx


class RerankerClient:
    """
    Cross-encoder reranker 的异步 HTTP 客户端。
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float = 3.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_endpoint(self) -> str:
        endpoint = (
            self.endpoint
            or os.getenv(
                "RERANKER_ENDPOINT",
                "",
            )
        ).strip()

        if not endpoint:
            raise RuntimeError(
                "缺少环境变量：RERANKER_ENDPOINT。"
            )

        return endpoint

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout
            )

        return self._client

    async def score(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> list[float]:
        response = await self._get_client().post(
            self._get_endpoint(),
            json={
                "query": query,
                "candidates": list(candidates),
            },
        )
        response.raise_for_status()
        scores = response.json().get("scores")

        if (
            not isinstance(scores, list)
            or len(scores) != len(candidates)
        ):
            raise ValueError(
                "reranker 返回的 scores 数量错误。"
            )

        return [
            float(score)
            for score in scores
        ]

    async def aclose(self) -> None:
        if (
            self._owns_client
            and self._client is not None
        ):
            await self._client.aclose()
            self._client = None


reranker_client = RerankerClient()


__all__ = [
    "RerankerClient",
    "reranker_client",
]
