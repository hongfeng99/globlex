from __future__ import annotations

import os

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


@tool
async def web_search(
    query: str,
    max_results: int = 5,
) -> list[WebSearchResult]:
    """检索测评、推荐与价格趋势等外部资料。"""

    api_key = os.getenv(
        "TAVILY_API_KEY", ""
    ).strip()
    if not api_key:
        # Web 搜索是增强能力，不应因为未配置可选 API 而阻断
        # 本地购物主链路。
        return []

    max_results = max(1, min(max_results, 10))
    async with httpx.AsyncClient(
        timeout=20
    ) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
            },
        )
        response.raise_for_status()

    return [
        WebSearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        )
        for item in response.json().get(
            "results", []
        )
    ]


__all__ = ["WebSearchResult", "web_search"]
