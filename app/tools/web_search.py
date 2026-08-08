from __future__ import annotations

import os
import time

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel

from app.api.monitor import monitor


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

    await monitor.report_tool_start(
        "web_search",
        {
            "query": query,
            "max_results": max_results,
        },
    )
    started_at = time.perf_counter()

    api_key = os.getenv(
        "TAVILY_API_KEY", ""
    ).strip()
    if not api_key:
        # Web 搜索是增强能力，不应因为未配置可选 API 而阻断
        # 本地购物主链路。
        await monitor.report_tool_end(
            "web_search",
            int(
                (time.perf_counter() - started_at)
                * 1000
            ),
        )
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

    results = [
        WebSearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        )
        for item in response.json().get(
            "results", []
        )
    ]
    await monitor.report_tool_end(
        "web_search",
        int(
            (time.perf_counter() - started_at)
            * 1000
        ),
    )
    return results


__all__ = ["WebSearchResult", "web_search"]
