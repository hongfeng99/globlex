from typing import Any

import httpx
import pytest

from app.recall.towers import TowerClient


@pytest.mark.asyncio
async def test_tower_client_encodes_all_towers() -> None:
    requests: list[
        tuple[str, dict[str, Any]]
    ] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = __import__("json").loads(
            request.content
        )
        requests.append(
            (str(request.url), payload)
        )
        return httpx.Response(
            200,
            json={
                "embedding": [0.1, 0.2],
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        )
    )
    client = TowerClient(
        user_endpoint="https://tower/user",
        query_endpoint="https://tower/query",
        item_endpoint="https://tower/item",
        client=http_client,
    )

    assert await client.encode_user(
        "user-1"
    ) == [0.1, 0.2]
    assert await client.encode_query(
        "旅行收纳袋"
    ) == [0.1, 0.2]
    assert await client.encode_item(
        {
            "item_id": "item-1",
        }
    ) == [0.1, 0.2]

    assert requests == [
        (
            "https://tower/user",
            {
                "user_id": "user-1",
            },
        ),
        (
            "https://tower/query",
            {
                "query": "旅行收纳袋",
            },
        ),
        (
            "https://tower/item",
            {
                "item": {
                    "item_id": "item-1",
                },
            },
        ),
    ]

    await http_client.aclose()


@pytest.mark.asyncio
async def test_invalid_embedding_response() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "unexpected": [],
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        )
    )
    client = TowerClient(
        query_endpoint="https://tower/query",
        client=http_client,
    )

    with pytest.raises(
        ValueError,
        match="embedding",
    ):
        await client.encode_query("query")

    await http_client.aclose()
