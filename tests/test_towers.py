from typing import Any

import httpx
import pytest
import math

from app.recall.towers import TowerClient
from app.recall.local_embeddings import (
    embedding_dimension,
)


@pytest.mark.asyncio
async def test_local_tower_needs_no_http_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOWER_BACKEND", "local")
    client = TowerClient()

    query_vector = await client.encode_query(
        "旅行收纳袋"
    )
    item_vector = await client.encode_item(
        {
            "title": "防水旅行收纳袋",
            "category": "旅行收纳",
        }
    )
    user_vector = await client.encode_user("user-1")
    preference_vector = await client.encode_user(
        "user-1",
        ["偏好轻量、防水的商品"],
    )

    assert len(query_vector) == embedding_dimension()
    assert len(item_vector) == embedding_dimension()
    assert user_vector == [
        0.0
    ] * embedding_dimension()
    assert math.sqrt(
        sum(value * value for value in preference_vector)
    ) == pytest.approx(1.0)
    assert preference_vector != user_vector


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
    assert await client.encode_user(
        "user-2",
        ["偏好静音键盘"],
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
            "https://tower/user",
            {
                "user_id": "user-2",
                "preferences": ["偏好静音键盘"],
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
