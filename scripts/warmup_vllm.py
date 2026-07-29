from __future__ import annotations

import asyncio
import os

import httpx


WARMUP_PROMPTS = [
    "帮我规划旅行三件套",
    "想买咖啡杯",
] * 5


async def warmup() -> None:
    endpoint = os.getenv(
        "OPENAI_BASE_URL",
        "http://localhost:8100/v1",
    )
    async with httpx.AsyncClient(
        timeout=30
    ) as client:
        for index, prompt in enumerate(
            WARMUP_PROMPTS, 1
        ):
            await client.post(
                f"{endpoint}/chat/completions",
                json={
                    "model": os.getenv(
                        "OPENAI_MODEL",
                        "globex-main",
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "max_tokens": 50,
                },
            )
            print(
                f"warmup {index}/"
                f"{len(WARMUP_PROMPTS)}"
            )


if __name__ == "__main__":
    asyncio.run(warmup())
