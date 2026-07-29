from __future__ import annotations

import json

from app.agent.llm import get_judge_llm


EXTRACT_PROMPT = """
你是 Globex 品类知识库的卡片抽取器。
输入一段关于品类 {category} 的原始资料。
只输出 JSON：summary, raw_evidence, confidence。
summary 根据 card_type={card_type}：
bestseller 写“商品名|价格|流行原因”，attribute 写属性分布，
price_range 写“低价款 60-150 / 中档 150-400 / 高端 400+”。
raw_evidence 取 1-3 条原始短句；confidence 为 0-1。
"""


async def extract_card(
    category: str,
    raw_text: str,
    card_type: str,
) -> dict:
    response = await get_judge_llm().ainvoke(
        [
            (
                "system",
                EXTRACT_PROMPT.format(
                    category=category,
                    card_type=card_type,
                ),
            ),
            ("user", raw_text),
        ]
    )
    content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    return json.loads(content)
