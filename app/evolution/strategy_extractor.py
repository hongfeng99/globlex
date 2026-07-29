from __future__ import annotations

import json
from uuid import uuid4

from app.agent.llm import get_judge_llm
from app.memory.strategy import StrategyEntry


EXTRACT_PROMPT = """
你是 Globex 的策略提炼器。给定高分购物轨迹，
提炼一条可复用成功策略。仅输出 JSON：
query_pattern, summary, tool_hints, key_decisions。
不要保存具体商品 ID、价格或用户隐私。
"""


async def extract_strategy(
    query: str,
    tool_sequence: list[str],
    score: float,
    trace_id: str,
) -> StrategyEntry | None:
    if score < 0.80 or len(tool_sequence) < 4:
        return None
    response = await get_judge_llm().ainvoke(
        f"{EXTRACT_PROMPT}\nquery={query}\n"
        f"tools={tool_sequence}"
    )
    content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    payload = json.loads(content)
    return StrategyEntry(
        strategy_id=f"strat-{uuid4().hex[:8]}",
        rubric_score=score,
        source_trace_id=trace_id,
        **payload,
    )
