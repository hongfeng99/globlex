from __future__ import annotations

from app.agent.llm import get_judge_llm
from app.evolution.prompt_versions import (
    prompt_store,
)


ANALYZE_PROMPT = """
你是 Globex 的 Prompt 优化器。
当前 system prompt:
{current_prompt}

最近 bad case:
{bad_cases_summary}

仅输出：共同根因、建议修改文本、预期效果。
不得修改工具名、fork 三件事规则和安全红线。
"""


async def suggest_prompt_improvement(
    bad_cases: list[dict],
) -> str:
    current = prompt_store.get_active()
    summary = "\n".join(
        f"- {case.get('query', '')[:50]}："
        f"{case.get('rubric_comment', '')}"
        for case in bad_cases[:5]
    )
    response = await get_judge_llm().ainvoke(
        ANALYZE_PROMPT.format(
            current_prompt=current.content,
            bad_cases_summary=summary,
        )
    )
    return (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
