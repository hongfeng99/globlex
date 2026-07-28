from app.agent.prompts import (
    get_planner_prompt,
    get_shopping_summary_prompt,
    get_system_prompt,
)


def test_system_prompt_injects_preferences() -> None:
    prompt = get_system_prompt(
        "不买塑料材质；偏好小众品牌。"
    )

    assert (
        "不买塑料材质；偏好小众品牌。"
        in prompt
    )
    assert "{long_term_preferences}" not in (
        prompt
    )
    assert "Think → Act → Observe → Reflect" in (
        prompt
    )
    assert "dispatch_tool" in prompt


def test_system_prompt_handles_empty_preferences() -> None:
    prompt = get_system_prompt()

    assert "（暂无长期偏好）" in prompt


def test_tool_prompts_are_available() -> None:
    assert "budget" in get_planner_prompt()
    assert (
        "最多 3 件商品"
        in get_shopping_summary_prompt()
    )
