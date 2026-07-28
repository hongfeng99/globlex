from app.agent.prompts import (
    get_system_prompt,
)


def test_prompt_contains_item_search_fork_rules() -> None:
    prompt = get_system_prompt()

    assert "2 个及以上平台" in prompt
    assert "同一轮" in prompt
    assert "只要求搜索 1 个平台" in prompt
    assert "直接调用 item_search" in prompt
    assert "不要调用 dispatch_tool" in prompt