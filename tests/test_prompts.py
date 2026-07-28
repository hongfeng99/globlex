from pathlib import Path

import pytest

import app.agent.prompts as prompts_module
from app.agent.prompts import (
    clear_prompt_cache,
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


@pytest.fixture(autouse=True)
def reset_prompt_cache() -> None:
    """
    防止不同提示词测试共享缓存。
    """

    clear_prompt_cache()

    yield

    clear_prompt_cache()


def test_invalid_yaml_root_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    YAML 顶层不是映射时应报错。
    """

    prompts_file = tmp_path / "prompts.yml"

    prompts_file.write_text(
        "- prompt-a\n- prompt-b\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prompts_module,
        "PROMPTS_PATH",
        prompts_file,
    )

    with pytest.raises(
        ValueError,
        match="顶层必须是映射",
    ):
        get_system_prompt()


def test_missing_prompt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    缺少必需提示词时应报错。
    """

    prompts_file = tmp_path / "prompts.yml"

    prompts_file.write_text(
        """
system_prompt: |
  用户偏好：
  {long_term_preferences}

planner_prompt: |
  Planner
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prompts_module,
        "PROMPTS_PATH",
        prompts_file,
    )

    with pytest.raises(
        ValueError,
        match="shopping_summary_prompt",
    ):
        get_system_prompt()


def test_non_string_prompt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    提示词节点不是字符串时应报错。
    """

    prompts_file = tmp_path / "prompts.yml"

    prompts_file.write_text(
        """
system_prompt:
  - invalid

planner_prompt: |
  Planner

shopping_summary_prompt: |
  Summary
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prompts_module,
        "PROMPTS_PATH",
        prompts_file,
    )

    with pytest.raises(
        ValueError,
        match="必须是字符串",
    ):
        get_system_prompt()


def test_prompt_cache_can_be_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    修改 YAML 后，清除缓存应读取新内容。
    """

    prompts_file = tmp_path / "prompts.yml"

    prompts_file.write_text(
        """
system_prompt: |
  第一版：{long_term_preferences}

planner_prompt: |
  Planner

shopping_summary_prompt: |
  Summary
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prompts_module,
        "PROMPTS_PATH",
        prompts_file,
    )

    first_prompt = get_system_prompt(
        "偏好 A"
    )

    prompts_file.write_text(
        """
system_prompt: |
  第二版：{long_term_preferences}

planner_prompt: |
  Planner

shopping_summary_prompt: |
  Summary
""".strip(),
        encoding="utf-8",
    )

    # 未清除缓存时仍然使用第一版。
    assert "第一版" in get_system_prompt(
        "偏好 B"
    )

    clear_prompt_cache()

    second_prompt = get_system_prompt(
        "偏好 B"
    )

    assert "第一版" in first_prompt
    assert "第二版" in second_prompt