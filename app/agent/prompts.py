from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROMPTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompt"
    / "prompts.yml"
)

_REQUIRED_PROMPTS = {
    "system_prompt",
    "planner_prompt",
    "shopping_summary_prompt",
}


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, str]:
    """
    从 YAML 读取并校验项目提示词。
    """

    with PROMPTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as prompt_file:
        raw_prompts: Any = yaml.safe_load(
            prompt_file
        )

    if not isinstance(raw_prompts, dict):
        raise ValueError(
            "prompts.yml 顶层必须是映射。"
        )

    missing_prompts = (
        _REQUIRED_PROMPTS
        - raw_prompts.keys()
    )

    if missing_prompts:
        missing_names = ", ".join(
            sorted(missing_prompts)
        )
        raise ValueError(
            "prompts.yml 缺少提示词："
            f"{missing_names}。"
        )

    prompts: dict[str, str] = {}

    for name, value in raw_prompts.items():

        if not isinstance(value, str):
            raise ValueError(
                f"提示词 {name} 必须是字符串。"
            )

        prompts[name] = value

    return prompts


def get_system_prompt(
    long_term_preferences: str = "",
) -> str:
    """
    返回主 / 子 AgentLoop 共用的 system prompt，
    并把长期偏好注入预留位置。
    """

    preferences = (
        long_term_preferences.strip()
        or "（暂无长期偏好）"
    )

    template = _load_prompts()[
        "system_prompt"
    ]

    return template.format(
        long_term_preferences=preferences
    )


def get_planner_prompt() -> str:
    """
    返回 Planner 工具提示词。
    """

    return _load_prompts()["planner_prompt"]


def get_shopping_summary_prompt() -> str:
    """
    返回 ShoppingSummary 工具提示词。
    """

    return _load_prompts()[
        "shopping_summary_prompt"
    ]


def get_session_summary_prompt() -> str:
    return _load_prompts()[
        "session_summary_prompt"
    ]


def get_tool_result_compress_prompt() -> str:
    return _load_prompts()[
        "tool_result_compress_prompt"
    ]


def get_rubric_judge_prompt() -> str:
    return _load_prompts()[
        "rubric_judge_prompt"
    ]


def clear_prompt_cache() -> None:
    """
    清空 YAML 读取缓存。
    """

    _load_prompts.cache_clear()


__all__ = [
    "clear_prompt_cache",
    "get_planner_prompt",
    "get_rubric_judge_prompt",
    "get_session_summary_prompt",
    "get_shopping_summary_prompt",
    "get_system_prompt",
    "get_tool_result_compress_prompt",
]
