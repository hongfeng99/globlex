from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from dotenv import load_dotenv

from app.utils.path_utils import PROJECT_ROOT


# 本地开发时从项目根目录加载配置。
# override=False（默认值）保证部署环境显式注入的变量优先。
load_dotenv(PROJECT_ROOT / ".env")


def _required_env(name: str) -> str:
    """
    读取必需的模型环境变量。

    配置缺失时在模型初始化阶段直接报出明确错误，
    避免把空值带到远端接口后才得到难以定位的异常。
    """

    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"缺少必需的环境变量：{name}。"
        )

    return value


def _timeout_seconds() -> float:
    """
    读取统一的模型请求超时秒数。
    """

    raw_timeout = os.getenv(
        "LLM_TIMEOUT",
        "60",
    ).strip()

    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise RuntimeError(
            "LLM_TIMEOUT 必须是大于 0 的数字。"
        ) from exc

    if timeout <= 0:
        raise RuntimeError(
            "LLM_TIMEOUT 必须是大于 0 的数字。"
        )

    return timeout


def _create_model(
    *,
    model_name: str,
    temperature: float,
) -> BaseChatModel:
    """
    通过 OpenAI 兼容协议创建统一的聊天模型。
    """

    return init_chat_model(
        model_name,
        model_provider="openai",
        api_key=_required_env("OPENAI_API_KEY"),
        base_url=_required_env("OPENAI_BASE_URL"),
        temperature=temperature,
        timeout=_timeout_seconds(),
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    返回主 / 子 AgentLoop 共用的大模型实例。

    lru_cache 保证同一进程只初始化一次客户端，
    fork 出的同质子 AgentLoop 会复用相同模型配置。
    """

    return _create_model(
        model_name=_required_env("LLM_MAIN"),
        temperature=0.3,
    )


@lru_cache(maxsize=1)
def get_judge_llm() -> BaseChatModel:
    """
    返回 Rubric judge 专用的稳定评审模型。
    """

    judge_model = (
        os.getenv("LLM_JUDGE", "").strip()
        or _required_env("LLM_MAIN")
    )

    return _create_model(
        model_name=judge_model,
        temperature=0.0,
    )


@lru_cache(maxsize=1)
def get_lite_llm() -> BaseChatModel:
    return _create_model(
        model_name=(
            os.getenv("LLM_LITE", "").strip()
            or _required_env("LLM_MAIN")
        ),
        temperature=0.2,
    )


@lru_cache(maxsize=1)
def get_minimal_llm() -> BaseChatModel:
    return _create_model(
        model_name=(
            os.getenv("LLM_MINIMAL", "").strip()
            or os.getenv("LLM_LITE", "").strip()
            or _required_env("LLM_MAIN")
        ),
        temperature=0.1,
    )


def clear_llm_cache() -> None:
    """
    清空模型单例缓存。

    主要用于测试，以及进程内主动重新加载配置的场景。
    """

    get_llm.cache_clear()
    get_judge_llm.cache_clear()
    get_lite_llm.cache_clear()
    get_minimal_llm.cache_clear()


__all__ = [
    "clear_llm_cache",
    "get_judge_llm",
    "get_lite_llm",
    "get_llm",
    "get_minimal_llm",
]
