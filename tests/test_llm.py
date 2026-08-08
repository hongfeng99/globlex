from typing import Any

import pytest

import app.agent.llm as llm_module


@pytest.fixture(autouse=True)
def reset_model_cache() -> None:
    llm_module.clear_llm_cache()
    yield
    llm_module.clear_llm_cache()


def test_main_and_child_share_cached_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models: list[
        tuple[str, dict[str, Any]]
    ] = []
    fake_model = object()

    def fake_init_chat_model(
        model_name: str,
        **kwargs: Any,
    ) -> object:
        created_models.append(
            (model_name, kwargs)
        )
        return fake_model

    monkeypatch.setattr(
        llm_module,
        "init_chat_model",
        fake_init_chat_model,
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv(
        "LLM_MAIN",
        "main-model",
    )
    monkeypatch.setenv(
        "LLM_TIMEOUT",
        "30",
    )

    assert llm_module.get_llm() is fake_model
    assert llm_module.get_llm() is fake_model

    assert created_models == [
        (
            "main-model",
            {
                "model_provider": "openai",
                "api_key": "test-key",
                "base_url": (
                    "https://example.test/v1"
                ),
                "temperature": 0.3,
                "timeout": 30.0,
                "max_retries": 2,
            },
        )
    ]


def test_judge_uses_stable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models: list[
        tuple[str, dict[str, Any]]
    ] = []

    def fake_init_chat_model(
        model_name: str,
        **kwargs: Any,
    ) -> object:
        created_models.append(
            (model_name, kwargs)
        )
        return object()

    monkeypatch.setattr(
        llm_module,
        "init_chat_model",
        fake_init_chat_model,
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv(
        "LLM_MAIN",
        "main-model",
    )
    monkeypatch.setenv(
        "LLM_JUDGE",
        "judge-model",
    )

    llm_module.get_judge_llm()

    assert created_models[0][0] == (
        "judge-model"
    )
    assert created_models[0][1][
        "temperature"
    ] == 0.0


def test_qwen_provider_disables_thinking_for_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model_name: str,
        **kwargs: Any,
    ) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        llm_module,
        "init_chat_model",
        fake_init_chat_model,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.cn-beijing.maas.aliyuncs.com/v1",
    )
    monkeypatch.setenv("LLM_MAIN", "Qwen/Qwen3.5-35B-A3B")
    monkeypatch.setenv("LLM_ENABLE_THINKING", "false")

    llm_module.get_llm()

    assert captured["extra_body"] == {
        "enable_thinking": False
    }


def test_missing_required_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "LLM_MAIN",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="LLM_MAIN",
    ):
        llm_module.get_llm()


def test_judge_defaults_to_main_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    未配置 LLM_JUDGE 时，
    Judge 应回退到 LLM_MAIN。
    """

    created_models: list[
        tuple[str, dict[str, Any]]
    ] = []

    def fake_init_chat_model(
        model_name: str,
        **kwargs: Any,
    ) -> object:
        created_models.append(
            (model_name, kwargs)
        )

        return object()

    monkeypatch.setattr(
        llm_module,
        "init_chat_model",
        fake_init_chat_model,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv(
        "LLM_MAIN",
        "shared-model",
    )
    monkeypatch.setenv(
        "LLM_JUDGE",
        "",
    )

    llm_module.get_judge_llm()

    assert created_models[0][0] == (
        "shared-model"
    )


@pytest.mark.parametrize(
    "invalid_timeout",
    [
        "not-a-number",
        "0",
        "-1",
    ],
)
def test_invalid_timeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    invalid_timeout: str,
) -> None:
    """
    非数字、零或负数超时均应被拒绝。
    """

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv(
        "LLM_MAIN",
        "main-model",
    )
    monkeypatch.setenv(
        "LLM_TIMEOUT",
        invalid_timeout,
    )

    with pytest.raises(
        RuntimeError,
        match="LLM_TIMEOUT",
    ):
        llm_module.get_llm()


@pytest.mark.parametrize(
    "missing_name",
    [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ],
)
def test_missing_openai_config_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    """
    API Key 或接口地址缺失时，
    应返回明确的配置错误。
    """

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv(
        "LLM_MAIN",
        "main-model",
    )

    monkeypatch.delenv(
        missing_name,
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match=missing_name,
    ):
        llm_module.get_llm()


def test_clear_llm_cache_recreates_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    清除缓存后应重新创建模型实例。
    """

    created_models: list[object] = []

    def fake_init_chat_model(
        model_name: str,
        **kwargs: Any,
    ) -> object:
        model = object()
        created_models.append(model)
        return model

    monkeypatch.setattr(
        llm_module,
        "init_chat_model",
        fake_init_chat_model,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.test/v1",
    )
    monkeypatch.setenv(
        "LLM_MAIN",
        "main-model",
    )

    first_model = llm_module.get_llm()

    llm_module.clear_llm_cache()

    second_model = llm_module.get_llm()

    assert first_model is not second_model
    assert len(created_models) == 2
