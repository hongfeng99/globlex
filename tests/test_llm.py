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
