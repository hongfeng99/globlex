from __future__ import annotations

import hashlib

from app.evolution.prompt_versions import (
    PromptVersion,
    prompt_store,
)


AB_TEST_RATIO = 0.10
_testing_version: PromptVersion | None = None


def set_testing_version(
    version: PromptVersion | None,
) -> None:
    global _testing_version
    _testing_version = version


def get_prompt_for_user(
    user_id: str,
) -> str:
    active = prompt_store.get_active()
    if (
        _testing_version is None
        or _testing_version.status != "testing"
    ):
        return active.content
    hash_value = int(
        hashlib.md5(
            user_id.encode(),
            usedforsecurity=False,
        ).hexdigest(),
        16,
    )
    if (hash_value % 100) / 100 < AB_TEST_RATIO:
        return _testing_version.content
    return active.content
