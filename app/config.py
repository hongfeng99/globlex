from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{name} 必须是布尔值（true/false）。"
    )


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} 必须是整数。"
            ) from exc

    if minimum is not None and value < minimum:
        raise RuntimeError(
            f"{name} 必须大于等于 {minimum}。"
        )
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} 必须是数字。"
            ) from exc

    if minimum is not None and value < minimum:
        raise RuntimeError(
            f"{name} 必须大于等于 {minimum}。"
        )
    return value


__all__ = ["env_bool", "env_float", "env_int"]
