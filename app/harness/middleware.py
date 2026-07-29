from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)
HookFn = Callable[
    [dict[str, Any]],
    Awaitable[dict[str, Any] | None],
]
HOOK_POINTS = {
    "on_session_start",
    "pre_think",
    "pre_tool_call",
    "post_tool_call",
    "post_reflect",
    "on_session_end",
}


class HookRejectSignal(RuntimeError):
    pass


@dataclass(order=True)
class HookRegistration:
    priority: int
    name: str
    callback: HookFn


class HarnessMiddleware:
    def __init__(self) -> None:
        self._hooks: dict[
            str, list[HookRegistration]
        ] = defaultdict(list)

    def register(
        self,
        hook_point: str,
        name: str,
        callback: HookFn,
        priority: int = 100,
    ) -> None:
        if hook_point not in HOOK_POINTS:
            raise ValueError(
                f"未知 Hook 点：{hook_point}"
            )
        registrations = self._hooks[hook_point]
        registrations[:] = [
            item
            for item in registrations
            if item.name != name
        ]
        registrations.append(
            HookRegistration(
                priority, name, callback
            )
        )
        registrations.sort()

    async def run(
        self,
        hook_point: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        for registration in self._hooks.get(
            hook_point, []
        ):
            started = time.perf_counter()
            try:
                update = await registration.callback(
                    context
                )
                if update:
                    context.update(update)
            except HookRejectSignal:
                raise
            except Exception:
                logger.exception(
                    "Hook %s failed",
                    registration.name,
                )
            finally:
                duration_ms = int(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000
                )
                if duration_ms >= 50:
                    logger.info(
                        "Hook %s took %dms",
                        registration.name,
                        duration_ms,
                    )
        return context


harness = HarnessMiddleware()


def harness_hook(
    hook_point: str,
    *,
    name: str,
    priority: int = 100,
):
    def decorator(callback: HookFn) -> HookFn:
        harness.register(
            hook_point,
            name,
            callback,
            priority,
        )
        return callback

    return decorator


__all__ = [
    "HOOK_POINTS",
    "HarnessMiddleware",
    "HookRejectSignal",
    "harness",
    "harness_hook",
]
