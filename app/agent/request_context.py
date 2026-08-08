from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import re
from typing import Any


SUPPORTED_PLATFORMS = (
    "amazon",
    "shopee",
    "aliexpress",
    "ebay",
)

_PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {
    "amazon": ("amazon", "亚马逊"),
    "shopee": ("shopee", "虾皮"),
    "aliexpress": ("aliexpress", "速卖通"),
    "ebay": ("ebay",),
}


@dataclass(slots=True)
class DispatchState:
    platforms: set[str] = field(default_factory=set)
    dispatch_count: int = 0


@dataclass(slots=True)
class CategoryInsightState:
    attempted: bool = False
    effective: bool = False
    confidence: float = 0.0
    degraded_reason: str | None = None


@dataclass(slots=True)
class SearchCandidateState:
    candidates: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    attempt_count: int = 0


_request_var: ContextVar[str] = ContextVar(
    "globex_original_request",
    default="",
)
_dispatch_state_var: ContextVar[DispatchState | None] = ContextVar(
    "globex_dispatch_state",
    default=None,
)
_category_insight_state_var: ContextVar[
    CategoryInsightState | None
] = ContextVar(
    "globex_category_insight_state",
    default=None,
)
_search_candidate_state_var: ContextVar[
    SearchCandidateState | None
] = ContextVar(
    "globex_search_candidate_state",
    default=None,
)


def get_original_request() -> str:
    return _request_var.get()


def platforms_in_text(text: str) -> set[str]:
    normalized = text.casefold()
    return {
        platform
        for platform, aliases in _PLATFORM_ALIASES.items()
        if any(alias.casefold() in normalized for alias in aliases)
    }


def get_category_insight_state() -> CategoryInsightState | None:
    return _category_insight_state_var.get()


def record_category_insight(
    *,
    effective: bool,
    confidence: float = 0.0,
    degraded_reason: str | None = None,
) -> None:
    """Record the root request's category-enrichment outcome."""

    state = _category_insight_state_var.get()
    if state is None:
        return
    state.attempted = True
    state.effective = effective
    state.confidence = confidence
    state.degraded_reason = degraded_reason


def record_search_candidates(
    candidates: Iterable[Any],
) -> None:
    """Collect structured ItemSearch observations across child loops."""

    state = _search_candidate_state_var.get()
    if state is None:
        return
    state.attempt_count += 1
    for candidate in candidates:
        if hasattr(candidate, "model_dump"):
            raw = candidate.model_dump()
        elif isinstance(candidate, Mapping):
            raw = dict(candidate)
        else:
            continue
        item_id = str(raw.get("item_id", "")).strip()
        if item_id:
            state.candidates[item_id] = raw


def get_search_candidates() -> list[dict[str, Any]]:
    state = _search_candidate_state_var.get()
    if state is None:
        return []
    return [dict(item) for item in state.candidates.values()]


def has_search_observations() -> bool:
    state = _search_candidate_state_var.get()
    return state is not None and state.attempt_count > 0


def reserve_dispatch(demands: str) -> tuple[bool, str | None]:
    """Reserve one platform dispatch inside the current root request."""

    state = _dispatch_state_var.get()
    if state is None:
        return True, None

    requested = platforms_in_text(demands)
    duplicates = requested & state.platforms
    if duplicates:
        names = "、".join(sorted(duplicates))
        return False, f"平台 {names} 已经派发，拒绝重复子任务。"
    requested_slots = len(requested) if requested else 1
    if (
        state.dispatch_count + requested_slots
        > len(SUPPORTED_PLATFORMS)
    ):
        return False, "本轮最多派发四个平台子任务。"

    state.platforms.update(requested)
    state.dispatch_count += requested_slots
    return True, None


@contextmanager
def bind_request_context(user_request: str) -> Iterator[None]:
    normalized = re.sub(r"\s+", " ", user_request).strip()
    request_token = _request_var.set(normalized)
    dispatch_token = _dispatch_state_var.set(DispatchState())
    insight_token = _category_insight_state_var.set(
        CategoryInsightState()
    )
    candidates_token = _search_candidate_state_var.set(
        SearchCandidateState()
    )
    try:
        yield
    finally:
        _search_candidate_state_var.reset(candidates_token)
        _category_insight_state_var.reset(insight_token)
        _dispatch_state_var.reset(dispatch_token)
        _request_var.reset(request_token)


__all__ = [
    "SUPPORTED_PLATFORMS",
    "CategoryInsightState",
    "SearchCandidateState",
    "bind_request_context",
    "get_category_insight_state",
    "get_original_request",
    "get_search_candidates",
    "has_search_observations",
    "platforms_in_text",
    "record_category_insight",
    "record_search_candidates",
    "reserve_dispatch",
]
