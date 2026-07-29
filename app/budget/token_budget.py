from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal


ModelTier = Literal[
    "main", "lite", "minimal", "fallback"
]


@dataclass
class TokenBudget:
    total_limit: int = 50_000
    consumed: int = 0
    model_tier: ModelTier = "main"

    @property
    def remaining(self) -> int:
        return max(
            0, self.total_limit - self.consumed
        )

    @property
    def remaining_ratio(self) -> float:
        if self.total_limit <= 0:
            return 0.0
        return self.remaining / self.total_limit

    def consume(self, tokens: int) -> None:
        self.consumed += max(0, tokens)
        self._update_tier()

    def _update_tier(self) -> None:
        ratio = self.remaining_ratio
        if ratio > 0.50:
            self.model_tier = "main"
        elif ratio > 0.20:
            self.model_tier = "lite"
        elif ratio > 0.05:
            self.model_tier = "minimal"
        else:
            self.model_tier = "fallback"


_budget_var: ContextVar[
    TokenBudget | None
] = ContextVar("globex_token_budget", default=None)


def init_budget(
    total_limit: int = 50_000,
) -> TokenBudget:
    budget = TokenBudget(total_limit=total_limit)
    _budget_var.set(budget)
    return budget


def get_budget() -> TokenBudget | None:
    return _budget_var.get()


__all__ = [
    "ModelTier",
    "TokenBudget",
    "get_budget",
    "init_budget",
]
