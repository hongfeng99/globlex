from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class StrategyEntry(BaseModel):
    strategy_id: str
    query_pattern: str
    summary: str
    tool_hints: list[str] = Field(
        default_factory=list
    )
    key_decisions: list[str] = Field(
        default_factory=list
    )
    rubric_score: float = Field(
        ge=0, le=1
    )
    confidence: float = Field(
        default=1.0, ge=0, le=1
    )
    times_referenced: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    source_trace_id: str | None = None


__all__ = ["StrategyEntry"]
