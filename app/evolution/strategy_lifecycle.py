from __future__ import annotations

import math
from datetime import UTC, datetime

from app.memory.strategy import StrategyEntry


def compute_confidence(
    strategy: StrategyEntry,
) -> float:
    now = datetime.now(UTC)
    created = strategy.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    days_old = (now - created).days
    time_decay = math.exp(
        -0.693 * days_old / 60
    )
    reference_boost = min(
        strategy.times_referenced * 0.05,
        0.5,
    )
    confidence = min(
        1.0,
        strategy.rubric_score
        * time_decay
        + reference_boost,
    )
    return round(confidence, 3)
