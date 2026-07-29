from app.memory.store import preference_store


async def feedback_on_strategy(
    strategy_id: str,
    rubric_score: float,
) -> None:
    strategy = await preference_store.get_strategy(
        strategy_id
    )
    if strategy is None:
        return
    if rubric_score >= 0.75:
        strategy.times_referenced += 1
    elif rubric_score < 0.50:
        strategy.confidence = max(
            0.0, strategy.confidence - 0.1
        )
    await preference_store.write_strategy(
        strategy
    )
