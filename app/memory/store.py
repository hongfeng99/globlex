from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from threading import Lock

from pydantic import BaseModel, Field
from app.memory.strategy import StrategyEntry


class PreferenceEntry(BaseModel):
    user_id: str
    preference: str
    source: str = "conversation"
    confidence: float = Field(
        default=0.8, ge=0, le=1
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class PreferenceStore:
    """Small process-local store; chapter 18 adds durable evolution."""

    def __init__(self) -> None:
        self._entries: dict[
            str, list[PreferenceEntry]
        ] = defaultdict(list)
        self._lock = Lock()
        self._strategies: dict[
            str, StrategyEntry
        ] = {}

    async def add(
        self, entry: PreferenceEntry
    ) -> None:
        with self._lock:
            existing = self._entries[entry.user_id]
            if not any(
                item.preference
                == entry.preference
                for item in existing
            ):
                existing.append(entry)

    async def add_many(
        self,
        user_id: str,
        preferences: list[str],
        *,
        source: str = "conversation",
    ) -> None:
        for preference in preferences:
            normalized = preference.strip()
            if normalized:
                await self.add(
                    PreferenceEntry(
                        user_id=user_id,
                        preference=normalized,
                        source=source,
                    )
                )

    async def get(
        self, user_id: str
    ) -> list[PreferenceEntry]:
        with self._lock:
            return list(
                self._entries.get(user_id, [])
            )

    async def read_relevant(
        self,
        user_id: str,
        query: str = "",
        limit: int = 12,
    ) -> list[PreferenceEntry]:
        entries = await self.get(user_id)
        query_tokens = set(query.lower().split())
        entries.sort(
            key=lambda entry: (
                bool(
                    query_tokens
                    & set(
                        entry.preference.lower().split()
                    )
                ),
                entry.confidence,
                entry.created_at,
            ),
            reverse=True,
        )
        return entries[:limit]

    async def render(
        self,
        user_id: str,
        query: str = "",
    ) -> str:
        entries = await self.read_relevant(
            user_id, query
        )
        return "\n".join(
            f"- {entry.preference}"
            for entry in entries
        )

    async def export_json(
        self, user_id: str
    ) -> str:
        entries = await self.get(user_id)
        return json.dumps(
            [
                entry.model_dump(mode="json")
                for entry in entries
            ],
            ensure_ascii=False,
        )

    async def write_strategy(
        self,
        strategy: StrategyEntry,
    ) -> None:
        with self._lock:
            self._strategies[
                strategy.strategy_id
            ] = strategy

    async def get_strategy(
        self,
        strategy_id: str,
    ) -> StrategyEntry | None:
        with self._lock:
            return self._strategies.get(
                strategy_id
            )

    async def read_relevant_strategies(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[StrategyEntry]:
        query_tokens = set(
            query.lower().split()
        )
        with self._lock:
            values = list(
                self._strategies.values()
            )
        values.sort(
            key=lambda strategy: (
                len(
                    query_tokens
                    & set(
                        strategy.query_pattern
                        .lower()
                        .split()
                    )
                ),
                strategy.confidence,
            ),
            reverse=True,
        )
        return values[:top_k]


preference_store = PreferenceStore()


__all__ = [
    "PreferenceEntry",
    "PreferenceStore",
    "preference_store",
]
