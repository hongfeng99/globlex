from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class PromptVersion:
    version: str
    content: str
    changelog: str
    author: str
    created_at: datetime = field(
        default_factory=datetime.now
    )
    rubric_score: float | None = None
    status: str = "draft"


class PromptVersionStore:
    def __init__(
        self,
        store_path: Path = Path(
            "data/prompt_versions"
        ),
    ) -> None:
        self._path = store_path
        self._path.mkdir(
            parents=True, exist_ok=True
        )

    def save(
        self, version: PromptVersion
    ) -> None:
        payload = asdict(version)
        payload["created_at"] = (
            version.created_at.isoformat()
        )
        (
            self._path
            / f"{version.version}.yml"
        ).write_text(
            yaml.safe_dump(
                payload,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def get(
        self, version: str
    ) -> PromptVersion:
        payload = yaml.safe_load(
            (
                self._path / f"{version}.yml"
            ).read_text(encoding="utf-8")
        )
        payload["created_at"] = (
            datetime.fromisoformat(
                payload["created_at"]
            )
        )
        return PromptVersion(**payload)

    def all(self) -> list[PromptVersion]:
        return [
            self.get(path.stem)
            for path in sorted(
                self._path.glob("*.yml")
            )
        ]

    def get_active(self) -> PromptVersion:
        active = [
            item
            for item in self.all()
            if item.status == "active"
        ]
        if not active:
            raise LookupError(
                "没有 active prompt 版本"
            )
        return active[-1]


prompt_store = PromptVersionStore()


__all__ = [
    "PromptVersion",
    "PromptVersionStore",
    "prompt_store",
]
