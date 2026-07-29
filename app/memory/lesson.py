from pydantic import BaseModel


class LessonEntry(BaseModel):
    lesson_id: str
    query_pattern: str
    what_went_wrong: str
    avoid_hints: list[str]
    rubric_comment: str
    confidence: float = 1.0


__all__ = ["LessonEntry"]
