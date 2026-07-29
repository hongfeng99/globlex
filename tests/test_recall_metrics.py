import pytest

from app.eval.recall_metrics import (
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_metrics() -> None:
    retrieved = ["a", "b", "c"]
    relevant = ["b", "c"]

    assert recall_at_k(
        retrieved,
        relevant,
        2,
    ) == 0.5
    assert mrr(
        retrieved,
        relevant,
    ) == 0.5
    assert 0.0 < ndcg_at_k(
        retrieved,
        relevant,
        3,
    ) <= 1.0


def test_empty_relevant_is_zero() -> None:
    assert recall_at_k(
        ["a"],
        [],
        1,
    ) == 0.0
    assert ndcg_at_k(
        ["a"],
        [],
        1,
    ) == 0.0
