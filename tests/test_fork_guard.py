import pytest

from app.agent.fork_guard import (
    ForkLimitExceeded,
    enter_fork,
    get_fork_depth,
)


def test_fork_guard_resets_depth():
    with enter_fork(max_depth=2):
        assert get_fork_depth() == 1
        with enter_fork(max_depth=2):
            assert get_fork_depth() == 2
            with pytest.raises(
                ForkLimitExceeded
            ):
                with enter_fork(max_depth=2):
                    pass
    assert get_fork_depth() == 0
