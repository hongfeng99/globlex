import asyncio
from pathlib import Path

import pytest

from app.api.context import (
    get_session_dir,
    get_thread_id,
    require_session_dir,
    require_thread_id,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


def test_context_can_be_bound_and_restored(
    tmp_path: Path,
) -> None:
    """
    验证上下文可以被设置，
    并且离开 with 代码块后能够恢复。
    """

    session_dir = tmp_path / "task-001"

    assert get_thread_id() is None
    assert get_session_dir() is None

    with bind_thread_context(
        thread_id="task-001",
        session_dir=session_dir,
    ):
        assert get_thread_id() == "task-001"
        assert require_thread_id() == "task-001"

        assert get_session_dir() == (
            session_dir.resolve()
        )
        assert require_session_dir() == (
            session_dir.resolve()
        )

    assert get_thread_id() is None
    assert get_session_dir() is None


def test_empty_thread_id_is_rejected(
    tmp_path: Path,
) -> None:
    """
    验证空 thread_id 会被拒绝。
    """

    with pytest.raises(
        ValueError,
        match="thread_id 不能为空",
    ):
        with bind_thread_context(
            thread_id="   ",
            session_dir=tmp_path,
        ):
            pass


@pytest.mark.asyncio
async def test_concurrent_requests_are_isolated(
    tmp_path: Path,
) -> None:
    """
    验证并发执行的两个 asyncio Task
    不会互相覆盖 thread_id 和 session_dir。
    """

    async def worker(
        thread_id: str,
    ) -> tuple[str, Path]:
        session_dir = tmp_path / thread_id

        with bind_thread_context(
            thread_id=thread_id,
            session_dir=session_dir,
        ):
            # 主动暂停当前任务，
            # 让另一个协程有机会执行。
            await asyncio.sleep(0.01)

            return (
                require_thread_id(),
                require_session_dir(),
            )

    result_a, result_b = await asyncio.gather(
        worker("thread-A"),
        worker("thread-B"),
    )

    assert result_a == (
        "thread-A",
        (tmp_path / "thread-A").resolve(),
    )

    assert result_b == (
        "thread-B",
        (tmp_path / "thread-B").resolve(),
    )


@pytest.mark.asyncio
async def test_child_task_inherits_parent_context(
    tmp_path: Path,
) -> None:
    """
    验证由父任务创建的子 asyncio Task
    可以继承父任务当前的 ContextVar。
    """

    async def child_agent() -> tuple[str, Path]:
        await asyncio.sleep(0)

        return (
            require_thread_id(),
            require_session_dir(),
        )

    session_dir = tmp_path / "parent-task"

    with bind_thread_context(
        thread_id="parent-thread",
        session_dir=session_dir,
    ):
        child_task = asyncio.create_task(
            child_agent()
        )

        child_result = await child_task

    assert child_result == (
        "parent-thread",
        session_dir.resolve(),
    )


@pytest.mark.asyncio
async def test_child_context_can_override_thread_id(
    tmp_path: Path,
) -> None:
    """
    验证子 Agent 可以：

    1. 使用独立的 thread_id；
    2. 继续继承主 Agent 的 session_dir；
    3. 结束后恢复主 Agent 上下文。
    """

    parent_session_dir = (
        tmp_path
        / "main-task"
    )

    async def child_agent(
        sub_thread_id: str,
    ) -> tuple[str, Path]:
        """
        模拟 dispatch_tool 对子上下文的覆盖。
        """

        with bind_thread_context(
            thread_id=sub_thread_id,
            session_dir=require_session_dir(),
        ):
            await asyncio.sleep(0)

            return (
                require_thread_id(),
                require_session_dir(),
            )

    with bind_thread_context(
        thread_id="main-thread",
        session_dir=parent_session_dir,
    ):
        child_a, child_b = await asyncio.gather(
            child_agent(
                "sub-thread-A"
            ),
            child_agent(
                "sub-thread-B"
            ),
        )

        assert child_a == (
            "sub-thread-A",
            parent_session_dir.resolve(),
        )

        assert child_b == (
            "sub-thread-B",
            parent_session_dir.resolve(),
        )

        # 子任务结束后，主任务上下文仍然存在。
        assert require_thread_id() == (
            "main-thread"
        )

        assert require_session_dir() == (
            parent_session_dir.resolve()
        )

    # 离开最外层上下文以后，恢复成未设置状态。
    assert get_thread_id() is None
    assert get_session_dir() is None