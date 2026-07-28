from pathlib import Path

import pytest

import app.utils.path_utils as path_utils
from app.utils.path_utils import (
    create_session_dir,
    create_uploaded_dir,
    get_output_root,
    get_project_root,
    get_uploaded_root,
    normalize_thread_id,
    resolve_output_path,
    resolve_uploaded_path,
)
from app.utils.thread_ctx import (
    bind_thread_context,
)


@pytest.fixture
def temporary_runtime_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """
    将测试中的 output 和 uploaded
    临时切换到 pytest 提供的目录。

    避免单元测试污染项目真实目录。
    """

    output_root = tmp_path / "output"
    uploaded_root = tmp_path / "uploaded"

    monkeypatch.setattr(
        path_utils,
        "OUTPUT_ROOT",
        output_root,
    )

    monkeypatch.setattr(
        path_utils,
        "UPLOADED_ROOT",
        uploaded_root,
    )

    return output_root, uploaded_root


def test_project_root_is_correct() -> None:
    """
    验证项目根目录可以正确定位。
    """

    project_root = get_project_root()

    assert (
        project_root
        / "pyproject.toml"
    ).is_file()

    assert (
        project_root
        / "app"
    ).is_dir()


def test_runtime_roots_can_be_created(
    temporary_runtime_roots: tuple[
        Path,
        Path,
    ],
) -> None:
    """
    验证 output 和 uploaded 根目录
    可以自动创建。
    """

    output_root, uploaded_root = (
        temporary_runtime_roots
    )

    assert not output_root.exists()
    assert not uploaded_root.exists()

    assert get_output_root() == output_root
    assert get_uploaded_root() == uploaded_root

    assert output_root.is_dir()
    assert uploaded_root.is_dir()


def test_thread_directories_can_be_created(
    temporary_runtime_roots: tuple[
        Path,
        Path,
    ],
) -> None:
    """
    验证每个 thread_id 都有独立的
    输出目录和上传目录。
    """

    output_root, uploaded_root = (
        temporary_runtime_roots
    )

    session_dir = create_session_dir(
        "task-001"
    )

    upload_dir = create_uploaded_dir(
        "task-001"
    )

    assert session_dir == (
        output_root
        / "task-001"
    ).resolve()

    assert upload_dir == (
        uploaded_root
        / "task-001"
    ).resolve()

    assert session_dir.is_dir()
    assert upload_dir.is_dir()


def test_resolve_output_path_uses_context(
    temporary_runtime_roots: tuple[
        Path,
        Path,
    ],
) -> None:
    """
    验证输出路径会自动使用当前
    ContextVar 中的 session_dir。
    """

    session_dir = create_session_dir(
        "task-002"
    )

    with bind_thread_context(
        thread_id="task-002",
        session_dir=session_dir,
    ):
        output_file = resolve_output_path(
            "reports/final.md",
            create_parent=True,
        )

    assert output_file == (
        session_dir
        / "reports"
        / "final.md"
    ).resolve()

    assert output_file.parent.is_dir()


def test_resolve_uploaded_path(
    temporary_runtime_roots: tuple[
        Path,
        Path,
    ],
) -> None:
    """
    验证上传文件路径可以正确解析。
    """

    _, uploaded_root = (
        temporary_runtime_roots
    )

    uploaded_file = resolve_uploaded_path(
        thread_id="task-003",
        relative_path="documents/input.pdf",
        create_parent=True,
    )

    assert uploaded_file == (
        uploaded_root
        / "task-003"
        / "documents"
        / "input.pdf"
    ).resolve()

    assert uploaded_file.parent.is_dir()


@pytest.mark.parametrize(
    "invalid_thread_id",
    [
        "",
        "   ",
        "../task",
        "task/001",
        r"task\001",
        "task 001",
    ],
)
def test_invalid_thread_id_is_rejected(
    invalid_thread_id: str,
) -> None:
    """
    验证不安全的 thread_id 会被拒绝。
    """

    with pytest.raises(ValueError):
        normalize_thread_id(
            invalid_thread_id
        )


def test_output_path_cannot_escape_session_dir(
    temporary_runtime_roots: tuple[
        Path,
        Path,
    ],
) -> None:
    """
    验证 ../ 不能逃出当前会话目录。
    """

    session_dir = create_session_dir(
        "task-004"
    )

    with bind_thread_context(
        thread_id="task-004",
        session_dir=session_dir,
    ):
        with pytest.raises(
            ValueError,
            match="不能超出允许的目录范围",
        ):
            resolve_output_path(
                "../outside.txt"
            )


def test_uploaded_path_cannot_escape_directory(
    temporary_runtime_roots: tuple[
        Path,
        Path,
    ],
) -> None:
    """
    验证上传路径不能逃出当前任务的目录。
    """

    with pytest.raises(
        ValueError,
        match="不能超出允许的目录范围",
    ):
        resolve_uploaded_path(
            thread_id="task-005",
            relative_path="../../outside.txt",
        )


def test_output_path_requires_context() -> None:
    """
    当前没有请求上下文时，
    不能解析会话输出路径。
    """

    with pytest.raises(
        RuntimeError,
        match="没有 session_dir",
    ):
        resolve_output_path(
            "result.json"
        )