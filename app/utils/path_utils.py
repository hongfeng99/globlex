from __future__ import annotations

import re
from pathlib import Path

from app.api.context import require_session_dir


# 当前文件：
#
# globex-agent/app/utils/path_utils.py
#
# parents[0] = app/utils
# parents[1] = app
# parents[2] = globex-agent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = PROJECT_ROOT / "output"
UPLOADED_ROOT = PROJECT_ROOT / "uploaded"


# thread_id 最终会成为目录名称。
#
# 为避免路径分隔符和特殊路径内容，
# 当前只允许：
#
# 1. 英文字母；
# 2. 数字；
# 3. 下划线；
# 4. 连字符。
_THREAD_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]+$"
)


def normalize_thread_id(
    thread_id: str,
) -> str:
    """
    清理并校验 thread_id。

    thread_id 后续会直接用作目录名称，
    因此不能包含斜杠、反斜杠、空格等字符。
    """

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise ValueError(
            "thread_id 不能为空字符串。"
        )

    if not _THREAD_ID_PATTERN.fullmatch(
        normalized_thread_id
    ):
        raise ValueError(
            "thread_id 只能包含英文字母、数字、"
            "下划线和连字符。"
        )

    return normalized_thread_id


def get_project_root() -> Path:
    """
    返回 Globex 项目的根目录。
    """

    return PROJECT_ROOT


def get_output_root(
    *,
    create: bool = True,
) -> Path:
    """
    返回项目统一的输出根目录。

    create=True 时，如果目录不存在，
    会自动创建。
    """

    if create:
        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    return OUTPUT_ROOT


def get_uploaded_root(
    *,
    create: bool = True,
) -> Path:
    """
    返回项目统一的上传文件根目录。

    create=True 时，如果目录不存在，
    会自动创建。
    """

    if create:
        UPLOADED_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    return UPLOADED_ROOT


def create_session_dir(
    thread_id: str,
) -> Path:
    """
    创建并返回当前任务的会话输出目录。

    例如：

        thread_id = "task-001"

    返回：

        globex-agent/output/task-001
    """

    normalized_thread_id = normalize_thread_id(
        thread_id
    )

    session_dir = (
        get_output_root()
        / normalized_thread_id
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return session_dir.resolve()


def create_uploaded_dir(
    thread_id: str,
) -> Path:
    """
    创建并返回当前任务的上传目录。

    例如：

        thread_id = "task-001"

    返回：

        globex-agent/uploaded/task-001
    """

    normalized_thread_id = normalize_thread_id(
        thread_id
    )

    uploaded_dir = (
        get_uploaded_root()
        / normalized_thread_id
    )

    uploaded_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return uploaded_dir.resolve()


def _resolve_under_directory(
    base_directory: Path,
    relative_path: str | Path,
) -> Path:
    """
    在指定基础目录下安全解析相对路径。

    如果最终路径逃出了 base_directory，
    就主动拒绝。

    例如下面这种路径会被拒绝：

        ../outside.txt
        ../../secret.txt
        C:\\other\\file.txt
    """

    base_directory = base_directory.resolve()

    candidate_path = (
        base_directory
        / Path(relative_path)
    ).resolve()

    try:
        candidate_path.relative_to(
            base_directory
        )
    except ValueError as exc:
        raise ValueError(
            "目标路径不能超出允许的目录范围。"
        ) from exc

    return candidate_path


def resolve_output_path(
    relative_path: str | Path,
    *,
    create_parent: bool = False,
) -> Path:
    """
    在当前请求的 session_dir 中解析输出路径。

    当前 session_dir 来自 ContextVar，
    因此调用者不需要再次传入 thread_id。

    示例：

        resolve_output_path(
            "reports/final.md",
            create_parent=True,
        )

    返回：

        当前 session_dir/reports/final.md
    """

    session_dir = require_session_dir()

    output_path = _resolve_under_directory(
        base_directory=session_dir,
        relative_path=relative_path,
    )

    if create_parent:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    return output_path


def resolve_uploaded_path(
    thread_id: str,
    relative_path: str | Path,
    *,
    create_parent: bool = False,
) -> Path:
    """
    在指定任务的 uploaded 目录中解析路径。

    上传接口通常先取得 thread_id，
    再将用户上传文件保存到对应目录。
    """

    uploaded_dir = create_uploaded_dir(
        thread_id
    )

    uploaded_path = _resolve_under_directory(
        base_directory=uploaded_dir,
        relative_path=relative_path,
    )

    if create_parent:
        uploaded_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    return uploaded_path