from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path


# 当前请求对应的 thread_id。
#
# 每个 asyncio Task 都会拥有独立的 ContextVar 上下文，
# 因而不同用户的请求不会因为并发执行而互相覆盖。
_thread_id_var: ContextVar[str | None] = ContextVar(
    "globex_thread_id",
    default=None,
)


# 当前请求对应的会话目录。
#
# 该目录后续用于保存：
#
# 1. Agent 输出文件；
# 2. 中间结果；
# 3. 上传文件的处理结果；
# 4. 当前会话相关的数据。
_session_dir_var: ContextVar[Path | None] = ContextVar(
    "globex_session_dir",
    default=None,
)


@dataclass(frozen=True, slots=True)
class ThreadContextTokens:
    """
    保存 ContextVar.set() 返回的 Token。

    Token 记录了设置变量之前的旧值，
    请求结束时可以利用 Token 恢复原来的上下文。
    """

    thread_id_token: Token[str | None]
    session_dir_token: Token[Path | None]


def set_thread_context(
    thread_id: str,
    session_dir: Path,
) -> ThreadContextTokens:
    """
    设置当前请求的 thread_id 和 session_dir。

    参数：
        thread_id:
            当前任务或会话的唯一标识。

        session_dir:
            当前任务对应的会话目录。

    返回：
        两个 ContextVar 对应的 Token，
        后续可以用来恢复旧上下文。
    """

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise ValueError(
            "thread_id 不能为空字符串。"
        )

    normalized_session_dir = Path(
        session_dir
    ).resolve()

    thread_id_token = _thread_id_var.set(
        normalized_thread_id
    )

    session_dir_token = _session_dir_var.set(
        normalized_session_dir
    )

    return ThreadContextTokens(
        thread_id_token=thread_id_token,
        session_dir_token=session_dir_token,
    )


def reset_thread_context(
    tokens: ThreadContextTokens,
) -> None:
    """
    将请求上下文恢复到 set_thread_context 调用前的状态。

    一般应当在 finally 中调用，避免请求结束后残留旧上下文。
    """

    # reset 必须使用对应 ContextVar 产生的 Token。
    _session_dir_var.reset(
        tokens.session_dir_token
    )

    _thread_id_var.reset(
        tokens.thread_id_token
    )


def get_thread_id() -> str | None:
    """
    获取当前请求的 thread_id。

    没有设置上下文时返回 None。
    """

    return _thread_id_var.get()


def get_session_dir() -> Path | None:
    """
    获取当前请求的 session_dir。

    没有设置上下文时返回 None。
    """

    return _session_dir_var.get()


def require_thread_id() -> str:
    """
    获取当前 thread_id。

    与 get_thread_id 不同：
    如果当前没有 thread_id，就直接抛出异常。
    """

    thread_id = get_thread_id()

    if thread_id is None:
        raise RuntimeError(
            "当前执行环境中没有 thread_id。"
            "请先调用 set_thread_context()。"
        )

    return thread_id


def require_session_dir() -> Path:
    """
    获取当前 session_dir。

    如果当前没有 session_dir，就直接抛出异常。
    """

    session_dir = get_session_dir()

    if session_dir is None:
        raise RuntimeError(
            "当前执行环境中没有 session_dir。"
            "请先调用 set_thread_context()。"
        )

    return session_dir