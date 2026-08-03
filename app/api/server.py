from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.main_agent import run_main_agent
from app.api.connection import manager
from app.utils.path_utils import (
    create_session_dir,
    normalize_thread_id,
    resolve_uploaded_path,
)
from app.utils.thread_ctx import bind_thread_context


app = FastAPI(title="Globex Agent")
active_tasks: dict[str, asyncio.Task[None]] = {}
task_records: dict[str, "TaskRecord"] = {}


class TaskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = None
    user_id: str | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空。")
        return normalized


TaskStatus = Literal[
    "started",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


class TaskRecord(BaseModel):
    thread_id: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


def _update_task_record(
    thread_id: str,
    *,
    status: TaskStatus,
    result: str | None = None,
    error: str | None = None,
) -> None:
    record = task_records[thread_id]
    record.status = status
    record.result = result
    record.error = error
    record.updated_at = datetime.now(UTC)


async def _run_task(
    request: TaskRequest,
    thread_id: str,
) -> None:
    session_dir = create_session_dir(thread_id)
    _update_task_record(
        thread_id,
        status="running",
    )
    try:
        with bind_thread_context(
            thread_id, session_dir
        ):
            result = await run_main_agent(
                request.query,
                thread_id=thread_id,
                user_id=request.user_id,
            )
        _update_task_record(
            thread_id,
            status="succeeded",
            result=result,
        )
    except asyncio.CancelledError:
        _update_task_record(
            thread_id,
            status="cancelled",
            error="任务已取消",
        )
    except Exception as exc:
        # 后台任务不能把异常重新抛到无人 await 的 Task 中；错误详情
        # 同时保存在状态接口，前端断线后仍可查询。
        _update_task_record(
            thread_id,
            status="failed",
            error=str(exc),
        )
    finally:
        active_tasks.pop(thread_id, None)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    # 进程级就绪检查；外部依赖采用按工具降级，不阻止 API 启动。
    return {"status": "ready"}


@app.post("/api/task")
async def create_task(
    request: TaskRequest,
) -> dict[str, str]:
    thread_id = normalize_thread_id(
        request.thread_id
        or f"task-{uuid4().hex[:12]}"
    )
    existing = active_tasks.get(thread_id)
    if existing and not existing.done():
        return {
            "status": "already_running",
            "thread_id": thread_id,
        }

    await manager.clear_history(thread_id)
    task_records[thread_id] = TaskRecord(
        thread_id=thread_id,
        status="started",
    )
    active_tasks[thread_id] = (
        asyncio.create_task(
            _run_task(request, thread_id)
        )
    )
    return {
        "status": "started",
        "thread_id": thread_id,
    }


@app.get(
    "/api/task/{thread_id}",
    response_model=TaskRecord,
)
async def get_task(thread_id: str) -> TaskRecord:
    thread_id = normalize_thread_id(thread_id)
    record = task_records.get(thread_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="任务不存在",
        )
    return record


@app.post("/api/task/{thread_id}/cancel")
async def cancel_task(
    thread_id: str,
) -> dict[str, str]:
    thread_id = normalize_thread_id(thread_id)
    task = active_tasks.get(thread_id)
    if task is None or task.done():
        return {
            "status": "not_running",
            "thread_id": thread_id,
        }
    task.cancel()
    return {
        "status": "cancelled",
        "thread_id": thread_id,
    }


@app.websocket("/ws/{thread_id}")
async def websocket_events(
    websocket: WebSocket,
    thread_id: str,
) -> None:
    thread_id = normalize_thread_id(thread_id)
    await manager.connect(websocket, thread_id)
    try:
        while True:
            # Keep the socket alive and notice client disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(
            websocket, thread_id
        )


@app.post("/api/upload/{thread_id}")
async def upload_file(
    thread_id: str,
    file: UploadFile = File(...),
) -> dict[str, str]:
    safe_name = Path(
        file.filename or "upload.bin"
    ).name
    target = resolve_uploaded_path(
        thread_id,
        safe_name,
        create_parent=True,
    )
    target.write_bytes(await file.read())
    return {
        "filename": safe_name,
        "path": str(target),
    }


@app.get(
    "/api/files/{thread_id}/{relative_path:path}"
)
async def download_file(
    thread_id: str,
    relative_path: str,
) -> FileResponse:
    session_dir = create_session_dir(thread_id)
    candidate = (
        session_dir / relative_path
    ).resolve()
    try:
        candidate.relative_to(session_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="非法文件路径",
        ) from exc
    if not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail="文件不存在",
        )
    return FileResponse(candidate)


__all__ = [
    "TaskRecord",
    "active_tasks",
    "app",
    "task_records",
]
