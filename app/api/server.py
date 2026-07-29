from __future__ import annotations

import asyncio
from pathlib import Path
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
from pydantic import BaseModel

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


class TaskRequest(BaseModel):
    query: str
    thread_id: str | None = None
    user_id: str | None = None


async def _run_task(
    request: TaskRequest,
    thread_id: str,
) -> None:
    session_dir = create_session_dir(thread_id)
    try:
        with bind_thread_context(
            thread_id, session_dir
        ):
            await run_main_agent(
                request.query,
                thread_id=thread_id,
                user_id=request.user_id,
            )
    finally:
        active_tasks.pop(thread_id, None)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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

    active_tasks[thread_id] = (
        asyncio.create_task(
            _run_task(request, thread_id)
        )
    )
    return {
        "status": "started",
        "thread_id": thread_id,
    }


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


__all__ = ["active_tasks", "app"]
