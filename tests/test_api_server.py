from __future__ import annotations

import time

from fastapi.testclient import TestClient

import app.api.server as server_module


def _wait_for_terminal(
    client: TestClient,
    thread_id: str,
) -> dict:
    for _ in range(100):
        response = client.get(
            f"/api/task/{thread_id}"
        )
        body = response.json()
        if body["status"] in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return body
        time.sleep(0.01)
    raise AssertionError("任务未进入终态")


def test_task_status_keeps_background_result(
    monkeypatch,
) -> None:
    async def fake_run_main_agent(*args, **kwargs):
        return "测试结果"

    monkeypatch.setattr(
        server_module,
        "run_main_agent",
        fake_run_main_agent,
    )

    with TestClient(server_module.app) as client:
        response = client.post(
            "/api/task",
            json={
                "query": "测试",
                "thread_id": "api-success",
            },
        )
        assert response.status_code == 200
        record = _wait_for_terminal(
            client,
            "api-success",
        )

    assert record["status"] == "succeeded"
    assert record["result"] == "测试结果"


def test_task_status_keeps_background_error(
    monkeypatch,
) -> None:
    async def failing_run_main_agent(*args, **kwargs):
        raise RuntimeError("模型服务不可用")

    monkeypatch.setattr(
        server_module,
        "run_main_agent",
        failing_run_main_agent,
    )

    with TestClient(server_module.app) as client:
        response = client.post(
            "/api/task",
            json={
                "query": "测试",
                "thread_id": "api-failure",
            },
        )
        assert response.status_code == 200
        record = _wait_for_terminal(
            client,
            "api-failure",
        )

    assert record["status"] == "failed"
    assert record["error"] == "模型服务不可用"


def test_blank_query_is_rejected() -> None:
    with TestClient(server_module.app) as client:
        response = client.post(
            "/api/task",
            json={"query": "   "},
        )

    assert response.status_code == 422


def test_ready_reports_category_kb(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        server_module,
        "get_category_kb_status",
        lambda: {
            "available": True,
            "document_count": 108,
        },
    )
    monkeypatch.setenv("CATEGORY_KB_REQUIRED", "true")

    with TestClient(server_module.app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["category_kb"] == {
        "available": True,
        "document_count": 108,
    }


def test_ready_fails_when_required_kb_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        server_module,
        "get_category_kb_status",
        lambda: {
            "available": False,
            "reason": "index_missing",
        },
    )
    monkeypatch.setenv("CATEGORY_KB_REQUIRED", "true")

    with TestClient(server_module.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["category_kb"][
        "reason"
    ] == "index_missing"
