from app.agent.offline_fallback import build_clarification


def test_bare_keyboard_request_gets_local_clarification() -> None:
    result = build_clarification("我要买机械键盘")

    assert result is not None
    assert "预算上限" in result
    assert "轴体偏好" in result
    assert "连接方式" in result
    assert "下一轮会自动与本需求合并" in result


def test_detailed_keyboard_request_does_not_clarify() -> None:
    assert build_clarification(
        "500元，办公，青轴，无线机械键盘"
    ) is None
