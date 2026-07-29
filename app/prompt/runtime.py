from __future__ import annotations


def build_system_reminder(
    *,
    timed_out_platforms: list[str] | None = None,
    budget_remaining: int | None = None,
    fork_count: int | None = None,
) -> str:
    lines: list[str] = []
    if timed_out_platforms:
        lines.append(
            "当前平台检索超时，请勿再次派发："
            + ", ".join(timed_out_platforms)
        )
    if budget_remaining is not None:
        lines.append(
            f"预算已更新为 {budget_remaining}。"
        )
    if fork_count is not None:
        lines.append(
            f"当前已经 fork {fork_count} 个子 loop，"
            "注意收敛。"
        )
    if not lines:
        return ""
    return (
        "<system-reminder>\n"
        + "\n".join(lines)
        + "\n</system-reminder>"
    )


__all__ = ["build_system_reminder"]
