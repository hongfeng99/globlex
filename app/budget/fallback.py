from __future__ import annotations

from app.tools.item_picker import PickedItem


def generate_fallback_answer(
    picks: list[PickedItem] | None,
    user_query: str,
) -> str:
    if picks:
        lines = ["## 推荐商品（基于已有检索结果）"]
        for index, pick in enumerate(
            picks[:3], 1
        ):
            lines.append(
                f"{index}. {pick.item_id} "
                f"({pick.platform}) — "
                f"到手约 ¥{pick.landed_cny:.2f}"
            )
            if pick.reasons:
                lines.append(
                    "   理由："
                    + "；".join(pick.reasons[:2])
                )
        return "\n".join(lines)
    return (
        f"你的请求“{user_query[:50]}…”处理时间较长，"
        "当前已有信息不足以给出完整推荐，请缩小范围后重试。"
    )


__all__ = ["generate_fallback_answer"]
