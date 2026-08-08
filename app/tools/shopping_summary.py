from __future__ import annotations

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.agent.prompts import get_shopping_summary_prompt
from app.agent.request_context import get_original_request
from app.api.monitor import monitor
from app.memory.context import get_current_user_id
from app.memory.store import preference_store
from app.recall.search_constraints import parse_search_constraints
from app.tools.item_picker import PickedItem


_PLATFORM_NAMES = {
    "amazon": "Amazon",
    "shopee": "Shopee",
    "aliexpress": "AliExpress",
    "ebay": "eBay",
}
_PUBLIC_ATTRIBUTE_KEYS = {
    "brand",
    "style",
    "material",
    "feature",
    "features",
    "size",
    "gender",
    "components",
    "pack_size",
}


class ShoppingSummaryOutput(BaseModel):
    final_text: str
    picks: list[PickedItem]
    learned_preferences: list[str] = Field(
        default_factory=list
    )

    def __str__(self) -> str:
        # return_direct 工具进入 ToolMessage 时直接把最终文案交给调用方，
        # 同时保留结构化对象供普通 Python 调用和测试使用。
        return self.final_text


def _content_as_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(
        content,
        ensure_ascii=False,
        default=str,
    )


def _normalize_learned_preferences(
    value: list[str] | str | None,
) -> list[str]:
    if value is None:
        return []
    raw_values: list[object]
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return []
        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError:
            decoded = normalized
        raw_values = decoded if isinstance(decoded, list) else [decoded]
    else:
        raw_values = list(value)
    result: list[str] = []
    for item in raw_values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_picks(
    value: list[PickedItem] | str,
) -> list[PickedItem]:
    raw_values: list[object]
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return []
        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError("picks 必须是商品数组或其 JSON 字符串。") from exc
        raw_values = decoded if isinstance(decoded, list) else [decoded]
    else:
        raw_values = list(value)
    return [
        item
        if isinstance(item, PickedItem)
        else PickedItem.model_validate(item)
        for item in raw_values
    ]


def _public_pick(pick: PickedItem) -> dict[str, object]:
    """Build the LLM payload without private/internal product identifiers."""

    return {
        "platform": pick.platform,
        "title": pick.title,
        "price_cny": pick.price_cny,
        "shipping_cny": pick.shipping_cny,
        "duty_cny": pick.duty_cny,
        "landed_cny": pick.landed_cny,
        "eta_days": pick.eta_days,
        "rating": pick.rating,
        "sales": pick.sales,
        "attributes": {
            key: value
            for key, value in pick.attributes.items()
            if key in _PUBLIC_ATTRIBUTE_KEYS
        },
        "reasons": pick.reasons,
    }


def _attribute_summary(pick: PickedItem) -> str:
    attributes = pick.attributes
    details: list[str] = []
    components = attributes.get("components")
    if isinstance(components, list) and components:
        details.append(
            "包含" + "、".join(str(value) for value in components)
        )
    for key in ("material", "size"):
        value = str(attributes.get(key, "")).strip()
        if value:
            details.append(value)
    features = attributes.get("features")
    if isinstance(features, list):
        feature_text = "、".join(
            str(value) for value in features[:3]
        )
        if feature_text:
            details.append(feature_text)
    return "；".join(details[:4]) or "商品属性以离线模拟库为准"


def _fallback_summary(picks: list[PickedItem]) -> str:
    lines = ["以下为符合当前硬条件的离线模拟候选："]
    for index, pick in enumerate(picks, start=1):
        platform = _PLATFORM_NAMES.get(
            pick.platform, pick.platform
        )
        title = pick.title or f"{platform} 模拟商品"
        cost_detail = (
            f"商品价 ¥{pick.price_cny:.2f} + "
            f"运费 ¥{pick.shipping_cny:.2f} + "
            f"关税 ¥{pick.duty_cny:.2f}"
        )
        reasons = "；".join(pick.reasons) or "满足当前硬条件"
        lines.extend(
            [
                f"{index}. **{title}**（{platform}）",
                f"   - 预估到手价：¥{pick.landed_cny:.2f}（{cost_detail}）",
                f"   - 商品信息：{_attribute_summary(pick)}",
                f"   - 推荐理由：{reasons}",
                (
                    f"   - 预计到货：约 {pick.eta_days} 天"
                    if pick.eta_days is not None
                    else "   - 预计到货：暂无可靠数据"
                ),
            ]
        )
    lines.append("注：以上为离线模拟数据，不代表平台实时库存或直邮状态。")
    return "\n".join(lines)


def _sanitize_summary(
    final_text: str,
    picks: list[PickedItem],
) -> str:
    # Remove both known IDs and arbitrary model-generated `ID: xxx` labels.
    sanitized = re.sub(
        r"\s*[（(]\s*(?:商品\s*)?ID\s*[:：][^）)\n]+[）)]",
        "",
        final_text,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"(?:商品\s*)?ID\s*[:：]\s*[A-Za-z0-9_.-]+",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    for pick in picks:
        if pick.item_id:
            sanitized = sanitized.replace(
                pick.item_id,
                pick.title or "模拟商品",
            )
    sanitized = re.sub(r"[（(]\s*[）)]", "", sanitized)

    has_all_titles = all(
        pick.title and pick.title in sanitized
        for pick in picks
    )
    has_all_platforms = all(
        _PLATFORM_NAMES.get(pick.platform, pick.platform)
        in sanitized
        for pick in picks
    )
    unsupported_claim = any(
        phrase in sanitized
        for phrase in ("可直邮", "支持直邮", "能够直邮")
    )
    if (
        not has_all_titles
        or not has_all_platforms
        or unsupported_claim
    ):
        return _fallback_summary(picks)
    return sanitized.strip()


@tool(return_direct=True)
async def shopping_summary(
    picks: list[PickedItem] | str,
    user_request: str,
    learned_preferences: list[str] | str | None = None,
) -> ShoppingSummaryOutput:
    """终结性工具：把最多 3 件精挑结果整理成简洁、可执行的购物清单。"""

    picks = _normalize_picks(picks)[:3]
    constraints = parse_search_constraints(
        get_original_request(),
        user_request,
    )
    if constraints.max_landed_cny is not None:
        picks = [
            pick
            for pick in picks
            if pick.landed_cny
            <= constraints.max_landed_cny
        ]
    learned_preferences = _normalize_learned_preferences(
        learned_preferences
    )
    await monitor.report_tool_start(
        "shopping_summary",
        {"picks_count": len(picks)},
    )
    started_at = time.perf_counter()
    if not picks:
        labels = constraints.labels()
        final_text = (
            "未找到同时满足当前硬条件的离线模拟商品"
            + (f"（{'、'.join(labels)}）" if labels else "")
            + "。系统未自动放宽预算或商品属性；"
            "如需备选，请明确说明允许放宽哪一项条件。"
        )
        current_user_id = get_current_user_id()
        if current_user_id and learned_preferences:
            await preference_store.add_many(
                current_user_id,
                learned_preferences,
                source="shopping_summary",
            )
        await monitor.report_tool_end(
            "shopping_summary",
            int((time.perf_counter() - started_at) * 1000),
        )
        return ShoppingSummaryOutput(
            final_text=final_text,
            picks=[],
            learned_preferences=learned_preferences,
        )

    payload = {
        "user_request": user_request,
        "hard_constraints": constraints.labels(),
        "picks": [_public_pick(pick) for pick in picks],
    }
    response = await get_llm().ainvoke(
        [
            SystemMessage(
                content=get_shopping_summary_prompt()
            ),
            HumanMessage(
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            ),
        ]
    )
    final_text = _sanitize_summary(
        _content_as_text(response.content),
        picks,
    )
    current_user_id = get_current_user_id()
    if current_user_id and learned_preferences:
        await preference_store.add_many(
            current_user_id,
            learned_preferences,
            source="shopping_summary",
        )
    await monitor.report_tool_end(
        "shopping_summary",
        int(
            (time.perf_counter() - started_at)
            * 1000
        ),
    )
    return ShoppingSummaryOutput(
        final_text=final_text,
        picks=picks,
        learned_preferences=(
            learned_preferences
        ),
    )


__all__ = [
    "ShoppingSummaryOutput",
    "_normalize_learned_preferences",
    "_normalize_picks",
    "_sanitize_summary",
    "shopping_summary",
]
