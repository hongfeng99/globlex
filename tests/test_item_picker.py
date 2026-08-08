from app.tools.item_picker import item_picker
from app.tools.shipping_calc import LandedCost


def _item(
    item_id: str,
    landed: float,
    *,
    eta: int = 10,
    duty: float = 0,
) -> LandedCost:
    return LandedCost(
        item_id=item_id,
        platform="shopee",
        price_cny=landed - duty,
        shipping_cny=0,
        duty_cny=duty,
        landed_cny=landed,
        eta_days=eta,
        duty_tier="免征",
    )


async def test_item_picker_filters_hard_constraint():
    result = await item_picker.ainvoke(
        {
            "items": [
                _item("A-PLASTIC", 80),
                _item("B", 100),
            ],
            "preferences": ["不要塑料"],
        }
    )
    assert [item.item_id for item in result.picks] == [
        "B"
    ]
    assert result.rejected_brief


async def test_item_picker_limits_to_three():
    result = await item_picker.ainvoke(
        {
            "items": [
                _item(str(index), 100 + index)
                for index in range(5)
            ]
        }
    )
    assert len(result.picks) == 3


async def test_item_picker_accepts_json_encoded_arguments():
    plastic = _item("A-PLASTIC", 80)
    accepted = _item("B", 100)
    result = await item_picker.ainvoke(
        {
            "items": (
                "["
                + plastic.model_dump_json()
                + ","
                + accepted.model_dump_json()
                + "]"
            ),
            "preferences": '["不要塑料"]',
        }
    )

    assert [item.item_id for item in result.picks] == ["B"]
    assert result.rejected_brief
