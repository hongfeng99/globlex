def migrate_prompt_gracefully(
    old_version: str,
    new_version: str,
) -> dict[str, object]:
    return {
        "from": old_version,
        "to": new_version,
        "ab_compare_token_cost": False,
        "warmup_hours": 24,
        "cache_control_preserved": True,
    }
