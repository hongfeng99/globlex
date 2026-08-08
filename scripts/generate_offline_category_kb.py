from app.recall.offline_category_kb import write_category_cards


if __name__ == "__main__":
    cards_path, manifest_path, count = write_category_cards()
    print(
        f"已生成 {count} 条离线模拟品类知识卡："
        f"{cards_path} / {manifest_path}"
    )
