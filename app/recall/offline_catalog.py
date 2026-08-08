from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.recall.fx import FX_RATES
from app.utils.path_utils import PROJECT_ROOT


CATALOG_VERSION = "offline-v2"
DEFAULT_SEED = 20260807
DEFAULT_VARIANTS_PER_PLATFORM = 14
DEFAULT_OFFLINE_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "offline_catalog.json"
)
DEFAULT_OFFLINE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "offline_catalog.manifest.json"
)
SYNTHETIC_NOTICE = (
    "项目生成的离线模拟商品，仅用于功能演示，"
    "不代表平台实时价格、库存或销量。"
)


@dataclass(frozen=True, slots=True)
class PlatformProfile:
    name: str
    display_name: str
    currency: str
    price_factor: float


@dataclass(frozen=True, slots=True)
class CategorySpec:
    key: str
    name: str
    keywords: tuple[str, ...]
    price_range_cny: tuple[int, int]
    styles: tuple[str, ...]
    materials: tuple[str, ...]
    features: tuple[str, ...]
    sizes: tuple[str, ...]
    weight_range_kg: tuple[float, float]
    gender: str = "通用"
    components: tuple[str, ...] = ()
    pack_sizes: tuple[int, ...] = (1,)


PLATFORMS: tuple[PlatformProfile, ...] = (
    PlatformProfile(
        "amazon",
        "Amazon（离线模拟）",
        "USD",
        1.08,
    ),
    PlatformProfile(
        "shopee",
        "Shopee（离线模拟）",
        "CNY",
        0.88,
    ),
    PlatformProfile(
        "aliexpress",
        "AliExpress（离线模拟）",
        "USD",
        0.82,
    ),
    PlatformProfile(
        "ebay",
        "eBay（离线模拟）",
        "USD",
        0.98,
    ),
)

SYNTHETIC_BRANDS: tuple[str, ...] = (
    "凌风实验室",
    "星驰工坊",
    "云帆制造",
    "远峰设计",
    "青岚科技",
    "逐光装备",
    "原野实验室",
    "轻舟工坊",
)

CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        key="cycling-kit",
        name="骑行套装",
        keywords=(
            "cycling kit",
            "cycling jersey bib shorts",
            "road bike clothing",
        ),
        price_range_cny=(180, 850),
        styles=(
            "短袖竞速套装",
            "长袖训练套装",
            "夏季透气套装",
            "耐力骑行套装",
        ),
        materials=(
            "速干聚酯纤维",
            "锦纶弹力面料",
            "透气网眼面料",
        ),
        features=(
            "吸湿排汗",
            "高弹贴身",
            "反光条",
            "3D坐垫",
            "UPF50+防晒",
            "后置口袋",
        ),
        sizes=("S", "M", "L", "XL", "2XL"),
        weight_range_kg=(0.35, 0.75),
        gender="男/女通用",
        components=("骑行上衣", "背带骑行裤", "骑行手套"),
        pack_sizes=(3,),
    ),
    CategorySpec(
        key="cycling-jersey",
        name="骑行服",
        keywords=(
            "cycling jersey",
            "bike shirt",
            "road cycling top",
        ),
        price_range_cny=(90, 480),
        styles=(
            "短袖骑行服",
            "长袖骑行服",
            "竞速版骑行服",
            "宽松版骑行服",
        ),
        materials=(
            "蜂窝速干布",
            "聚酯纤维",
            "锦纶氨纶混纺",
        ),
        features=(
            "吸湿排汗",
            "全开拉链",
            "防滑下摆",
            "反光标识",
            "三联后袋",
        ),
        sizes=("S", "M", "L", "XL", "2XL"),
        weight_range_kg=(0.18, 0.42),
        gender="男/女通用",
    ),
    CategorySpec(
        key="cycling-helmet",
        name="骑行头盔",
        keywords=(
            "cycling helmet",
            "road bike helmet",
            "bicycle safety helmet",
        ),
        price_range_cny=(120, 1200),
        styles=(
            "公路一体成型头盔",
            "山地全包围头盔",
            "通勤轻量头盔",
            "空气动力学头盔",
        ),
        materials=(
            "PC外壳+EPS内衬",
            "碳纤维复合材料",
            "ABS外壳+EPS内衬",
        ),
        features=(
            "多孔通风",
            "旋钮调节",
            "可拆洗内衬",
            "夜间反光",
            "磁吸扣",
        ),
        sizes=("M", "L", "M/L"),
        weight_range_kg=(0.21, 0.48),
    ),
    CategorySpec(
        key="cycling-gloves",
        name="骑行手套",
        keywords=(
            "cycling gloves",
            "bike gloves",
            "padded bicycle gloves",
        ),
        price_range_cny=(35, 260),
        styles=(
            "半指骑行手套",
            "全指骑行手套",
            "夏季薄款手套",
            "防风保暖手套",
        ),
        materials=(
            "弹力莱卡",
            "超细纤维掌面",
            "透气网布",
        ),
        features=(
            "减震掌垫",
            "触屏指尖",
            "防滑硅胶",
            "快速脱卸",
            "吸汗毛巾布",
        ),
        sizes=("S", "M", "L", "XL"),
        weight_range_kg=(0.06, 0.18),
    ),
    CategorySpec(
        key="mechanical-keyboard",
        name="机械键盘",
        keywords=(
            "mechanical keyboard",
            "hot swap keyboard",
            "gaming keyboard",
        ),
        price_range_cny=(160, 1200),
        styles=(
            "75%配列机械键盘",
            "87键机械键盘",
            "98键机械键盘",
            "紧凑型机械键盘",
        ),
        materials=("铝合金上盖", "PBT键帽", "ABS机身"),
        features=(
            "热插拔",
            "三模连接",
            "RGB背光",
            "全键无冲",
            "消音填充",
        ),
        sizes=("标准",),
        weight_range_kg=(0.65, 1.45),
    ),
    CategorySpec(
        key="wireless-mouse",
        name="无线鼠标",
        keywords=(
            "wireless mouse",
            "bluetooth mouse",
            "gaming mouse",
        ),
        price_range_cny=(80, 900),
        styles=(
            "轻量游戏鼠标",
            "人体工学办公鼠标",
            "便携蓝牙鼠标",
            "双模无线鼠标",
        ),
        materials=("ABS机身", "镁合金外壳", "磨砂复合材料"),
        features=(
            "低延迟",
            "可调DPI",
            "静音按键",
            "长续航",
            "多设备切换",
        ),
        sizes=("小手型", "中手型", "大手型"),
        weight_range_kg=(0.05, 0.16),
    ),
    CategorySpec(
        key="noise-cancelling-headphones",
        name="降噪耳机",
        keywords=(
            "noise cancelling headphones",
            "ANC headphones",
            "over ear bluetooth headphones",
        ),
        price_range_cny=(220, 2400),
        styles=(
            "头戴式降噪耳机",
            "便携折叠降噪耳机",
            "录音室监听耳机",
            "通勤蓝牙耳机",
        ),
        materials=("蛋白皮耳罩", "织物耳罩", "铝合金支架"),
        features=(
            "主动降噪",
            "环境声模式",
            "多点连接",
            "空间音频",
            "快充",
        ),
        sizes=("标准",),
        weight_range_kg=(0.19, 0.38),
    ),
    CategorySpec(
        key="bluetooth-earbuds",
        name="蓝牙耳机",
        keywords=(
            "bluetooth earbuds",
            "true wireless earbuds",
            "TWS earphones",
        ),
        price_range_cny=(120, 1800),
        styles=(
            "入耳式蓝牙耳机",
            "半入耳式蓝牙耳机",
            "运动挂耳耳机",
            "开放式蓝牙耳机",
        ),
        materials=("亲肤硅胶", "磨砂ABS", "液态硅胶"),
        features=(
            "通话降噪",
            "低延迟",
            "入耳检测",
            "防水防汗",
            "无线充电",
        ),
        sizes=("标准",),
        weight_range_kg=(0.04, 0.12),
    ),
    CategorySpec(
        key="travel-backpack",
        name="旅行背包",
        keywords=(
            "travel backpack",
            "carry on backpack",
            "laptop travel bag",
        ),
        price_range_cny=(160, 1000),
        styles=(
            "大开口旅行背包",
            "登机旅行背包",
            "城市通勤旅行包",
            "轻量徒步背包",
        ),
        materials=("防泼水尼龙", "再生聚酯纤维", "耐磨牛津布"),
        features=(
            "干湿分离",
            "电脑隔层",
            "行李箱固定带",
            "防盗口袋",
            "透气背负",
        ),
        sizes=("20L", "28L", "35L", "42L"),
        weight_range_kg=(0.55, 1.65),
    ),
    CategorySpec(
        key="travel-organizer",
        name="旅行收纳",
        keywords=(
            "travel organizer",
            "packing cubes",
            "compression packing bags",
        ),
        price_range_cny=(40, 240),
        styles=(
            "压缩收纳袋套装",
            "行李箱收纳袋",
            "防水旅行收纳包",
            "轻量分装袋套装",
        ),
        materials=("涤纶", "防水尼龙", "牛津布"),
        features=(
            "可压缩",
            "防尘防水",
            "网面可视",
            "可折叠",
            "双向拉链",
        ),
        sizes=("标准",),
        weight_range_kg=(0.28, 0.85),
        components=("衣物袋", "鞋袋", "内衣袋"),
        pack_sizes=(5, 6, 7, 8),
    ),
    CategorySpec(
        key="thermal-mug",
        name="保温杯",
        keywords=(
            "thermal mug",
            "insulated travel mug",
            "vacuum coffee cup",
        ),
        price_range_cny=(50, 450),
        styles=(
            "真空保温杯",
            "车载咖啡杯",
            "便携弹盖保温杯",
            "大容量吸管杯",
        ),
        materials=("304不锈钢", "316不锈钢", "陶瓷内胆"),
        features=(
            "防漏",
            "长效保温",
            "单手开盖",
            "可拆洗",
            "杯底防滑",
        ),
        sizes=("350ml", "450ml", "600ml", "900ml"),
        weight_range_kg=(0.22, 0.62),
    ),
    CategorySpec(
        key="camping-tent",
        name="露营帐篷",
        keywords=(
            "camping tent",
            "backpacking tent",
            "waterproof outdoor tent",
        ),
        price_range_cny=(300, 3000),
        styles=(
            "双层徒步帐篷",
            "自动速开帐篷",
            "家庭露营帐篷",
            "轻量登山帐篷",
        ),
        materials=("涂硅尼龙", "防水涤纶", "铝合金帐杆"),
        features=(
            "防雨",
            "双门通风",
            "快速搭建",
            "雪裙",
            "防蚊纱网",
        ),
        sizes=("1人", "2人", "3人", "4人"),
        weight_range_kg=(1.1, 5.8),
    ),
    CategorySpec(
        key="sleeping-bag",
        name="睡袋",
        keywords=(
            "sleeping bag",
            "camping sleeping bag",
            "down sleeping bag",
        ),
        price_range_cny=(180, 1800),
        styles=(
            "木乃伊式睡袋",
            "信封式睡袋",
            "轻量羽绒睡袋",
            "三季棉睡袋",
        ),
        materials=("鸭绒填充", "化纤棉填充", "防泼水尼龙"),
        features=(
            "保暖",
            "可压缩",
            "双向拉链",
            "防潮",
            "可拼接",
        ),
        sizes=("标准", "加宽", "儿童"),
        weight_range_kg=(0.65, 2.4),
    ),
    CategorySpec(
        key="running-shoes",
        name="跑步鞋",
        keywords=(
            "running shoes",
            "road running sneakers",
            "cushioned running shoes",
        ),
        price_range_cny=(180, 1500),
        styles=(
            "缓震跑步鞋",
            "竞速训练鞋",
            "稳定支撑跑鞋",
            "轻量慢跑鞋",
        ),
        materials=("工程网布", "超临界泡棉", "耐磨橡胶"),
        features=(
            "缓震",
            "回弹",
            "透气",
            "防滑",
            "足弓支撑",
        ),
        sizes=("36", "38", "40", "42", "44"),
        weight_range_kg=(0.38, 0.72),
        gender="男/女通用",
    ),
    CategorySpec(
        key="hiking-shoes",
        name="徒步鞋",
        keywords=(
            "hiking shoes",
            "waterproof trail shoes",
            "trekking footwear",
        ),
        price_range_cny=(260, 1800),
        styles=(
            "低帮徒步鞋",
            "中帮登山鞋",
            "轻量越野鞋",
            "防水徒步靴",
        ),
        materials=("防水膜", "头层牛皮", "耐磨织物"),
        features=(
            "防水",
            "防滑大底",
            "脚踝支撑",
            "鞋头防撞",
            "透气",
        ),
        sizes=("36", "38", "40", "42", "44"),
        weight_range_kg=(0.55, 1.25),
        gender="男/女通用",
    ),
    CategorySpec(
        key="yoga-mat",
        name="瑜伽垫",
        keywords=(
            "yoga mat",
            "non slip exercise mat",
            "fitness mat",
        ),
        price_range_cny=(60, 650),
        styles=(
            "防滑瑜伽垫",
            "加厚健身垫",
            "便携折叠瑜伽垫",
            "天然橡胶瑜伽垫",
        ),
        materials=("天然橡胶", "TPE", "软木+橡胶"),
        features=(
            "双面防滑",
            "回弹缓冲",
            "无异味",
            "辅助体位线",
            "易清洁",
        ),
        sizes=("4mm", "6mm", "8mm", "10mm"),
        weight_range_kg=(0.7, 3.6),
    ),
    CategorySpec(
        key="power-bank",
        name="充电宝",
        keywords=(
            "power bank",
            "portable charger",
            "USB C battery pack",
        ),
        price_range_cny=(80, 600),
        styles=(
            "快充移动电源",
            "自带线充电宝",
            "磁吸无线充电宝",
            "大容量户外电源",
        ),
        materials=("阻燃PC", "铝合金", "磨砂ABS"),
        features=(
            "USB-C快充",
            "电量显示",
            "多设备充电",
            "过温保护",
            "可登机容量",
        ),
        sizes=("5000mAh", "10000mAh", "20000mAh"),
        weight_range_kg=(0.12, 0.58),
    ),
    CategorySpec(
        key="laptop-stand",
        name="笔记本支架",
        keywords=(
            "laptop stand",
            "notebook riser",
            "ergonomic computer stand",
        ),
        price_range_cny=(60, 550),
        styles=(
            "折叠笔记本支架",
            "升降旋转支架",
            "便携电脑支架",
            "桌面散热支架",
        ),
        materials=("铝合金", "碳钢", "玻纤增强尼龙"),
        features=(
            "多档调节",
            "稳固承重",
            "开放散热",
            "防滑硅胶",
            "可折叠",
        ),
        sizes=("11-13英寸", "13-16英寸", "通用"),
        weight_range_kg=(0.22, 1.15),
    ),
)


def _local_price(
    price_cny: float,
    platform: PlatformProfile,
) -> float:
    adjusted_cny = price_cny * platform.price_factor
    if platform.currency == "CNY":
        return round(adjusted_cny, 2)
    return round(
        adjusted_cny / FX_RATES[platform.currency],
        2,
    )


def _keyboard_attributes(
    variant_index: int,
    style: str,
) -> dict[str, object]:
    """Return deterministic, filterable keyboard attributes."""

    switches = (
        ("青轴", "blue switch", "清脆"),
        ("红轴", "red switch", "适中"),
        ("茶轴", "brown switch", "适中"),
        ("静音红轴", "silent red switch", "安静"),
    )
    connections = (
        ("三模连接", ["USB-C", "2.4G", "蓝牙"]),
        ("无线双模", ["2.4G", "蓝牙"]),
        ("有线连接", ["USB-C"]),
        ("三模连接", ["USB-C", "2.4G", "蓝牙"]),
    )
    switch_type, switch_en, noise_level = switches[
        variant_index % len(switches)
    ]
    connection_type, connection_modes = connections[
        variant_index % len(connections)
    ]
    layout = style.split("机械键盘", 1)[0]
    return {
        "switch_type": switch_type,
        "switch_type_en": switch_en,
        "connection_type": connection_type,
        "connection_modes": connection_modes,
        "layout": layout,
        "use_cases": ["办公", "游戏"],
        "noise_level": noise_level,
    }


def generate_offline_catalog(
    *,
    variants_per_platform: int = (
        DEFAULT_VARIANTS_PER_PLATFORM
    ),
    seed: int = DEFAULT_SEED,
) -> list[dict[str, object]]:
    """生成可重复的多品类离线模拟商品目录。"""

    if variants_per_platform <= 0:
        raise ValueError(
            "variants_per_platform 必须大于 0。"
        )

    randomizer = random.Random(seed)
    catalog: list[dict[str, object]] = []

    for category in CATEGORY_SPECS:
        for platform_index, platform in enumerate(
            PLATFORMS
        ):
            for variant_index in range(
                variants_per_platform
            ):
                brand = SYNTHETIC_BRANDS[
                    (
                        variant_index
                        + platform_index
                        + len(category.key)
                    )
                    % len(SYNTHETIC_BRANDS)
                ]
                style = category.styles[
                    variant_index % len(category.styles)
                ]
                material = category.materials[
                    (
                        variant_index
                        + platform_index
                    )
                    % len(category.materials)
                ]
                size = category.sizes[
                    (
                        variant_index
                        + 2 * platform_index
                    )
                    % len(category.sizes)
                ]
                feature_count = min(
                    3, len(category.features)
                )
                features = randomizer.sample(
                    list(category.features),
                    k=feature_count,
                )
                if category.key == "mechanical-keyboard":
                    # Keep every platform's keyboard catalogue spread across
                    # the complete price range. Variant zero is deliberately
                    # an entry-level 青轴三模 model, so strict demo queries
                    # can have a genuine match instead of a relaxed fallback.
                    span = (
                        category.price_range_cny[1]
                        - category.price_range_cny[0]
                    )
                    denominator = max(
                        1, variants_per_platform - 1
                    )
                    base_cny = (
                        category.price_range_cny[0]
                        + span
                        * variant_index
                        / denominator
                    )
                else:
                    base_cny = randomizer.uniform(
                        *category.price_range_cny
                    )
                    base_cny *= randomizer.uniform(
                        0.94, 1.06
                    )
                item_number = (
                    platform_index
                    * variants_per_platform
                    + variant_index
                    + 1
                )
                item_id = (
                    f"offline-{platform.name}-"
                    f"{category.key}-{item_number:03d}"
                )
                size_text = (
                    "" if size == "标准" else f" {size}"
                )
                category_text = (
                    ""
                    if category.name in style
                    else f" {category.name}"
                )
                title = (
                    f"{brand} {style}{category_text} "
                    f"{material}"
                    f"{size_text}"
                )
                stock = randomizer.randint(8, 240)
                pack_size = category.pack_sizes[
                    variant_index
                    % len(category.pack_sizes)
                ]
                attributes: dict[str, object] = {
                    "brand": brand,
                    "style": style,
                    "material": material,
                    "feature": " ".join(features),
                    "features": features,
                    "size": size,
                    "gender": category.gender,
                    "weight_kg": round(
                        randomizer.uniform(
                            *category.weight_range_kg
                        ),
                        2,
                    ),
                    "keywords": " ".join(
                        category.keywords
                    ),
                }
                if category.key == "mechanical-keyboard":
                    keyboard_attributes = _keyboard_attributes(
                        variant_index,
                        style,
                    )
                    attributes.update(keyboard_attributes)
                    title += (
                        f" {keyboard_attributes['switch_type']}"
                        f" {keyboard_attributes['connection_type']}"
                    )
                if category.components:
                    attributes["components"] = list(
                        category.components
                    )
                if pack_size > 1:
                    attributes["pack_size"] = pack_size
                    title += f" {pack_size}件套"

                catalog.append(
                    {
                        "item_id": item_id,
                        "platform": platform.name,
                        "source": "offline_catalog",
                        "data_mode": "synthetic",
                        "category_key": category.key,
                        "catalog_version": (
                            CATALOG_VERSION
                        ),
                        "platform_display": (
                            platform.display_name
                        ),
                        "title": title,
                        "category": category.name,
                        "price": _local_price(
                            base_cny, platform
                        ),
                        "currency": platform.currency,
                        "rating": round(
                            randomizer.uniform(4.0, 4.9),
                            1,
                        ),
                        "sales": randomizer.randint(
                            30, 12000
                        ),
                        "stock": stock,
                        "availability": "in_stock",
                        "is_purchasable": False,
                        "product_url": None,
                        "image_url": None,
                        "attributes": attributes,
                    }
                )

    return catalog


def build_catalog_manifest(
    catalog: list[dict[str, object]],
    *,
    variants_per_platform: int,
    seed: int,
) -> dict[str, object]:
    platform_counts = Counter(
        str(item["platform"]) for item in catalog
    )
    category_counts = Counter(
        str(item["category"]) for item in catalog
    )
    return {
        "catalog_version": CATALOG_VERSION,
        "schema_version": "2.0",
        "data_mode": "synthetic",
        "notice": SYNTHETIC_NOTICE,
        "deterministic": True,
        "seed": seed,
        "variants_per_platform": (
            variants_per_platform
        ),
        "item_count": len(catalog),
        "platform_counts": dict(
            sorted(platform_counts.items())
        ),
        "category_counts": dict(
            sorted(category_counts.items())
        ),
    }


def write_offline_catalog(
    catalog_path: Path = DEFAULT_OFFLINE_CATALOG_PATH,
    manifest_path: Path = (
        DEFAULT_OFFLINE_MANIFEST_PATH
    ),
    *,
    variants_per_platform: int = (
        DEFAULT_VARIANTS_PER_PLATFORM
    ),
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path, int]:
    catalog = generate_offline_catalog(
        variants_per_platform=variants_per_platform,
        seed=seed,
    )
    manifest = build_catalog_manifest(
        catalog,
        variants_per_platform=variants_per_platform,
        seed=seed,
    )
    catalog_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    manifest_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    catalog_path.write_text(
        json.dumps(
            catalog,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return catalog_path, manifest_path, len(catalog)


__all__ = [
    "CATALOG_VERSION",
    "CATEGORY_SPECS",
    "DEFAULT_OFFLINE_CATALOG_PATH",
    "DEFAULT_OFFLINE_MANIFEST_PATH",
    "DEFAULT_SEED",
    "DEFAULT_VARIANTS_PER_PLATFORM",
    "PLATFORMS",
    "SYNTHETIC_NOTICE",
    "build_catalog_manifest",
    "generate_offline_catalog",
    "write_offline_catalog",
]
