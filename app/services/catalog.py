import json
import re
from html import unescape
from typing import Any

from pydantic import BaseModel, Field


class Product(BaseModel):
    sku: str
    product_id: str = ""
    title: str
    brand: str = ""
    category: str = ""
    category_path: list[str] = Field(default_factory=list)
    condition: str = ""
    part_number: str = ""
    style: str = ""
    heading: str = ""
    sale_price: float | None = None
    msrp: float | None = None
    in_stock: bool = False
    rating: float | None = None
    rating_count: int = 0
    image_url: str = ""
    embed_image_url: str = ""
    thumb_url: str = ""
    sku_url: str = ""
    search_text: str
    document: str


def load_products(path) -> list[Product]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    products: list[Product] = []
    for record in records:
        product = _unwrap_product(record)
        if product is not None:
            products.append(product)
    return products


def _unwrap_product(record: Any) -> Product | None:
    if not isinstance(record, dict):
        return None
    body = record.get("data", record)
    if isinstance(body, dict) and "product" in body:
        body = body["product"]
    if not isinstance(body, dict):
        return None
    variant = (
        body.get("masterData", {})
        .get("current", {})
        .get("masterVariant")
    )
    if not isinstance(variant, dict) or not variant.get("sku"):
        return None
    if variant.get("searchable") is False:
        return None
    return _build_product(body, variant)


def _build_product(body: dict, variant: dict) -> Product:
    attrs = _attributes(variant.get("attributesRaw") or [])
    images = _images(variant.get("images") or [], attrs.get("productImages"))
    detail = attrs.get("pdpDetail") if isinstance(attrs.get("pdpDetail"), dict) else {}
    price_info = attrs.get("priceTypeInfo") if isinstance(attrs.get("priceTypeInfo"), dict) else {}
    breadcrumb = _breadcrumb(attrs.get("productBreadcrumb"))
    category = str(detail.get("categoryName") or (breadcrumb[-1] if breadcrumb else ""))
    styles = [
        str(attrs[name])
        for name in ("styleDescription1", "styleDescription2")
        if attrs.get(name)
    ]
    title = str(
        attrs.get("title")
        or attrs.get("productDisplayName")
        or attrs.get("displayName")
        or variant["sku"]
    )
    specs = _html_to_text(str(attrs.get("productSpecifications") or ""))
    description = _html_to_text(str(attrs.get("productLongDescription") or ""))
    heading = str(attrs.get("heading") or "")
    brand = str(detail.get("brandName") or "")
    condition = str(attrs.get("condition") or detail.get("commerceType") or "")
    part_number = str(attrs.get("partNumber") or "")
    rating_stats = body.get("reviewRatingStatistics") or {}
    document = _document(
        title=title,
        brand=brand,
        category_path=breadcrumb,
        condition=condition,
        style=", ".join(styles),
        part_number=part_number,
        heading=heading,
        specs=specs,
        description=description,
    )
    return Product(
        sku=str(variant["sku"]),
        product_id=str(attrs.get("enterpriseProductId") or detail.get("productId") or ""),
        title=title,
        brand=brand,
        category=category,
        category_path=breadcrumb,
        condition=condition,
        part_number=part_number,
        style=", ".join(styles),
        heading=heading,
        sale_price=_number(price_info.get("salePrice")),
        msrp=_number(attrs.get("MSRP")),
        in_stock=_in_stock(attrs),
        rating=_number(rating_stats.get("averageRating")),
        rating_count=int(rating_stats.get("count") or 0),
        image_url=images.get("large", ""),
        embed_image_url=images.get("middle") or images.get("large", ""),
        thumb_url=images.get("small", ""),
        sku_url=str(attrs.get("skuUrl") or ""),
        search_text=_clip_text(document, 1500),
        document=document,
    )


def _attributes(raw: list) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            continue
        parsed[str(item["name"])] = _parse_value(item.get("value"))
    return parsed


def _parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped[:1] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _images(images: list, product_images: Any) -> dict[str, str]:
    by_label = {
        "large": "",
        "middle": "",
        "small": "",
    }
    label_map = {
        "largeImageSrc": "large",
        "middleImageSrc": "middle",
        "smallImageSrc": "small",
    }
    for image in images:
        if not isinstance(image, dict):
            continue
        slot = label_map.get(str(image.get("label") or ""))
        if slot and image.get("url"):
            by_label[slot] = str(image["url"])
    if isinstance(product_images, dict):
        for label, slot in label_map.items():
            if not by_label[slot] and product_images.get(label):
                by_label[slot] = str(product_images[label])
    return by_label


def _breadcrumb(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("displayName") or "")
        if name and name.lower() != "home":
            names.append(name)
    return names


def _in_stock(attrs: dict[str, Any]) -> bool:
    detail = attrs.get("inventoryCTADetail")
    if isinstance(detail, dict) and str(detail.get("avlStatus") or "").lower() == "in_stock":
        return True
    stock = attrs.get("repositoryStockInfo")
    if isinstance(stock, dict):
        try:
            return float(stock.get("global") or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _document(
    title: str,
    brand: str,
    category_path: list[str],
    condition: str,
    style: str,
    part_number: str,
    heading: str,
    specs: str,
    description: str,
) -> str:
    lines = [title]
    if brand:
        lines.append(f"Brand: {brand}")
    if category_path:
        lines.append("Category: " + " > ".join(category_path))
    if condition:
        lines.append(f"Condition: {condition}")
    if style:
        lines.append(f"Style: {style}")
    if part_number:
        lines.append(f"Part number: {part_number}")
    if heading:
        lines.append(heading)
    if specs:
        lines.append("Specifications: " + specs)
    if description:
        lines.append(description)
    return "\n".join(lines)


def _html_to_text(value: str) -> str:
    if not value:
        return ""
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|li|h[1-6]|ul|ol|tr)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _clip_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    clipped = compact[:limit].rsplit(" ", 1)[0]
    return clipped or compact[:limit]
