"""Content fingerprint for fields that affect search and answers."""

import hashlib
import json

from app.services.catalog import Product


def content_hash(product: Product) -> str:
    payload = {
        "sku": product.sku,
        "product_id": product.product_id,
        "title": product.title,
        "brand": product.brand,
        "category": product.category,
        "category_path": product.category_path,
        "condition": product.condition,
        "part_number": product.part_number,
        "style": product.style,
        "heading": product.heading,
        "sale_price": product.sale_price,
        "msrp": product.msrp,
        "in_stock": product.in_stock,
        "image_url": product.image_url,
        "embed_image_url": product.embed_image_url,
        "thumb_url": product.thumb_url,
        "sku_url": product.sku_url,
        "document": product.document,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
