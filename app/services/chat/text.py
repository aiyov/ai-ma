"""Turn catalog hits into text for the model and the user."""

import json

from app.schemas.chat import RetrievedChunk
from app.schemas.product import ProductHit


def dump_products(hits: list[ProductHit], excerpt) -> str:
    rows = [
        {
            "sku": hit.sku,
            "part_number": hit.part_number,
            "title": hit.title,
            "brand": hit.brand,
            "sale_price": hit.sale_price,
            "in_stock": hit.in_stock,
            "style": hit.style,
            "sku_url": hit.sku_url,
            "excerpt": excerpt(hit),
        }
        for hit in hits
    ]
    return json.dumps({"products": rows}, ensure_ascii=False)


def search_fallback(products: list[ProductHit]) -> str:
    if not products:
        return "没有找到匹配的商品。"
    lines = ["根据商品库，比较接近的是："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        lines.append(f"- {product.title}（{product.brand}，{price}，{stock}）{product.sku_url}")
    return "\n".join(lines)


def chunk_from_hit(product: ProductHit) -> RetrievedChunk:
    price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
    text = " · ".join(part for part in (product.part_number, product.brand, product.style, price) if part)
    return RetrievedChunk(
        sku=product.sku,
        title=product.title,
        chunk_index=0,
        score=None,
        text=text or product.title,
    )
