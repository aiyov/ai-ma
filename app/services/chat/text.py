"""把检索到的商品整理成文本。

工具结果要交给通义千问，所以是 JSON；模型没写出回答时，再退回一段给人看的纯文本。
"""

import json

from app.schemas.chat import RetrievedChunk
from app.schemas.product import ProductHit


def dump_products(hits: list[ProductHit], excerpt) -> str:
    """把商品列表转成工具返回值。

    excerpt 由调用方决定摘哪一段：检索用命中的文本块，列目录用标题，读单品用更长的正文。
    价格、库存、链接都原样放进 JSON，模型只能根据这些字段回答，不能自己编。
    """
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
    # ensure_ascii=False 让中文标题留在 JSON 里，而不是变成 \\uXXXX。
    return json.dumps({"products": rows}, ensure_ascii=False)


def search_fallback(products: list[ProductHit]) -> str:
    """模型没有生成文字时，用商品库结果拼一段回答。

    只在工具已经查到商品、但模型回复为空时使用。价格带美元符号，和提示词里“不要改成人民币”一致。
    """
    if not products:
        return "没有找到匹配的商品。"
    lines = ["根据商品库，比较接近的是："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        lines.append(f"- {product.title}（{product.brand}，{price}，{stock}）{product.sku_url}")
    return "\n".join(lines)


def chunk_from_hit(product: ProductHit) -> RetrievedChunk:
    """列目录时给前端的短引用，不是商品全文。

    列出来的是型号、品牌、款式和价格。chunk_index 固定为 0，因为目录项对应每个 SKU 的第一条记录。
    """
    price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
    text = " · ".join(part for part in (product.part_number, product.brand, product.style, price) if part)
    return RetrievedChunk(
        sku=product.sku,
        title=product.title,
        chunk_index=0,
        score=None,
        text=text or product.title,
    )
