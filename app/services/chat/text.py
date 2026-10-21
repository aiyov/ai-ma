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


def list_fallback(products: list[ProductHit]) -> str:
    if not products:
        return "商品库里还没有型号。"
    lines = ["商品库里现在有这些型号："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        model_name = product.part_number or product.sku
        lines.append(f"- {model_name}　{product.brand}　{product.title}（{price}，{stock}）")
    return "\n".join(lines)


def direct_reply(message: str, social) -> str:
    if social(message.strip()):
        return "我是乐器商品库的导购，可以帮你查型号、价格和库存。直接说想找什么样的乐器就行。"
    return "我只能根据商品库回答乐器相关的问题。想找长笛、看价格或库存，直接说就行。"


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


def chunk_from_document(doc) -> RetrievedChunk:
    meta = doc.metadata
    score = meta.get("score")
    return RetrievedChunk(
        sku=str(meta.get("sku") or ""),
        title=str(meta.get("title") or ""),
        chunk_index=int(meta.get("chunk_index") or 0),
        score=None if score is None else float(score),
        text=doc.page_content,
    )


def format_docs(docs: list) -> str:
    if not docs:
        return "No matching products."
    blocks = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"[{index}] sku={meta.get('sku')} brand={meta.get('brand')} category={meta.get('category')}",
                    f"price={meta.get('sale_price')} msrp={meta.get('msrp')} in_stock={meta.get('in_stock')}",
                    f"style={meta.get('style')}",
                    f"sku_url={meta.get('sku_url')}",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def catalog_context(products: list[ProductHit]) -> str:
    if not products:
        return "The catalog is empty."
    blocks = []
    for index, product in enumerate(products, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] part_number={product.part_number or product.sku} brand={product.brand}",
                    f"title={product.title}",
                    f"price={product.sale_price} in_stock={product.in_stock} style={product.style}",
                    f"sku_url={product.sku_url}",
                ]
            )
        )
    return "\n\n".join(blocks)
