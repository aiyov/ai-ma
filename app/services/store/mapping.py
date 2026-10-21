"""Payload and document helpers shared by write and read paths."""

import uuid

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from app.schemas.product import ProductFilter, ProductHit
from app.services.catalog import Product


def point_id(sku: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"product:{sku}"))


def chunk_id(sku: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"product:{sku}:{index}"))


def payload_of(product: Product) -> dict:
    category_keys = [name.casefold() for name in product.category_path]
    if product.category and product.category.casefold() not in category_keys:
        category_keys.append(product.category.casefold())
    return {
        "sku": product.sku,
        "product_id": product.product_id,
        "title": product.title,
        "brand": product.brand,
        "brand_key": product.brand.casefold(),
        "category": product.category,
        "category_key": product.category.casefold(),
        "category_path": product.category_path,
        "category_keys": category_keys,
        "condition": product.condition,
        "condition_key": product.condition.casefold(),
        "part_number": product.part_number,
        "style": product.style,
        "heading": product.heading,
        "sale_price": product.sale_price,
        "msrp": product.msrp,
        "in_stock": product.in_stock,
        "rating": product.rating,
        "rating_count": product.rating_count,
        "image_url": product.image_url,
        "thumb_url": product.thumb_url,
        "sku_url": product.sku_url,
        "search_text": product.search_text,
        "document": product.document,
    }


def hit_from_payload(payload: dict, score: float | None) -> ProductHit:
    return ProductHit(
        sku=str(payload.get("sku") or ""),
        product_id=str(payload.get("product_id") or ""),
        title=str(payload.get("title") or ""),
        brand=str(payload.get("brand") or ""),
        category=str(payload.get("category") or ""),
        category_path=list(payload.get("category_path") or []),
        condition=str(payload.get("condition") or ""),
        part_number=str(payload.get("part_number") or ""),
        style=str(payload.get("style") or ""),
        heading=str(payload.get("heading") or ""),
        sale_price=payload.get("sale_price"),
        msrp=payload.get("msrp"),
        in_stock=bool(payload.get("in_stock")),
        rating=payload.get("rating"),
        rating_count=int(payload.get("rating_count") or 0),
        image_url=str(payload.get("image_url") or ""),
        thumb_url=str(payload.get("thumb_url") or ""),
        sku_url=str(payload.get("sku_url") or ""),
        score=None if score is None else round(float(score), 4),
        document=str(payload.get("document") or ""),
    )


def document_from_point(payload: dict, score: float | None) -> Document:
    metadata = dict(payload)
    if score is not None:
        metadata["score"] = round(float(score), 4)
    content = str(payload.get("search_text") or payload.get("document") or "")
    return Document(page_content=content, metadata=metadata)


def hit_from_document(document: Document) -> ProductHit:
    hit = hit_from_payload(document.metadata, score=document.metadata.get("score"))
    return hit.model_copy(update={"document": document.page_content})


def dedupe(documents: list[Document], limit: int) -> list[Document]:
    chosen: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        sku = str(document.metadata.get("sku") or "")
        if sku in seen:
            continue
        seen.add(sku)
        chosen.append(document)
        if len(chosen) >= limit:
            break
    return chosen


def lookup_keys(key: str) -> list[str]:
    raw = key.strip()
    if not raw:
        return []
    parts = [raw, raw.upper(), *raw.replace(",", " ").split()]
    parts.extend(part.upper() for part in raw.replace(",", " ").split())
    return [part for part in parts if part]


def build_filter(filters: ProductFilter | None) -> Filter | None:
    if filters is None:
        return None
    must: list = []
    if filters.brand:
        must.append(FieldCondition(key="brand_key", match=MatchValue(value=filters.brand.casefold())))
    if filters.category:
        must.append(
            Filter(
                should=[
                    FieldCondition(key="category_key", match=MatchValue(value=filters.category.casefold())),
                    FieldCondition(key="category_keys", match=MatchValue(value=filters.category.casefold())),
                ]
            )
        )
    if filters.condition:
        must.append(
            FieldCondition(key="condition_key", match=MatchValue(value=filters.condition.casefold()))
        )
    if filters.in_stock is not None:
        must.append(FieldCondition(key="in_stock", match=MatchValue(value=filters.in_stock)))
    if filters.min_price is not None or filters.max_price is not None:
        must.append(
            FieldCondition(
                key="sale_price",
                range=Range(gte=filters.min_price, lte=filters.max_price),
            )
        )
    if not must:
        return None
    return Filter(must=must)
