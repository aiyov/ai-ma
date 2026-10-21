"""Qdrant collection schema for the product index.

Collection name comes from settings so environments can isolate indexes.
Vector sizes and payload indexes live here so write/read stay aligned.
"""

from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app.core.config import settings


def vectors_config() -> dict[str, VectorParams]:
    return {
        "text": VectorParams(size=settings.text_dim, distance=Distance.COSINE),
        "image": VectorParams(size=settings.image_dim, distance=Distance.COSINE),
    }


def expected_vector_sizes() -> dict[str, int]:
    return {"text": settings.text_dim, "image": settings.image_dim}


PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "sku": PayloadSchemaType.KEYWORD,
    "brand_key": PayloadSchemaType.KEYWORD,
    "category_key": PayloadSchemaType.KEYWORD,
    "category_keys": PayloadSchemaType.KEYWORD,
    "condition_key": PayloadSchemaType.KEYWORD,
    "part_number": PayloadSchemaType.KEYWORD,
    "in_stock": PayloadSchemaType.BOOL,
    "sale_price": PayloadSchemaType.FLOAT,
    "chunk_index": PayloadSchemaType.INTEGER,
}
