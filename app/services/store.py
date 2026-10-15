import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.core.config import settings
from app.schemas.product import ProductFilter, ProductHit
from app.services.catalog import Product
from app.services.embeddings import Embedder


class ProductStore:
    """One point per SKU.

    Named vectors:
    - text: product title, category, specs, and description
    - image: CLIP embedding of the product photo

    A text question searches `text`. An uploaded photo searches `image`.
    A phrase meant to match how a product looks searches `image` with the CLIP text encoder.
    """

    def __init__(self, client: QdrantClient | None = None, embedder: Embedder | None = None) -> None:
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            timeout=60,
            trust_env=False,
        )
        self.embedder = embedder or Embedder()
        self.collection = settings.collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._check_schema()
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "text": VectorParams(size=settings.text_dim, distance=Distance.COSINE),
                    "image": VectorParams(size=settings.image_dim, distance=Distance.COSINE),
                },
            )
        info = self.client.get_collection(self.collection)
        existing = set(info.payload_schema)
        indexes = {
            "sku": PayloadSchemaType.KEYWORD,
            "brand_key": PayloadSchemaType.KEYWORD,
            "category_key": PayloadSchemaType.KEYWORD,
            "category_keys": PayloadSchemaType.KEYWORD,
            "condition_key": PayloadSchemaType.KEYWORD,
            "part_number": PayloadSchemaType.KEYWORD,
            "in_stock": PayloadSchemaType.BOOL,
            "sale_price": PayloadSchemaType.FLOAT,
        }
        for field_name, schema in indexes.items():
            if field_name not in existing:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )

    def upsert(self, products: list[Product], image_paths: dict[str, Path]) -> int:
        self.ensure_collection()
        if not products:
            return 0
        text_vectors = self.embedder.embed_documents([product.search_text for product in products])
        points: list[PointStruct] = []
        for product, text_vector in zip(products, text_vectors, strict=True):
            vectors: dict[str, list[float]] = {"text": text_vector}
            image_path = image_paths.get(product.sku)
            if image_path is not None:
                vectors["image"] = self.embedder.embed_image(image_path)
            points.append(
                PointStruct(
                    id=point_id(product.sku),
                    vector=vectors,
                    payload=payload_of(product),
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = self.embedder.embed_query(query)
        return self._search(vector, using="text", limit=limit, filters=filters)

    def search_visual_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = self.embedder.embed_visual_query(query)
        return self._search(vector, using="image", limit=limit, filters=filters)

    def search_image(self, path: Path, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = self.embedder.embed_image(path)
        return self._search(vector, using="image", limit=limit, filters=filters)

    def get_by_sku(self, sku: str) -> ProductHit | None:
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id(sku)],
            with_payload=True,
        )
        if not points:
            return None
        return hit_from_payload(points[0].payload or {}, score=None)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def _search(
        self,
        vector: list[float],
        using: str,
        limit: int,
        filters: ProductFilter | None,
    ) -> list[ProductHit]:
        found = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=using,
            query_filter=build_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return [hit_from_payload(point.payload or {}, score=point.score) for point in found.points]

    def _check_schema(self) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        expected = {"text": settings.text_dim, "image": settings.image_dim}
        actual = {name: params.size for name, params in vectors.items()} if isinstance(vectors, dict) else {}
        if actual != expected:
            raise RuntimeError(
                f"Collection {self.collection!r} has vectors {actual}, expected {expected}. "
                "Drop it in the Qdrant dashboard and import again."
            )


def point_id(sku: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"product:{sku}"))


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
