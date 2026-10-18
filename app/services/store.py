import uuid
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.core.config import settings
from app.schemas.product import ProductFilter, ProductHit
from app.services.catalog import Product
from app.services.documents import split_product
from app.services.embeddings import ClipTextEmbeddings, FastEmbedTextEmbeddings, embed_image


class ProductStore:
    """Product chunks in one Qdrant collection.

    Named vectors:
    - text: a LangChain document chunk from the title, specs, and description
    - image: CLIP embedding of the product photo, stored on the first chunk

    A text question searches `text`. An uploaded photo searches `image`.
    A phrase meant to match how a product looks searches `image` with the CLIP text encoder.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        text_embeddings: FastEmbedTextEmbeddings | None = None,
        clip_embeddings: ClipTextEmbeddings | None = None,
    ) -> None:
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            timeout=60,
            trust_env=False,
        )
        self.text_embeddings = text_embeddings or FastEmbedTextEmbeddings()
        self.clip_embeddings = clip_embeddings or ClipTextEmbeddings()
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
            "chunk_index": PayloadSchemaType.INTEGER,
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
        self._delete_skus([product.sku for product in products])
        documents: list[Document] = []
        image_for_chunk: list[Path | None] = []
        for product in products:
            chunks = split_product(product, payload_of(product))
            image_path = image_paths.get(product.sku)
            for chunk in chunks:
                documents.append(chunk)
                image_for_chunk.append(image_path if chunk.metadata.get("chunk_index") == 0 else None)
        vectors = self.text_embeddings.embed_documents([doc.page_content for doc in documents])
        points: list[PointStruct] = []
        for document, text_vector, image_path in zip(documents, vectors, image_for_chunk, strict=True):
            named: dict[str, list[float]] = {"text": text_vector}
            if image_path is not None:
                named["image"] = embed_image(image_path)
            payload = {**document.metadata, "search_text": document.page_content}
            points.append(
                PointStruct(
                    id=chunk_id(str(payload["sku"]), int(payload["chunk_index"])),
                    vector=named,
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def as_retriever(self, limit: int = 4, filters: ProductFilter | None = None) -> "ProductRetriever":
        return ProductRetriever(store=self, k=limit, filters=filters)

    def search_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        return [hit_from_document(doc) for doc in self.relevant_documents(query, limit, filters)]

    def relevant_documents(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[Document]:
        vector = self.text_embeddings.embed_query(query)
        return self._search_documents(vector, using="text", limit=limit, filters=filters)

    def search_visual_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = self.clip_embeddings.embed_query(query)
        return self._search(vector, using="image", limit=limit, filters=filters)

    def search_image(self, path: Path, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = embed_image(path)
        return self._search(vector, using="image", limit=limit, filters=filters)

    def get_by_sku(self, sku: str) -> ProductHit | None:
        found, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="sku", match=MatchValue(value=sku)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not found:
            return None
        return hit_from_payload(found[0].payload or {}, score=None)

    def find_product(self, key: str) -> ProductHit | None:
        seen: set[str] = set()
        for candidate in _lookup_keys(key):
            if candidate in seen:
                continue
            seen.add(candidate)
            found = self.get_by_sku(candidate) or self._by_part_number(candidate)
            if found is not None:
                return found
        return None

    def _by_part_number(self, part_number: str) -> ProductHit | None:
        found, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="part_number", match=MatchValue(value=part_number)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not found:
            return None
        return hit_from_payload(found[0].payload or {}, score=None)

    def list_products(self, limit: int = 20, filters: ProductFilter | None = None) -> list[ProductHit]:
        must: list = [FieldCondition(key="chunk_index", match=MatchValue(value=0))]
        extra = build_filter(filters)
        if extra is not None:
            must.extend(extra.must or [])
        found, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [hit_from_payload(point.payload or {}, score=None) for point in found]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        counted = self.client.count(
            self.collection,
            count_filter=Filter(must=[FieldCondition(key="chunk_index", match=MatchValue(value=0))]),
            exact=True,
        )
        return counted.count

    def _search(
        self,
        vector: list[float],
        using: str,
        limit: int,
        filters: ProductFilter | None,
    ) -> list[ProductHit]:
        return [hit_from_document(doc) for doc in self._search_documents(vector, using, limit, filters)]

    def _search_documents(
        self,
        vector: list[float],
        using: str,
        limit: int,
        filters: ProductFilter | None,
    ) -> list[Document]:
        found = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=using,
            query_filter=build_filter(filters),
            limit=max(limit * 4, limit),
            with_payload=True,
        )
        documents = [
            document_from_point(point.payload or {}, score=point.score)
            for point in found.points
        ]
        return _dedupe(documents, limit)

    def _delete_skus(self, skus: list[str]) -> None:
        if not skus:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="sku", match=MatchAny(any=skus))])
            ),
        )
        self.client.delete(
            collection_name=self.collection,
            points_selector=[chunk_id(sku, 0) for sku in skus] + [point_id(sku) for sku in skus],
        )

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


class ProductRetriever(BaseRetriever):
    """LangChain retriever over product text chunks, collapsed to one hit per SKU."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    store: Any
    k: int = 4
    filters: ProductFilter | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.store.relevant_documents(query, limit=self.k, filters=self.filters)


def _lookup_keys(key: str) -> list[str]:
    raw = key.strip()
    if not raw:
        return []
    parts = [raw, raw.upper(), *raw.replace(",", " ").split()]
    parts.extend(part.upper() for part in raw.replace(",", " ").split())
    return [part for part in parts if part]


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


def _dedupe(documents: list[Document], limit: int) -> list[Document]:
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
