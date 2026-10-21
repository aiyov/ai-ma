"""Read path: vector search, list, and exact lookup."""

from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.core.logging import timed
from app.schemas.product import ProductFilter, ProductHit
from app.services.embeddings import ClipTextEmbeddings, FastEmbedTextEmbeddings, embed_image
from app.services.store.mapping import (
    build_filter,
    dedupe,
    document_from_point,
    hit_from_document,
    hit_from_payload,
    lookup_keys,
)


class ProductReader:
    def __init__(
        self,
        client: QdrantClient,
        text_embeddings: FastEmbedTextEmbeddings | None = None,
        clip_embeddings: ClipTextEmbeddings | None = None,
        collection: str | None = None,
    ) -> None:
        self.client = client
        self.text_embeddings = text_embeddings or FastEmbedTextEmbeddings()
        self.clip_embeddings = clip_embeddings or ClipTextEmbeddings()
        self.collection = collection or settings.collection

    def search_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        return [hit_from_document(doc) for doc in self.relevant_documents(query, limit, filters)]

    def relevant_documents(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[Document]:
        vector = self.text_embeddings.embed_query(query)
        return self._search_documents(vector, using="text", limit=limit, filters=filters, query=query)

    def search_visual_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = self.clip_embeddings.embed_query(query)
        return self._search(vector, using="image", limit=limit, filters=filters, query=query)

    def search_image(self, path: Path, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        vector = embed_image(path)
        return self._search(vector, using="image", limit=limit, filters=filters, query=path.name)

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
        for candidate in lookup_keys(key):
            if candidate in seen:
                continue
            seen.add(candidate)
            found = self.get_by_sku(candidate) or self._by_part_number(candidate)
            if found is not None:
                return found
        return None

    def list_products(self, limit: int = 20, filters: ProductFilter | None = None) -> list[ProductHit]:
        must: list = [FieldCondition(key="chunk_index", match=MatchValue(value=0))]
        extra = build_filter(filters)
        if extra is not None:
            must.extend(extra.must or [])
        with timed("store.list", limit=limit) as meta:
            found, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=must),
                limit=limit,
                with_payload=True,
            )
            hits = [hit_from_payload(point.payload or {}, score=None) for point in found]
            meta["hits"] = len(hits)
            meta["skus"] = [hit.sku for hit in hits]
            return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        counted = self.client.count(
            self.collection,
            count_filter=Filter(must=[FieldCondition(key="chunk_index", match=MatchValue(value=0))]),
            exact=True,
        )
        return counted.count

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

    def _search(
        self,
        vector: list[float],
        using: str,
        limit: int,
        filters: ProductFilter | None,
        query: str = "",
    ) -> list[ProductHit]:
        return [
            hit_from_document(doc)
            for doc in self._search_documents(vector, using, limit, filters, query=query)
        ]

    def _search_documents(
        self,
        vector: list[float],
        using: str,
        limit: int,
        filters: ProductFilter | None,
        query: str = "",
    ) -> list[Document]:
        with timed("store.search", using=using, limit=limit, query=query) as meta:
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
            chosen = dedupe(documents, limit)
            meta["hits"] = len(chosen)
            meta["skus"] = [str(doc.metadata.get("sku") or "") for doc in chosen]
            meta["scores"] = [doc.metadata.get("score") for doc in chosen]
            return chosen
