"""Write path: collection schema, delete-by-sku, and chunk upsert."""

from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny, PointStruct

from app.core.config import settings
from app.core.logging import timed
from app.services.catalog import Product
from app.services.documents import split_product
from app.services.embeddings import FastEmbedTextEmbeddings, embed_image
from app.services.store.mapping import chunk_id, payload_of, point_id
from app.services.store.schema import PAYLOAD_INDEXES, expected_vector_sizes, vectors_config


class ProductWriter:
    def __init__(
        self,
        client: QdrantClient,
        text_embeddings: FastEmbedTextEmbeddings | None = None,
        collection: str | None = None,
    ) -> None:
        self.client = client
        self.text_embeddings = text_embeddings or FastEmbedTextEmbeddings()
        self.collection = collection or settings.collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._check_schema()
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config(),
            )
        info = self.client.get_collection(self.collection)
        existing = set(info.payload_schema)
        for field_name, schema in PAYLOAD_INDEXES.items():
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
        with timed("store.upsert", products=len(products)) as meta:
            self.delete_skus([product.sku for product in products])
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
            meta["chunks"] = len(points)
            return len(points)

    def delete_skus(self, skus: list[str]) -> None:
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
        expected = expected_vector_sizes()
        actual = {name: params.size for name, params in vectors.items()} if isinstance(vectors, dict) else {}
        if actual != expected:
            raise RuntimeError(
                f"Collection {self.collection!r} has vectors {actual}, expected {expected}. "
                "Drop it in the Qdrant dashboard and import again."
            )
