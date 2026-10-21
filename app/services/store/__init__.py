"""Product index over Qdrant.

Write and read are separate modules. ProductStore remains a convenience facade
for the importer CLI; HTTP routes should depend on ProductWriter or ProductReader.
"""

from qdrant_client import QdrantClient

from app.core.config import settings
from app.services.embeddings import ClipTextEmbeddings, FastEmbedTextEmbeddings
from app.services.store.mapping import hit_from_document, hit_from_payload, payload_of
from app.services.store.reader import ProductReader
from app.services.store.writer import ProductWriter


def make_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or settings.qdrant_url, timeout=60, trust_env=False)


class ProductStore:
    """Facade used by the importer. Prefer ProductReader / ProductWriter in routes."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        text_embeddings: FastEmbedTextEmbeddings | None = None,
        clip_embeddings: ClipTextEmbeddings | None = None,
        collection: str | None = None,
    ) -> None:
        self.client = client or make_client()
        self.text_embeddings = text_embeddings or FastEmbedTextEmbeddings()
        self.clip_embeddings = clip_embeddings or ClipTextEmbeddings()
        self.collection = collection or settings.collection
        self.writer = ProductWriter(self.client, self.text_embeddings, self.collection)
        self.reader = ProductReader(
            self.client,
            self.text_embeddings,
            self.clip_embeddings,
            self.collection,
        )

    def ensure_collection(self) -> None:
        self.writer.ensure_collection()

    def upsert(self, *args, **kwargs):
        return self.writer.upsert(*args, **kwargs)

    def search_text(self, *args, **kwargs):
        return self.reader.search_text(*args, **kwargs)

    def search_visual_text(self, *args, **kwargs):
        return self.reader.search_visual_text(*args, **kwargs)

    def search_image(self, *args, **kwargs):
        return self.reader.search_image(*args, **kwargs)

    def get_by_sku(self, *args, **kwargs):
        return self.reader.get_by_sku(*args, **kwargs)

    def find_product(self, *args, **kwargs):
        return self.reader.find_product(*args, **kwargs)

    def list_products(self, *args, **kwargs):
        return self.reader.list_products(*args, **kwargs)

    def count(self) -> int:
        return self.reader.count()


__all__ = [
    "ProductReader",
    "ProductStore",
    "ProductWriter",
    "hit_from_document",
    "hit_from_payload",
    "make_client",
    "payload_of",
]
