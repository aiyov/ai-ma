"""Catalog tools the chat model can call.

Methods are plain callables so they can be unit-tested without the LLM.
StructuredTool wraps them for bind_tools.
"""

import json

from langchain_core.tools import StructuredTool

from app.schemas.chat import RetrievedChunk
from app.schemas.product import ProductFilter, ProductHit
from app.services.chat.text import chunk_from_hit, dump_products
from app.services.store import ProductReader


class CatalogSession:
    """One chat turn's catalog access and the products that were actually used."""

    def __init__(self, reader: ProductReader, filters: ProductFilter | None, limit: int) -> None:
        self.reader = reader
        self.filters = filters or ProductFilter()
        self.limit = limit
        self.products: list[ProductHit] = []
        self.chunks: list[RetrievedChunk] = []
        self._seen: set[str] = set()

    def remember(self, hit: ProductHit, text: str, chunk_index: int = 0, score: float | None = None) -> None:
        if hit.sku not in self._seen:
            self._seen.add(hit.sku)
            self.products.append(hit)
        self.chunks.append(
            RetrievedChunk(
                sku=hit.sku,
                title=hit.title,
                chunk_index=chunk_index,
                score=score,
                text=text,
            )
        )

    def narrowed(self, brand: str, max_price: float, in_stock: bool) -> ProductFilter:
        data = self.filters.model_dump()
        if not data.get("brand") and brand.strip():
            data["brand"] = brand.strip()
        if data.get("max_price") is None and max_price > 0:
            data["max_price"] = max_price
        if data.get("in_stock") is None and in_stock:
            data["in_stock"] = True
        return ProductFilter(**data)

    def search_products(self, query: str, brand: str = "", max_price: float = 0, in_stock: bool = False) -> str:
        """Search the catalog by a description of the instrument the customer wants.

        query: words for the instrument, material, or feature.
        brand: brand name, or an empty string when none was given.
        max_price: highest sale price, or 0 when there is no limit.
        in_stock: true only when the customer asked for items in stock.
        """
        hits = self.reader.search_text(
            query,
            limit=self.limit,
            filters=self.narrowed(brand, max_price, in_stock),
        )
        for hit in hits:
            self.remember(hit, hit.document or hit.title, score=hit.score)
        return dump_products(hits, lambda hit: hit.document or hit.title)

    def list_products(self, brand: str = "", max_price: float = 0, in_stock: bool = False) -> str:
        """List models in the catalog when the customer asks what exists.

        brand: brand name, or an empty string for every brand.
        max_price: highest sale price, or 0 when there is no limit.
        in_stock: true only when the customer asked for items in stock.
        """
        hits = self.reader.list_products(limit=20, filters=self.narrowed(brand, max_price, in_stock))
        for hit in hits:
            self.remember(hit, chunk_from_hit(hit).text)
        return dump_products(hits, lambda hit: hit.heading or hit.style or hit.title)

    def get_product(self, sku_or_part_number: str) -> str:
        """Read one product by SKU or part number, for example GFL5G."""
        hit = self.reader.find_product(sku_or_part_number)
        if hit is None:
            return json.dumps({"found": False, "query": sku_or_part_number}, ensure_ascii=False)
        excerpt = (hit.document or hit.title)[:1800]
        self.remember(hit, excerpt[:500] or hit.title)
        return dump_products([hit], lambda item: (item.document or item.title)[:1800])

    def as_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(self.search_products),
            StructuredTool.from_function(self.list_products),
            StructuredTool.from_function(self.get_product),
        ]


# Backward-compatible alias used by older imports in this package.
Catalog = CatalogSession
