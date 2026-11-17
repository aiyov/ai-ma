"""对话模型可调用的商品目录工具。

方法本身是普通可调用对象，方便不经过 LLM 做单元测试。
as_tools() 再用 StructuredTool 包一层，供 bind_tools 使用。
"""

import json

from langchain_core.tools import StructuredTool

from app.schemas.chat import RetrievedChunk
from app.schemas.product import ProductFilter, ProductHit
from app.services.chat.text import chunk_from_hit, dump_products
from app.services.store import ProductReader


class CatalogSession:
    """一次对话轮次里的商品目录访问器。

    持有 ProductReader，并把本轮实际用到的商品记到 products / chunks，
    方便接口把「引用了哪些商品」回传给前端。
    """

    def __init__(self, reader: ProductReader, filters: ProductFilter | None, limit: int) -> None:
        # reader：向量库读接口；filters / limit：本轮检索的默认约束
        self.reader = reader
        self.filters = filters or ProductFilter()
        self.limit = limit
        self.products: list[ProductHit] = []  # 本轮去重后的命中商品
        self.chunks: list[RetrievedChunk] = []  # 本轮引用过的文本片段（可含同一 SKU 多次）
        self._seen: set[str] = set()  # 已记入 products 的 SKU，避免重复

    def remember(self, hit: ProductHit, text: str, chunk_index: int = 0, score: float | None = None) -> None:
        """记下本轮用到的商品与文本片段。

        products 按 SKU 去重；chunks 每次调用都追加，供前端展示引用出处。
        """
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
        """把工具参数叠到会话默认 filters 上。

        只在请求侧尚未指定时才用工具参数补全，避免覆盖用户/前端已有的过滤条件。
        """
        data = self.filters.model_dump()
        if not data.get("brand") and brand.strip():
            data["brand"] = brand.strip()
        if data.get("max_price") is None and max_price > 0:
            data["max_price"] = max_price
        if data.get("in_stock") is None and in_stock:
            data["in_stock"] = True
        return ProductFilter(**data)

    def search_products(self, query: str, brand: str = "", max_price: float = 0, in_stock: bool = False) -> str:
        """按文字描述检索商品（BGE / text 向量）。

        给模型看的 docstring 须保留英文参数说明，StructuredTool 会拿去当工具描述。

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
        """列出目录里有哪些型号（不做向量相似度，走 scroll）。

        适合「有哪些型号 / 有什么货」这类问题。

        brand: brand name, or an empty string for every brand.
        max_price: highest sale price, or 0 when there is no limit.
        in_stock: true only when the customer asked for items in stock.
        """
        hits = self.reader.list_products(limit=20, filters=self.narrowed(brand, max_price, in_stock))
        for hit in hits:
            self.remember(hit, chunk_from_hit(hit).text)
        return dump_products(hits, lambda hit: hit.heading or hit.style or hit.title)

    def get_product(self, sku_or_part_number: str) -> str:
        """按 SKU 或型号精确读一件商品，例如 GFL5G。

        sku_or_part_number: SKU or part number to look up.
        """
        hit = self.reader.find_product(sku_or_part_number)
        if hit is None:
            return json.dumps({"found": False, "query": sku_or_part_number}, ensure_ascii=False)
        excerpt = (hit.document or hit.title)[:1800]
        self.remember(hit, excerpt[:500] or hit.title)
        return dump_products([hit], lambda item: (item.document or item.title)[:1800])

    def as_tools(self) -> list[StructuredTool]:
        """把上述三个方法包装成 LangChain StructuredTool，供模型 bind_tools。"""
        return [
            StructuredTool.from_function(self.search_products),
            StructuredTool.from_function(self.list_products),
            StructuredTool.from_function(self.get_product),
        ]


# 旧导入名兼容别名
Catalog = CatalogSession
