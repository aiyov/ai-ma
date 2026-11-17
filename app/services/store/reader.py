"""商品库的读路径：向量检索、列目录、按 SKU/型号精确查找。

写入在 writer.py。这里只读 Qdrant，不改集合内容。
"""

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
    """面向检索和对话的只读接口。

    有两套向量，不能混用：
    - text：BGE，对齐商品介绍文字
    - image：CLIP，对齐商品照片（外观描述和上传图都搜它）
    """

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
        """按介绍、规格等文字内容检索。页面「描述」模式和对话里的 search_products 走这里。"""
        return [hit_from_document(doc) for doc in self.relevant_documents(query, limit, filters)]

    def relevant_documents(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[Document]:
        """先把问题编成 BGE 向量，再在 text 向量上做近邻搜索。"""
        vector = self.text_embeddings.embed_query(query)
        return self._search_documents(vector, using="text", limit=limit, filters=filters, query=query)

    def search_visual_text(self, query: str, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        """按外观描述检索，例如「银色长笛」。

        用 CLIP 文本编码器，去搜入库时由商品图生成的 image 向量。
        """
        vector = self.clip_embeddings.embed_query(query)
        return self._search(vector, using="image", limit=limit, filters=filters, query=query)

    def search_image(self, path: Path, limit: int = 5, filters: ProductFilter | None = None) -> list[ProductHit]:
        """上传一张图，找外观相近的商品。图和库里的图都在 CLIP 视觉空间。"""
        vector = embed_image(path)
        return self._search(vector, using="image", limit=limit, filters=filters, query=path.name)

    def get_by_sku(self, sku: str) -> ProductHit | None:
        """按 SKU 精确读取一件商品。

        只取 chunk_index == 0：那是每个 SKU 的第一条记录，payload 里带着完整 document。
        """
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
        """按 SKU 或型号查找，例如 GFL5G、site7sku1599831。

        lookup_keys 会生成大小写等候选；先当 SKU 查，再当 part_number 查。
        对话工具 get_product 用这个。
        """
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
        """列出商品目录，不做向量相似度。

        同样只取 chunk_index == 0，一件商品只出现一次。对话问「有哪些型号」时用。
        """
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
        """商品件数，不是切块后的 point 总数。用 chunk_index == 0 计数。"""
        if not self.client.collection_exists(self.collection):
            return 0
        counted = self.client.count(
            self.collection,
            count_filter=Filter(must=[FieldCondition(key="chunk_index", match=MatchValue(value=0))]),
            exact=True,
        )
        return counted.count

    def _by_part_number(self, part_number: str) -> ProductHit | None:
        """按型号（part_number）精确匹配，例如 TFL-09L-GRB。"""
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
        """向量检索后转成 API 用的 ProductHit。"""
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
        """在指定命名向量上做近邻搜索，再按 SKU 去重。

        多取 limit*4 再去重：同一商品可能有多块文本，先多拿一些，再每个 SKU 只留分最高的一块。
        using 只能是 "text" 或 "image"，必须和传入向量所在的空间一致。
        """
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
