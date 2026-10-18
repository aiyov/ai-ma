from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.catalog import Product

_splitter: RecursiveCharacterTextSplitter | None = None


def split_product(product: Product, metadata: dict) -> list[Document]:
    """Split a product write-up into overlapping chunks.

    Each chunk keeps the product metadata. Later chunks repeat the title so a
    specification fragment can still be found by the product name.
    """
    source = Document(page_content=product.document, metadata=dict(metadata))
    chunks = text_splitter().split_documents([source]) or [source]
    documents: list[Document] = []
    for index, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if index and product.title and not content.startswith(product.title):
            content = f"{product.title}\n{content}"
        documents.append(
            Document(
                page_content=content,
                metadata={**chunk.metadata, "chunk_index": index},
            )
        )
    return documents


def text_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter
