from pydantic import BaseModel, Field

from app.schemas.product import ProductFilter, ProductHit


class ChatTurn(BaseModel):
    role: str
    content: str


class RetrievedChunk(BaseModel):
    sku: str
    title: str
    chunk_index: int = 0
    score: float | None = None
    text: str


class ChatResult(BaseModel):
    answer: str
    products: list[ProductHit]
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    llm: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    limit: int = Field(default=4, ge=1, le=8)
    filters: ProductFilter = Field(default_factory=ProductFilter)
