from pydantic import BaseModel, Field


class ProductFilter(BaseModel):
    brand: str | None = None
    category: str | None = None
    condition: str | None = None
    in_stock: bool | None = None
    min_price: float | None = None
    max_price: float | None = None


class ProductHit(BaseModel):
    sku: str
    product_id: str = ""
    title: str
    brand: str = ""
    category: str = ""
    category_path: list[str] = Field(default_factory=list)
    condition: str = ""
    part_number: str = ""
    style: str = ""
    heading: str = ""
    sale_price: float | None = None
    msrp: float | None = None
    in_stock: bool = False
    rating: float | None = None
    rating_count: int = 0
    image_url: str = ""
    thumb_url: str = ""
    sku_url: str = ""
    score: float | None = None
    document: str = ""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "text"
    limit: int = Field(default=5, ge=1, le=20)
    filters: ProductFilter = Field(default_factory=ProductFilter)


class SearchResponse(BaseModel):
    products: list[ProductHit]


class ImageFailure(BaseModel):
    sku: str
    url: str = ""
    reason: str


class ImportResponse(BaseModel):
    products: int
    chunks: int
    count: int
    failed_images: list[ImageFailure] = Field(default_factory=list)


class SyncResponse(BaseModel):
    scanned: int
    changed: int
    upserted: int
    skipped: int
    chunks: int
    count: int
    watermark: str | None = None
    failed_images: list[ImageFailure] = Field(default_factory=list)
    full: bool = False
