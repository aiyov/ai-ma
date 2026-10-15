from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_store
from app.schemas.product import ProductFilter, SearchRequest, SearchResponse
from app.services.store import ProductStore

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(body: SearchRequest, store: ProductStore = Depends(get_store)) -> SearchResponse:
    if body.mode == "text":
        products = store.search_text(body.query, limit=body.limit, filters=body.filters)
    elif body.mode == "visual":
        products = store.search_visual_text(body.query, limit=body.limit, filters=body.filters)
    else:
        raise HTTPException(status_code=400, detail="mode must be text or visual")
    return SearchResponse(products=products)


@router.post("/image", response_model=SearchResponse)
def search_image(
    file: UploadFile = File(...),
    limit: int = Form(5),
    brand: str | None = Form(None),
    category: str | None = Form(None),
    condition: str | None = Form(None),
    in_stock: bool | None = Form(None),
    min_price: float | None = Form(None),
    max_price: float | None = Form(None),
    store: ProductStore = Depends(get_store),
) -> SearchResponse:
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 20")
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(file.file.read())
        image_path = Path(temporary.name)
    filters = ProductFilter(
        brand=brand,
        category=category,
        condition=condition,
        in_stock=in_stock,
        min_price=min_price,
        max_price=max_price,
    )
    try:
        products = store.search_image(image_path, limit=limit, filters=filters)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}") from exc
    finally:
        image_path.unlink(missing_ok=True)
    return SearchResponse(products=products)
