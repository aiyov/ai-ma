from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_reader, require_api_key
from app.schemas.product import ProductFilter, SearchRequest, SearchResponse
from app.services.store import ProductReader

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchResponse)
async def search(body: SearchRequest, reader: ProductReader = Depends(get_reader)) -> SearchResponse:
    if body.mode == "text":
        products = await run_in_threadpool(
            reader.search_text, body.query, body.limit, body.filters
        )
    elif body.mode == "visual":
        products = await run_in_threadpool(
            reader.search_visual_text, body.query, body.limit, body.filters
        )
    else:
        raise HTTPException(status_code=400, detail="mode must be text or visual")
    return SearchResponse(products=products)


@router.post("/image", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    limit: int = Form(5),
    brand: str | None = Form(None),
    category: str | None = Form(None),
    condition: str | None = Form(None),
    in_stock: bool | None = Form(None),
    min_price: float | None = Form(None),
    max_price: float | None = Form(None),
    reader: ProductReader = Depends(get_reader),
) -> SearchResponse:
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 20")
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(await file.read())
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
        products = await run_in_threadpool(reader.search_image, image_path, limit, filters)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}") from exc
    finally:
        image_path.unlink(missing_ok=True)
    return SearchResponse(products=products)
