from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_reader, get_store, require_api_key
from app.core.config import resolve_path, settings
from app.schemas.product import ImportResponse, ProductHit, SyncResponse
from app.services.importer import import_file
from app.services.store import ProductReader, ProductStore
from app.services.sync import run_sync

router = APIRouter(prefix="/products", tags=["products"], dependencies=[Depends(require_api_key)])


@router.post("/import", response_model=ImportResponse)
async def import_products(
    path: str | None = None,
    store: ProductStore = Depends(get_store),
) -> ImportResponse:
    data_path = _data_path(path)
    if not data_path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {data_path.name}")
    return await run_in_threadpool(import_file, data_path, store)


@router.post("/sync", response_model=SyncResponse)
async def sync_products(
    full: bool = False,
    store: ProductStore = Depends(get_store),
) -> SyncResponse:
    """Pull changed catalog rows (mock ATG file source today) and upsert only real changes."""
    return await run_in_threadpool(run_sync, full=full, store=store)


@router.get("/{sku}", response_model=ProductHit)
async def get_product(sku: str, reader: ProductReader = Depends(get_reader)) -> ProductHit:
    product = await run_in_threadpool(reader.get_by_sku, sku)
    if product is None:
        raise HTTPException(status_code=404, detail="product not found")
    return product


def _data_path(path: str | None) -> Path:
    candidate = resolve_path(Path(path) if path else settings.data_path)
    root = settings.root.resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail="path must stay inside the project")
    return candidate
