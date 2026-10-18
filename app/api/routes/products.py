from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_store, require_api_key
from app.core.config import resolve_path, settings
from app.schemas.product import ImportResponse, ProductHit
from app.services.importer import import_file
from app.services.store import ProductStore

router = APIRouter(prefix="/products", tags=["products"], dependencies=[Depends(require_api_key)])


@router.post("/import", response_model=ImportResponse)
def import_products(
    path: str | None = None,
    store: ProductStore = Depends(get_store),
) -> ImportResponse:
    data_path = _data_path(path)
    if not data_path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {data_path.name}")
    result = import_file(data_path, store)
    return result


@router.get("/{sku}", response_model=ProductHit)
def get_product(sku: str, store: ProductStore = Depends(get_store)) -> ProductHit:
    product = store.get_by_sku(sku)
    if product is None:
        raise HTTPException(status_code=404, detail="product not found")
    return product


def _data_path(path: str | None) -> Path:
    candidate = resolve_path(Path(path) if path else settings.data_path)
    root = settings.root.resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail="path must stay inside the project")
    return candidate
