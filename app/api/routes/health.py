from fastapi import APIRouter, Depends

from app.api.deps import get_store
from app.services.store import ProductStore

router = APIRouter()


@router.get("/health")
def health(store: ProductStore = Depends(get_store)) -> dict:
    return {"ok": True, "collection": store.collection, "count": store.count()}
