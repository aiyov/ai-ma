from fastapi import APIRouter, Depends

from app.api.deps import get_reader
from app.services.store import ProductReader

router = APIRouter(tags=["health"])


@router.get("/health")
def health(reader: ProductReader = Depends(get_reader)) -> dict:
    return {"ok": True, "collection": reader.collection, "count": reader.count()}
