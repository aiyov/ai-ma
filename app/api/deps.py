from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.services.store import ProductStore

store = ProductStore()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_store() -> ProductStore:
    return store


def require_api_key(key: str | None = Depends(_api_key_header)) -> None:
    expected = settings.api_key
    if not expected:
        return
    if key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")
