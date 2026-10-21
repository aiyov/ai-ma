from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.services.embeddings import ClipTextEmbeddings, FastEmbedTextEmbeddings
from app.services.store import ProductReader, ProductStore, ProductWriter, make_client

_client = None
_text_embeddings: FastEmbedTextEmbeddings | None = None
_clip_embeddings: ClipTextEmbeddings | None = None
_writer: ProductWriter | None = None
_reader: ProductReader | None = None
_store: ProductStore | None = None
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _shared_client():
    global _client
    if _client is None:
        _client = make_client()
    return _client


def _shared_text() -> FastEmbedTextEmbeddings:
    global _text_embeddings
    if _text_embeddings is None:
        _text_embeddings = FastEmbedTextEmbeddings()
    return _text_embeddings


def _shared_clip() -> ClipTextEmbeddings:
    global _clip_embeddings
    if _clip_embeddings is None:
        _clip_embeddings = ClipTextEmbeddings()
    return _clip_embeddings


def get_writer() -> ProductWriter:
    global _writer
    if _writer is None:
        _writer = ProductWriter(_shared_client(), _shared_text())
    return _writer


def get_reader() -> ProductReader:
    global _reader
    if _reader is None:
        _reader = ProductReader(_shared_client(), _shared_text(), _shared_clip())
    return _reader


def get_store() -> ProductStore:
    """Importer-facing facade sharing the same client and models."""
    global _store
    if _store is None:
        _store = ProductStore(
            client=_shared_client(),
            text_embeddings=_shared_text(),
            clip_embeddings=_shared_clip(),
        )
    return _store


def set_store(store: ProductStore | None) -> None:
    """Replace process singletons. Used by tests."""
    global _store, _writer, _reader, _client, _text_embeddings, _clip_embeddings
    _store = store
    if store is None:
        _writer = None
        _reader = None
        _client = None
        _text_embeddings = None
        _clip_embeddings = None
        return
    _client = store.client
    _text_embeddings = store.text_embeddings
    _clip_embeddings = store.clip_embeddings
    _writer = store.writer
    _reader = store.reader


def require_api_key(key: str | None = Depends(_api_key_header)) -> None:
    expected = settings.api_key
    if not expected:
        return
    if key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")
