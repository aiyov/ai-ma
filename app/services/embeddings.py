import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.core.config import settings

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class FastEmbedTextEmbeddings(Embeddings):
    """BGE text embeddings. Queries use the model's query prefix."""

    def __init__(self) -> None:
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed([text]))).tolist()

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            settings.model_cache.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=settings.text_model,
                cache_dir=str(settings.model_cache),
            )
        return self._model


class ClipTextEmbeddings(Embeddings):
    """CLIP text encoder. Its vectors live in the same space as product photos."""

    def __init__(self) -> None:
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            settings.model_cache.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=settings.clip_text_model,
                cache_dir=str(settings.model_cache),
            )
        return self._model


def embed_image(path: Path) -> list[float]:
    return next(iter(_image_model().embed([str(path)]))).tolist()


_image = None


def _image_model():
    global _image
    if _image is None:
        from fastembed import ImageEmbedding

        settings.model_cache.mkdir(parents=True, exist_ok=True)
        _image = ImageEmbedding(
            model_name=settings.image_model,
            cache_dir=str(settings.model_cache),
        )
    return _image
