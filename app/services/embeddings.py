import os
from pathlib import Path

from app.core.config import settings

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class Embedder:
    """Lazy local models. Text search and image search do not share a vector space."""

    def __init__(self) -> None:
        self._text = None
        self._clip_text = None
        self._image = None
        self.cache_dir = settings.model_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._text_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._text_model().query_embed([text]))).tolist()

    def embed_visual_query(self, text: str) -> list[float]:
        return next(iter(self._clip_text_model().embed([text]))).tolist()

    def embed_image(self, path: Path) -> list[float]:
        return next(iter(self._image_model().embed([str(path)]))).tolist()

    def _text_model(self):
        if self._text is None:
            from fastembed import TextEmbedding

            self._text = TextEmbedding(
                model_name=settings.text_model,
                cache_dir=str(self.cache_dir),
            )
        return self._text

    def _clip_text_model(self):
        if self._clip_text is None:
            from fastembed import TextEmbedding

            self._clip_text = TextEmbedding(
                model_name=settings.clip_text_model,
                cache_dir=str(self.cache_dir),
            )
        return self._clip_text

    def _image_model(self):
        if self._image is None:
            from fastembed import ImageEmbedding

            self._image = ImageEmbedding(
                model_name=settings.image_model,
                cache_dir=str(self.cache_dir),
            )
        return self._image
