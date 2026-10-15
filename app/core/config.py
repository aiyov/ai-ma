import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parents[2]
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("QDRANT_COLLECTION", "products")
    data_path: Path = Path(os.getenv("DATA_PATH", "data.json"))
    model_cache: Path = Path(os.getenv("MODEL_CACHE", ".cache/fastembed"))
    image_cache: Path = Path(os.getenv("IMAGE_CACHE", ".cache/images"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    text_model: str = "BAAI/bge-small-en-v1.5"
    text_dim: int = 384
    clip_text_model: str = "Qdrant/clip-ViT-B-32-text"
    image_model: str = "Qdrant/clip-ViT-B-32-vision"
    image_dim: int = 512


settings = Settings()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return (settings.root / path).resolve()
