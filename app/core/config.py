import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parents[2]
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("QDRANT_COLLECTION", "products")
    data_path: Path = Path(os.getenv("DATA_PATH", "data.json"))
    model_cache: Path = Path(os.getenv("MODEL_CACHE", ".cache/fastembed"))
    image_cache: Path = Path(os.getenv("IMAGE_CACHE", ".cache/images"))
    api_key: str = os.getenv("API_KEY", "")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    qwen_timeout: float = float(os.getenv("QWEN_TIMEOUT", "20"))
    qwen_max_retries: int = int(os.getenv("QWEN_MAX_RETRIES", "2"))
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    text_model: str = "BAAI/bge-small-en-v1.5"
    text_dim: int = 384
    clip_text_model: str = "Qdrant/clip-ViT-B-32-text"
    image_model: str = "Qdrant/clip-ViT-B-32-vision"
    image_dim: int = 512
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    redis_url: str = os.getenv(
        "REDIS_URL",
        f"redis://:{os.getenv('REDIS_PASSWORD', 'change-me-local-dev')}@localhost:6379/0",
    )
    chat_session_ttl: int = int(os.getenv("CHAT_SESSION_TTL", "86400"))
    sync_state_path: Path = Path(os.getenv("SYNC_STATE_PATH", ".cache/sync_state.sqlite"))
    sync_source: str = os.getenv("SYNC_SOURCE", "file")
    sync_interval_seconds: int = int(os.getenv("SYNC_INTERVAL_SECONDS", "3600"))


settings = Settings()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return (settings.root / path).resolve()
