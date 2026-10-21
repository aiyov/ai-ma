from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import get_writer
from app.api.middleware import RequestIdMiddleware
from app.api.routes import chat, health, products, search
from app.core.logging import configure
from app.services.chat.sessions import get_session_store

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure()
    get_session_store()
    get_writer().ensure_collection()
    yield


app = FastAPI(title="Product search", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)
app.add_middleware(RequestIdMiddleware)
app.include_router(health.router)
app.include_router(products.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
