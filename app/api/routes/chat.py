from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_reader, require_api_key
from app.schemas.chat import ChatRequest, ChatResult
from app.services.chat import answer
from app.services.store import ProductReader

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResult)
async def chat(body: ChatRequest, reader: ProductReader = Depends(get_reader)) -> ChatResult:
    return await run_in_threadpool(
        answer,
        reader,
        body.message,
        body.history,
        body.filters,
        body.limit,
        body.session_id,
    )
