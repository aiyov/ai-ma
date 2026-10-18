from fastapi import APIRouter, Depends

from app.api.deps import get_store, require_api_key
from app.schemas.chat import ChatRequest, ChatResult
from app.services.chat import answer
from app.services.store import ProductStore

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResult)
def chat(body: ChatRequest, store: ProductStore = Depends(get_store)) -> ChatResult:
    return answer(
        store,
        message=body.message,
        history=body.history,
        filters=body.filters,
        limit=body.limit,
    )
