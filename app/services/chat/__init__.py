"""Catalog chat.

With a Qwen key, the model chooses catalog tools. Otherwise a small set of
rules answers from the same store.
"""

from app.core.config import settings
from app.schemas.chat import ChatResult, ChatTurn
from app.schemas.product import ProductFilter
from app.services.chat.agent import answer_with_tools
from app.services.chat.rules import answer_by_rules
from app.services.store import ProductStore


def answer(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None = None,
    filters: ProductFilter | None = None,
    limit: int = 4,
) -> ChatResult:
    if settings.dashscope_api_key:
        try:
            return answer_with_tools(store, message, history, filters, limit)
        except Exception:
            return answer_by_rules(store, message, history, filters, limit)
    return answer_by_rules(store, message, history, filters, limit)
