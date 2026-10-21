"""Catalog chat. The model chooses catalog tools; a DashScope key is required."""

from app.core.config import settings
from app.core.logging import event
from app.schemas.chat import ChatResult, ChatTurn
from app.schemas.product import ProductFilter
from app.services.chat.agent import answer_with_tools
from app.services.chat.sessions import get_session_store
from app.services.store import ProductReader


def answer(
    reader: ProductReader,
    message: str,
    history: list[ChatTurn] | None = None,
    filters: ProductFilter | None = None,
    limit: int = 4,
    session_id: str | None = None,
) -> ChatResult:
    if not settings.dashscope_api_key:
        event("chat.route", path="blocked", reason="no_api_key")
        return ChatResult(
            answer="未配置 DASHSCOPE_API_KEY，无法对话。请在 .env 中填写通义千问密钥后重启服务。",
            products=[],
            chunks=[],
            llm=None,
            session_id=session_id or "",
        )

    sessions = get_session_store()
    sid = session_id or sessions.create()
    stored = sessions.get(sid)
    # Prefer server history; accept client history only for a brand-new session.
    turns = stored if stored else list(history or [])

    event("chat.route", path="agent", session=sid, history=len(turns))
    result = answer_with_tools(reader, message, turns, filters, limit)
    if result.llm:
        turns = [*turns, ChatTurn(role="user", content=message), ChatTurn(role="assistant", content=result.answer)]
        sessions.save(sid, turns)
    return result.model_copy(update={"session_id": sid})
