"""Tool-calling loop. The model decides whether to search, list, or read one product."""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.core.config import settings
from app.schemas.chat import ChatResult, ChatTurn
from app.schemas.product import ProductFilter
from app.services.chat import prompts
from app.services.chat.llm import history_messages, message_text, model
from app.services.chat.text import search_fallback
from app.services.chat.tools import Catalog
from app.services.store import ProductStore


def answer_with_tools(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None,
    filters: ProductFilter | None,
    limit: int,
) -> ChatResult:
    catalog = Catalog(store, filters, limit)
    tools = catalog.as_tools()
    by_name = {item.name: item for item in tools}
    messages: list = [
        SystemMessage(content=prompts.with_language(prompts.AGENT, message)),
        *history_messages(history),
        HumanMessage(content=message),
    ]
    tool_model = model().bind_tools(tools)
    plain = model()
    for _ in range(2):
        ai = tool_model.invoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return _result(message_text(ai), catalog)
        for call in calls:
            messages.append(_run_tool(by_name, call))
    return _result(message_text(plain.invoke(messages)), catalog)


def _run_tool(by_name: dict, call: dict) -> ToolMessage:
    name = str(call.get("name") or "")
    args = call.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    selected = by_name.get(name)
    if selected is None:
        content = json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
    else:
        try:
            content = str(selected.invoke(args if isinstance(args, dict) else {}))
        except Exception as exc:
            content = json.dumps({"error": f"{exc.__class__.__name__}: {exc}"}, ensure_ascii=False)
    return ToolMessage(content=content, tool_call_id=str(call.get("id") or "call"))


def _result(text: str, catalog: Catalog) -> ChatResult:
    if not text:
        text = search_fallback(catalog.products) if catalog.products else "我没有查到可以回答的内容。"
    return ChatResult(
        answer=text,
        products=catalog.products,
        chunks=catalog.chunks,
        llm=settings.qwen_model,
    )
