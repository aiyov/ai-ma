"""Tool-calling loop. The model decides whether to search, list, or read one product."""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.core.config import settings
from app.core.logging import event, timed
from app.schemas.chat import ChatResult, ChatTurn
from app.schemas.product import ProductFilter
from app.services.chat import prompts
from app.services.chat.llm import history_messages, message_text, model
from app.services.chat.text import search_fallback
from app.services.chat.tools import CatalogSession
from app.services.store import ProductReader


def answer_with_tools(
    reader: ProductReader,
    message: str,
    history: list[ChatTurn] | None,
    filters: ProductFilter | None,
    limit: int,
) -> ChatResult:
    catalog = CatalogSession(reader, filters, limit)
    tools = catalog.as_tools()
    by_name = {item.name: item for item in tools}
    messages: list = [
        SystemMessage(content=prompts.with_language(prompts.AGENT, message)),
        *history_messages(history),
        HumanMessage(content=message),
    ]
    tool_model = model().bind_tools(tools)
    plain = model()
    with timed("chat.agent", query=message, limit=limit) as meta:
        try:
            for step in range(2):
                ai = tool_model.invoke(messages)
                messages.append(ai)
                calls = getattr(ai, "tool_calls", None) or []
                if not calls:
                    meta["steps"] = step + 1
                    meta["tools"] = len(catalog.chunks)
                    meta["products"] = len(catalog.products)
                    return _result(message_text(ai), catalog)
                for call in calls:
                    messages.append(_run_tool(by_name, call))
            meta["steps"] = 2
            meta["tools"] = len(catalog.chunks)
            meta["products"] = len(catalog.products)
            return _result(message_text(plain.invoke(messages)), catalog)
        except Exception as exc:
            event("chat.agent.error", error=f"{exc.__class__.__name__}:{exc}")
            return ChatResult(
                answer=f"对话服务暂时不可用（{exc.__class__.__name__}: {exc}）。",
                products=[],
                chunks=[],
                llm=None,
            )


def _run_tool(by_name: dict, call: dict) -> ToolMessage:
    name = str(call.get("name") or "")
    args = call.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    selected = by_name.get(name)
    with timed("chat.tool", tool=name, args=_args_summary(args if isinstance(args, dict) else {})) as meta:
        if selected is None:
            content = json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
            meta["ok"] = False
        else:
            try:
                content = str(selected.invoke(args if isinstance(args, dict) else {}))
                meta["ok"] = True
                meta["bytes"] = len(content)
            except Exception as exc:
                content = json.dumps({"error": f"{exc.__class__.__name__}: {exc}"}, ensure_ascii=False)
                meta["ok"] = False
                meta["error"] = exc.__class__.__name__
    return ToolMessage(content=content, tool_call_id=str(call.get("id") or "call"))


def _args_summary(args: dict) -> str:
    keys = ("query", "sku_or_part_number", "brand", "max_price", "in_stock")
    parts = [f"{key}={args[key]}" for key in keys if args.get(key) not in (None, "", 0, False)]
    return ";".join(parts)[:120]


def _result(text: str, catalog: CatalogSession) -> ChatResult:
    if not text:
        text = search_fallback(catalog.products) if catalog.products else "我没有查到可以回答的内容。"
    return ChatResult(
        answer=text,
        products=catalog.products,
        chunks=catalog.chunks,
        llm=settings.qwen_model,
    )
