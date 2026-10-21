"""Keyword fallback used when Qwen is not configured or the tool call fails."""

import re
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.core.config import settings
from app.schemas.chat import ChatResult, ChatTurn
from app.schemas.product import ProductFilter
from app.services.chat import prompts
from app.services.chat.llm import history_messages, model
from app.services.chat.text import (
    catalog_context,
    chunk_from_document,
    chunk_from_hit,
    direct_reply,
    format_docs,
    list_fallback,
    search_fallback,
)
from app.services.store import ProductStore, hit_from_document

_SOCIAL = re.compile(
    r"^(?:"
    r"你好|您好|嗨|在吗|早上好|晚上好|中午好|"
    r"你是谁|您是谁|你叫什么|你是什么|你是做什么的|"
    r"你能做什么|你会什么|你能干什么|"
    r"谢谢|感谢|再见|拜拜|"
    r"hi|hello|hey|who are you|what can you do|thanks|thank you|bye"
    r")[\s!！?？.。~～]*$",
    re.IGNORECASE,
)
_PRODUCT = re.compile(
    r"长笛|乐器|钢琴|吉他|笛|琴|价格|多少钱|售价|库存|有货|无货|推荐|品牌|唇板|琴盒|型号|"
    r"便宜|规格|材质|这支|那个|对比|区别|"
    r"flute|piano|guitar|brand|price|stock|grenadilla|giardinelli|tomasi|"
    r"head\s*joint|plush|offset|sku",
    re.IGNORECASE,
)
_LISTING = re.compile(
    r"有哪些|有哪几|哪些型号|什么型号|有什么型号|都有哪些|都有什么|列出|全部商品|所有型号|"
    r"what models|which models|what products|list (?:all|the)",
    re.IGNORECASE,
)
_COMPARE = re.compile(r"区别|对比|比较|哪个好|\bvs\b", re.IGNORECASE)


def answer_by_rules(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None = None,
    filters: ProductFilter | None = None,
    limit: int = 4,
) -> ChatResult:
    if _is_listing(message):
        return _answer_listing(store, message, history, filters)
    if not _needs_products(message):
        return _reply_without_products(message, history)
    return _answer_search(store, message, history, filters, limit)


def _is_listing(message: str) -> bool:
    text = message.strip()
    if _COMPARE.search(text):
        return False
    return bool(_LISTING.search(text))


def _needs_products(message: str) -> bool:
    text = message.strip()
    if _SOCIAL.match(text):
        return False
    if _PRODUCT.search(text):
        return True
    if settings.dashscope_api_key:
        return _model_wants_catalog(text)
    return bool(re.search(r"[A-Za-z]{4,}", text))


def _model_wants_catalog(message: str) -> bool:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.ROUTER),
            ("human", "{question}"),
        ]
    )
    try:
        route = (prompt | model() | StrOutputParser()).invoke({"question": message})
    except Exception:
        return False
    return route.strip().lower().startswith("catalog")


def _reply_without_products(message: str, history: list[ChatTurn] | None) -> ChatResult:
    if not settings.dashscope_api_key:
        return ChatResult(answer=direct_reply(message, _SOCIAL.match), products=[], chunks=[], llm=None)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.with_language(prompts.CHAT, message)),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    try:
        text = (prompt | model() | StrOutputParser()).invoke(
            {"question": message, "history": history_messages(history)}
        )
    except Exception:
        return ChatResult(answer=direct_reply(message, _SOCIAL.match), products=[], chunks=[], llm=None)
    return ChatResult(answer=text, products=[], chunks=[], llm=settings.qwen_model)


def _answer_search(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None,
    filters: ProductFilter | None,
    limit: int,
) -> ChatResult:
    retriever = store.as_retriever(limit=limit, filters=filters)
    prepared = RunnableParallel(
        docs=itemgetter("question") | retriever,
        question=itemgetter("question"),
        history=itemgetter("history"),
    ) | RunnablePassthrough.assign(context=lambda inputs: format_docs(inputs["docs"]))
    try:
        found = prepared.invoke({"question": message, "history": history_messages(history)})
    except Exception as exc:
        return ChatResult(
            answer=f"检索失败（{exc.__class__.__name__}: {exc}）。",
            products=[],
            chunks=[],
            llm=None,
        )
    products = [hit_from_document(doc) for doc in found["docs"]]
    chunks = [chunk_from_document(doc) for doc in found["docs"]]
    if not settings.dashscope_api_key:
        return ChatResult(
            answer=search_fallback(products) + "\n\n未配置 DASHSCOPE_API_KEY，上面是直接检索到的商品。",
            products=products,
            chunks=chunks,
            llm=None,
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.with_language(prompts.SYSTEM, message)),
            MessagesPlaceholder("history"),
            ("human", "{question}\n\n商品资料：\n{context}"),
        ]
    )
    try:
        text = (prompt | model() | StrOutputParser()).invoke(found)
    except Exception as exc:
        return ChatResult(
            answer=search_fallback(products) + f"\n\n通义千问暂时不可用（{exc.__class__.__name__}: {exc}）。上面是直接检索到的商品。",
            products=products,
            chunks=chunks,
            llm=None,
        )
    return ChatResult(answer=text, products=products, chunks=chunks, llm=settings.qwen_model)


def _answer_listing(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None,
    filters: ProductFilter | None,
) -> ChatResult:
    try:
        products = store.list_products(limit=20, filters=filters)
    except Exception as exc:
        return ChatResult(
            answer=f"检索失败（{exc.__class__.__name__}: {exc}）。",
            products=[],
            chunks=[],
            llm=None,
        )
    chunks = [chunk_from_hit(product) for product in products]
    if not settings.dashscope_api_key:
        return ChatResult(answer=list_fallback(products), products=products, chunks=chunks, llm=None)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.with_language(f"{prompts.SYSTEM}\n{prompts.LISTING}", message)),
            MessagesPlaceholder("history"),
            ("human", "{question}\n\n商品目录：\n{context}"),
        ]
    )
    try:
        text = (prompt | model() | StrOutputParser()).invoke(
            {
                "question": message,
                "history": history_messages(history),
                "context": catalog_context(products),
            }
        )
    except Exception as exc:
        return ChatResult(
            answer=list_fallback(products) + f"\n\n通义千问暂时不可用（{exc.__class__.__name__}: {exc}）。",
            products=products,
            chunks=chunks,
            llm=None,
        )
    return ChatResult(answer=text, products=products, chunks=chunks, llm=settings.qwen_model)
