import json
import re
from operator import itemgetter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.schemas.chat import ChatResult, ChatTurn, RetrievedChunk
from app.schemas.product import ProductFilter, ProductHit
from app.services.store import ProductStore, hit_from_document

SYSTEM = """You are a sales assistant for a musical instrument store.
Answer only from the product context. Do not invent prices, stock, or specifications.
Reply in the same language as the user's question.
When you recommend a product, include its title, brand, price, and sku_url.
If the context has no matching product, say so."""

AGENT_SYSTEM = """You are a sales assistant for a musical instrument store.
Use a tool when the user needs the catalog. Otherwise answer directly.
- search_products: they describe an instrument, a material, a feature, or ask for a recommendation.
- list_products: they ask which models, brands, or products exist.
- get_product: they ask about one model already mentioned. Pass its sku or part number.
Greetings and questions about who you are need no tool.
Answer only from tool results. Do not invent prices, stock, or specifications.
Reply in the same language as the user.
When you mention a product, include its title, brand, part number, price, and sku_url."""

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


def answer(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None = None,
    filters: ProductFilter | None = None,
    limit: int = 4,
) -> ChatResult:
    if settings.dashscope_api_key:
        try:
            return _answer_with_tools(store, message, history, filters, limit)
        except Exception:
            return _answer_by_rules(store, message, history, filters, limit)
    return _answer_by_rules(store, message, history, filters, limit)


class _Catalog:
    def __init__(self, store: ProductStore, filters: ProductFilter | None, limit: int) -> None:
        self.store = store
        self.filters = filters or ProductFilter()
        self.limit = limit
        self.products: list[ProductHit] = []
        self.chunks: list[RetrievedChunk] = []
        self._seen: set[str] = set()

    def remember(self, hit: ProductHit, text: str, chunk_index: int = 0, score: float | None = None) -> None:
        if hit.sku not in self._seen:
            self._seen.add(hit.sku)
            self.products.append(hit)
        self.chunks.append(
            RetrievedChunk(
                sku=hit.sku,
                title=hit.title,
                chunk_index=chunk_index,
                score=score,
                text=text,
            )
        )

    def narrowed(self, brand: str, max_price: float, in_stock: bool) -> ProductFilter:
        data = self.filters.model_dump()
        if not data.get("brand") and brand.strip():
            data["brand"] = brand.strip()
        if data.get("max_price") is None and max_price > 0:
            data["max_price"] = max_price
        if data.get("in_stock") is None and in_stock:
            data["in_stock"] = True
        return ProductFilter(**data)

    def tools(self):
        catalog = self

        @tool
        def search_products(query: str, brand: str = "", max_price: float = 0, in_stock: bool = False) -> str:
            """Search the catalog by a description of the instrument the customer wants.

            query: words for the instrument, material, or feature.
            brand: brand name, or an empty string when none was given.
            max_price: highest sale price, or 0 when there is no limit.
            in_stock: true only when the customer asked for items in stock.
            """
            hits = catalog.store.search_text(
                query,
                limit=catalog.limit,
                filters=catalog.narrowed(brand, max_price, in_stock),
            )
            for hit in hits:
                catalog.remember(hit, hit.document or hit.title, score=hit.score)
            return _dump_products(hits, lambda hit: hit.document or hit.title)

        @tool
        def list_products(brand: str = "", max_price: float = 0, in_stock: bool = False) -> str:
            """List models in the catalog when the customer asks what exists.

            brand: brand name, or an empty string for every brand.
            max_price: highest sale price, or 0 when there is no limit.
            in_stock: true only when the customer asked for items in stock.
            """
            hits = catalog.store.list_products(
                limit=20,
                filters=catalog.narrowed(brand, max_price, in_stock),
            )
            for hit in hits:
                catalog.remember(hit, _chunk_from_hit(hit).text)
            return _dump_products(hits, lambda hit: hit.heading or hit.style or hit.title)

        @tool
        def get_product(sku_or_part_number: str) -> str:
            """Read one product by SKU or part number, for example GFL5G."""
            hit = catalog.store.find_product(sku_or_part_number)
            if hit is None:
                return json.dumps({"found": False, "query": sku_or_part_number}, ensure_ascii=False)
            excerpt = (hit.document or hit.title)[:1800]
            catalog.remember(hit, excerpt[:500] or hit.title)
            return _dump_products([hit], lambda item: (item.document or item.title)[:1800])

        return [search_products, list_products, get_product]


def _answer_with_tools(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None,
    filters: ProductFilter | None,
    limit: int,
) -> ChatResult:
    catalog = _Catalog(store, filters, limit)
    tools = catalog.tools()
    by_name = {item.name: item for item in tools}
    messages: list = [SystemMessage(content=AGENT_SYSTEM), *_history_messages(history), HumanMessage(content=message)]
    tool_model = _model().bind_tools(tools)
    plain = _model()
    for _ in range(2):
        ai = tool_model.invoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return _tool_result(_message_text(ai), catalog)
        for call in calls:
            messages.append(_run_tool(by_name, call))
    final = plain.invoke(messages)
    return _tool_result(_message_text(final), catalog)


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


def _tool_result(text: str, catalog: _Catalog) -> ChatResult:
    if not text:
        text = _fallback(catalog.products) if catalog.products else "我没有查到可以回答的内容。"
    return ChatResult(
        answer=text,
        products=catalog.products,
        chunks=catalog.chunks,
        llm=settings.qwen_model,
    )


def _history_messages(history: list[ChatTurn] | None) -> list:
    turns = []
    for turn in history or []:
        if turn.role == "assistant":
            turns.append(AIMessage(content=turn.content))
        else:
            turns.append(HumanMessage(content=turn.content))
    return turns


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _dump_products(hits: list[ProductHit], excerpt) -> str:
    rows = [
        {
            "sku": hit.sku,
            "part_number": hit.part_number,
            "title": hit.title,
            "brand": hit.brand,
            "sale_price": hit.sale_price,
            "in_stock": hit.in_stock,
            "style": hit.style,
            "sku_url": hit.sku_url,
            "excerpt": excerpt(hit),
        }
        for hit in hits
    ]
    return json.dumps({"products": rows}, ensure_ascii=False)


def _answer_by_rules(
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
    retriever = store.as_retriever(limit=limit, filters=filters)
    turns = []
    for turn in history or []:
        if turn.role == "assistant":
            turns.append(AIMessage(content=turn.content))
        else:
            turns.append(HumanMessage(content=turn.content))
    prepared = RunnableParallel(
        docs=itemgetter("question") | retriever,
        question=itemgetter("question"),
        history=itemgetter("history"),
    ) | RunnablePassthrough.assign(context=lambda inputs: _format_docs(inputs["docs"]))
    try:
        found = prepared.invoke({"question": message, "history": turns})
    except Exception as exc:
        return ChatResult(
            answer=f"检索失败（{exc.__class__.__name__}: {exc}）。",
            products=[],
            chunks=[],
            llm=None,
        )
    products = [hit_from_document(doc) for doc in found["docs"]]
    chunks = [_chunk(doc) for doc in found["docs"]]
    if not settings.dashscope_api_key:
        return ChatResult(
            answer=_fallback(products) + "\n\n未配置 DASHSCOPE_API_KEY，上面是直接检索到的商品。",
            products=products,
            chunks=chunks,
            llm=None,
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nProduct context:\n{context}"),
        ]
    )
    try:
        text = (prompt | _model() | StrOutputParser()).invoke(found)
    except Exception as exc:
        return ChatResult(
            answer=_fallback(products) + f"\n\n通义千问暂时不可用（{exc.__class__.__name__}: {exc}）。上面是直接检索到的商品。",
            products=products,
            chunks=chunks,
            llm=None,
        )
    return ChatResult(answer=text, products=products, chunks=chunks, llm=settings.qwen_model)


def _is_listing(message: str) -> bool:
    text = message.strip()
    if _COMPARE.search(text):
        return False
    return bool(_LISTING.search(text))


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
    chunks = [_chunk_from_hit(product) for product in products]
    if not settings.dashscope_api_key:
        return ChatResult(answer=_list_fallback(products), products=products, chunks=chunks, llm=None)
    turns = []
    for turn in history or []:
        if turn.role == "assistant":
            turns.append(AIMessage(content=turn.content))
        else:
            turns.append(HumanMessage(content=turn.content))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LIST_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nCatalog:\n{context}"),
        ]
    )
    try:
        text = (prompt | _model() | StrOutputParser()).invoke(
            {"question": message, "history": turns, "context": _catalog_context(products)}
        )
    except Exception as exc:
        return ChatResult(
            answer=_list_fallback(products) + f"\n\n通义千问暂时不可用（{exc.__class__.__name__}: {exc}）。",
            products=products,
            chunks=chunks,
            llm=None,
        )
    return ChatResult(answer=text, products=products, chunks=chunks, llm=settings.qwen_model)


def _list_fallback(products: list[ProductHit]) -> str:
    if not products:
        return "商品库里还没有型号。"
    lines = ["商品库里现在有这些型号："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        model = product.part_number or product.sku
        lines.append(f"- {model}　{product.brand}　{product.title}（{price}，{stock}）")
    return "\n".join(lines)


def _catalog_context(products: list[ProductHit]) -> str:
    if not products:
        return "The catalog is empty."
    blocks = []
    for index, product in enumerate(products, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] part_number={product.part_number or product.sku} brand={product.brand}",
                    f"title={product.title}",
                    f"price={product.sale_price} in_stock={product.in_stock} style={product.style}",
                    f"sku_url={product.sku_url}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _chunk_from_hit(product: ProductHit) -> RetrievedChunk:
    price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
    text = " · ".join(part for part in (product.part_number, product.brand, product.style, price) if part)
    return RetrievedChunk(
        sku=product.sku,
        title=product.title,
        chunk_index=0,
        score=None,
        text=text or product.title,
    )


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
            (
                "system",
                "Reply with exactly catalog or chat. "
                "catalog: the user is asking about instruments, prices, stock, brands, or recommendations. "
                "chat: greetings, identity, thanks, or anything unrelated to the catalog.",
            ),
            ("human", "{question}"),
        ]
    )
    model = _model()
    try:
        route = (prompt | model | StrOutputParser()).invoke({"question": message})
    except Exception:
        return False
    return route.strip().lower().startswith("catalog")


def _reply_without_products(message: str, history: list[ChatTurn] | None) -> ChatResult:
    if not settings.dashscope_api_key:
        return ChatResult(answer=_direct_reply(message), products=[], chunks=[], llm=None)
    turns = []
    for turn in history or []:
        if turn.role == "assistant":
            turns.append(AIMessage(content=turn.content))
        else:
            turns.append(HumanMessage(content=turn.content))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHAT_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    try:
        text = (prompt | _model() | StrOutputParser()).invoke({"question": message, "history": turns})
    except Exception:
        text = _direct_reply(message)
        return ChatResult(answer=text, products=[], chunks=[], llm=None)
    return ChatResult(answer=text, products=[], chunks=[], llm=settings.qwen_model)


def _direct_reply(message: str) -> str:
    if _SOCIAL.match(message.strip()):
        return "我是乐器商品库的导购，可以帮你查型号、价格和库存。直接说想找什么样的乐器就行。"
    return "我只能根据商品库回答乐器相关的问题。想找长笛、看价格或库存，直接说就行。"


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.qwen_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.qwen_base_url,
        temperature=0.2,
        timeout=settings.qwen_timeout,
        max_retries=settings.qwen_max_retries,
        extra_body={"enable_thinking": False},
    )


def _fallback(products: list[ProductHit]) -> str:
    if not products:
        return "没有找到匹配的商品。"
    lines = ["根据商品库，比较接近的是："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        lines.append(f"- {product.title}（{product.brand}，{price}，{stock}）{product.sku_url}")
    return "\n".join(lines)


def _chunk(doc) -> RetrievedChunk:
    meta = doc.metadata
    score = meta.get("score")
    return RetrievedChunk(
        sku=str(meta.get("sku") or ""),
        title=str(meta.get("title") or ""),
        chunk_index=int(meta.get("chunk_index") or 0),
        score=None if score is None else float(score),
        text=doc.page_content,
    )


def _format_docs(docs: list) -> str:
    if not docs:
        return "No matching products."
    blocks = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"[{index}] sku={meta.get('sku')} brand={meta.get('brand')} category={meta.get('category')}",
                    f"price={meta.get('sale_price')} msrp={meta.get('msrp')} in_stock={meta.get('in_stock')}",
                    f"style={meta.get('style')}",
                    f"sku_url={meta.get('sku_url')}",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)
