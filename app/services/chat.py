from app.core.config import settings
from app.schemas.chat import ChatResult, ChatTurn
from app.schemas.product import ProductFilter, ProductHit
from app.services.store import ProductStore

SYSTEM = """You are a sales assistant for a musical instrument store.
Answer only from the product context. Do not invent prices, stock, or specifications.
Reply in the same language as the user's question.
When you recommend a product, include its title, brand, price, and sku_url.
If the context has no matching product, say so."""


def answer(
    store: ProductStore,
    message: str,
    history: list[ChatTurn] | None = None,
    filters: ProductFilter | None = None,
    limit: int = 4,
) -> ChatResult:
    products = store.search_text(message, limit=limit, filters=filters)
    llm_name = _ollama_model()
    if llm_name is None:
        return ChatResult(answer=_fallback(products), products=products, llm=None)
    try:
        text = _ollama_answer(message, history or [], products, llm_name)
    except Exception as exc:
        logger_message = f"{exc.__class__.__name__}: {exc}"
        return ChatResult(
            answer=_fallback(products) + f"\n\n模型暂时不可用（{logger_message}）。上面是直接检索到的商品。",
            products=products,
            llm=None,
        )
    return ChatResult(answer=text, products=products, llm=llm_name)


def _fallback(products: list[ProductHit]) -> str:
    if not products:
        return "没有找到匹配的商品。"
    lines = ["根据商品库，比较接近的是："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        lines.append(f"- {product.title}（{product.brand}，{price}，{stock}）{product.sku_url}")
    return "\n".join(lines)


def _ollama_model() -> str | None:
    import httpx

    try:
        response = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=1.5,
            trust_env=False,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    names = {item.get("name", "") for item in response.json().get("models", [])}
    wanted = settings.ollama_model
    if wanted in names or f"{wanted}:latest" in names:
        return wanted
    return None


def _ollama_answer(
    message: str,
    history: list[ChatTurn],
    products: list[ProductHit],
    model_name: str,
) -> str:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_ollama import ChatOllama

    messages = []
    for turn in history:
        if turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))
        else:
            messages.append(HumanMessage(content=turn.content))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nProduct context:\n{context}"),
        ]
    )
    model = ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke(
        {
            "history": messages,
            "question": message,
            "context": _context(products),
        }
    )


def _context(products: list[ProductHit]) -> str:
    if not products:
        return "No matching products."
    blocks = []
    for index, product in enumerate(products, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] sku={product.sku} brand={product.brand} category={product.category}",
                    f"price={product.sale_price} msrp={product.msrp} in_stock={product.in_stock}",
                    f"style={product.style}",
                    f"sku_url={product.sku_url}",
                    product.document[:4000],
                ]
            )
        )
    return "\n\n".join(blocks)
