from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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
    if not settings.dashscope_api_key:
        return ChatResult(
            answer=_fallback(products) + "\n\n未配置 DASHSCOPE_API_KEY，上面是直接检索到的商品。",
            products=products,
            llm=None,
        )
    try:
        text = _qwen_answer(message, history or [], products)
    except Exception as exc:
        return ChatResult(
            answer=_fallback(products) + f"\n\n通义千问暂时不可用（{exc.__class__.__name__}: {exc}）。上面是直接检索到的商品。",
            products=products,
            llm=None,
        )
    return ChatResult(answer=text, products=products, llm=settings.qwen_model)


def _fallback(products: list[ProductHit]) -> str:
    if not products:
        return "没有找到匹配的商品。"
    lines = ["根据商品库，比较接近的是："]
    for product in products:
        price = "价格未知" if product.sale_price is None else f"${product.sale_price:.0f}"
        stock = "有货" if product.in_stock else "无货"
        lines.append(f"- {product.title}（{product.brand}，{price}，{stock}）{product.sku_url}")
    return "\n".join(lines)


def _qwen_answer(message: str, history: list[ChatTurn], products: list[ProductHit]) -> str:
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
    model = ChatOpenAI(
        model=settings.qwen_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.qwen_base_url,
        temperature=0.2,
        extra_body={"enable_thinking": False},
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
