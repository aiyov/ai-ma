"""Qwen client and message helpers."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.schemas.chat import ChatTurn


def model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.qwen_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.qwen_base_url,
        temperature=0.2,
        timeout=settings.qwen_timeout,
        max_retries=settings.qwen_max_retries,
        extra_body={"enable_thinking": False},
    )


def history_messages(history: list[ChatTurn] | None) -> list:
    turns = []
    for turn in history or []:
        if turn.role == "assistant":
            turns.append(AIMessage(content=turn.content))
        else:
            turns.append(HumanMessage(content=turn.content))
    return turns


def message_text(message) -> str:
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
