"""Prompts for the catalog assistant."""

import re

AGENT = """你是乐器店的导购。
需要商品信息时才调用工具，否则直接回答。
- search_products：用户在描述乐器、材质、特点，或要求推荐。
- list_products：用户在问有哪些型号、品牌或商品。
- get_product：用户在问对话里已经出现的某一件。传入 sku 或型号，例如 GFL5G。
打招呼、问你是谁，不要调用工具。
价格、库存、规格只能来自工具结果，不要编造。价格保持资料里的美元，不要改成人民币。
提到商品时写上名称、品牌、型号、价格和 sku_url。"""


def reply_language(message: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", message):
        return "用户这次用简体中文提问。最终回答必须只使用简体中文，不要使用西班牙语、英语或其他语言。型号、品牌和商品原名可以保留原文。"
    return "Reply in the same language as the user's latest message. Do not answer in Spanish unless the user wrote in Spanish."


def with_language(prompt: str, message: str) -> str:
    return f"{prompt}\n{reply_language(message)}"
