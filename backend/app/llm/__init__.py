"""LLM 调用封装（兼容 OpenAI 接口）。"""
from .client import LLMClient, LLMError, get_llm

__all__ = ["LLMClient", "LLMError", "get_llm"]