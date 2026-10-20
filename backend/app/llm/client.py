"""LLM 客户端：统一封装所有「兼容 OpenAI 接口」的模型服务。

同一份代码即可对接 DeepSeek / Ollama / 通义千问 / vLLM 等，
只需在 .env 里改 base_url 与 model。
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from ..config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM 调用相关的可展示错误。"""


class LLMClient:
    def __init__(self) -> None:
        api_key = settings.llm_api_key or "sk-no-key-required"
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=api_key,
            timeout=settings.llm_timeout,
        )

    def _check_config(self) -> None:
        if not settings.llm_api_key and "localhost" not in settings.llm_base_url \
                and "127.0.0.1" not in settings.llm_base_url:
            raise LLMError(
                "未配置 LLM_API_KEY。请在 backend/.env 中填入 API Key，"
                "或把 LLM_BASE_URL 指向本地 Ollama（如 http://localhost:11434/v1）。"
            )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        """非流式补全，用于工具路由决策等场景。"""
        self._check_config()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature if temperature is None else temperature,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """流式补全，逐块产出文本增量。"""
        self._check_config()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature if temperature is None else temperature,
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content: Any = getattr(delta, "content", None)
            if content:
                yield content


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm