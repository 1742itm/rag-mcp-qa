"""轻量自研 MCP 协议：定义工具的声明结构与调用约定。

设计目标：不引入官方 MCP SDK，用最少的代码表达同一套思想——
「把能力声明成工具（name / description / parameters），由模型自主选择是否调用」，
从而做到与具体 LLM 厂商解耦（DeepSeek / Ollama / 通义千问 均可用）。
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ToolParameter:
    """工具参数的声明。"""

    name: str
    type: str  # JSON Schema 类型：string / number / integer / boolean
    description: str
    required: bool = True


@dataclass
class Tool:
    """一个可被 Agent 调用的工具。"""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Awaitable[Any]] | None = None

    # ---- 序列化 ----
    def schema(self) -> dict[str, Any]:
        """输出标准 JSON Schema，便于渲染给模型或对外暴露。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {"type": p.type, "description": p.description}
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }

    def to_prompt(self) -> str:
        """渲染成注入系统提示词的纯文本描述。"""
        lines = [f"- {self.name}: {self.description}"]
        if self.parameters:
            params = ", ".join(
                f'{p.name} ({p.type}{"" if p.required else ", 可选"})={p.description}'
                for p in self.parameters
            )
            lines.append(f"  参数: {params}")
        return "\n".join(lines)


@dataclass
class ToolCall:
    """模型做出的一次工具调用决策。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: str) -> "ToolCall | None":
        """从模型输出中解析工具调用（容错处理多余文本）。"""
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        tool = data.get("tool") or data.get("name")
        if not tool or tool in ("null", "none", None):
            return None
        args = data.get("arguments") or data.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": args}
        if not isinstance(args, dict):
            args = {}
        return cls(name=str(tool), arguments=args)


def _extract_json(text: str) -> Any:
    """从任意文本里抠出第一个合法 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 代码块围栏
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """装饰器：把一个 async 函数注册成工具。"""

    def wrapper(func: Callable[..., Awaitable[Any]]) -> Tool:
        assert inspect.iscoroutinefunction(func), f"工具 {name} 必须是 async 函数"
        return Tool(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=func,
        )

    return wrapper