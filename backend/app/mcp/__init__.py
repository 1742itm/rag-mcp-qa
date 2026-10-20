"""轻量自研 MCP 模块：协议定义、工具注册表、上下文管理与 Agent 编排。"""
from .agent import Agent
from .context import ContextManager, context_manager
from .discriminator import Classification, Verdict, classify
from .protocol import Tool, ToolCall, ToolParameter, tool
from .registry import ToolRegistry, registry
from .tools import register_builtin_tools

__all__ = [
    "Agent",
    "ContextManager",
    "context_manager",
    "Classification",
    "Verdict",
    "classify",
    "Tool",
    "ToolCall",
    "ToolParameter",
    "tool",
    "ToolRegistry",
    "registry",
    "register_builtin_tools",
    "build_agent",
]


def build_agent() -> Agent:
    """组装 Agent：注册内置工具 + 注入 LLM 与上下文管理器。"""
    from ..llm.client import get_llm

    register_builtin_tools()
    return Agent(get_llm(), registry, context_manager)