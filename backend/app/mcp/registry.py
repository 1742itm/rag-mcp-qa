"""工具注册表：负责工具的登记、查询与安全执行。"""
from __future__ import annotations

import logging
from typing import Any

from .protocol import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """MCP 工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.handler is None:
            raise ValueError(f"工具 {tool.name} 未绑定 handler")
        if tool.name in self._tools:
            logger.warning("工具 %s 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def prompt_description(self) -> str:
        """把所有工具渲染成给模型看的纯文本说明。"""
        if not self._tools:
            return "（当前无可用工具）"
        return "\n".join(t.to_prompt() for t in self._tools.values())

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """执行工具调用，参数不匹配时做兜底裁剪。"""
        tool = self.get(name)
        if tool is None:
            return {"error": f"未知工具: {name}"}

        valid = {p.name for p in tool.parameters}
        kwargs = {k: v for k, v in arguments.items() if k in valid}
        if not kwargs and arguments:
            # 模型偶尔会用错参数名，退化为把首个值当作主参数
            first = tool.parameters[0].name if tool.parameters else None
            if first:
                kwargs = {first: next(iter(arguments.values()))}

        try:
            return await tool.handler(**kwargs)  # type: ignore[misc]
        except Exception as exc:  # 工具内部异常不应中断整个对话
            logger.exception("工具 %s 执行失败", name)
            return {"error": f"工具执行失败: {exc}"}


# 全局单例
registry = ToolRegistry()