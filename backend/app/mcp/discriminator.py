"""流式判别器：增量判断模型本轮输出的是「工具调用」还是「最终回答」。

ReAct 风格的单次调用 Agent 必须边收流边判别：
    - 工具调用的 JSON **绝不能**展示给用户；
    - 最终回答又要尽早开始流式输出，不能等整段生成完。

判别策略（关键点：不依赖「首个字符」，而是锁定第一个 `{`）：
    - 文本里**还没有** `{`  → 是纯正文，攒够 MIN_ANSWER_CHARS 即可判定 ANSWER；
    - 文本里**出现** `{`      → 从该位置起是「JSON 候选」，在它解析出完整对象前
                              一律不判定，避免把半截 JSON 当成正文泄露出去。

这样即使模型先写一句寒暄（「好的，我来查一下」）再输出工具调用，
寒暄也不会被抢先当作回答流出去——因为 `{` 一出现判别器就停在那儿等 JSON 收全。

判别本身是**只读**的：它只返回结论，不修改文本，
调用方负责决定展示哪一段（见 agent.py）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

# 无 `{` 的纯正文，至少攒够这么多字符才判定为回答。
# 这个窗口的作用：给模型可能写的前置语（「好的，我来查一下」）留出观测时间——
# 只要紧接着出现 `{`，判别器就会转入 JSON 分支，不会把后续工具 JSON 泄露出去。
# 窗口越大越安全，但会推迟所有回答的首字时间，故取一个折中值。
MIN_ANSWER_CHARS = 24

# 出现 `{` 但迟迟收不到完整 JSON 时的等待上限；超过则放弃判别、按正文处理。
# 这是 fail-open 兜底：宁可把文本显示出来，也不要让请求一直卡住。
MAX_TOOL_BUFFER = 2048

# 模型有时会把 JSON 包在 ```json 围栏里，判别前先剥掉围栏（只影响判别，不影响展示）
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?")

# 这些取值都表示「模型明确表示不需要工具」
_NULLISH = {"", "null", "none", "nil"}


class Verdict(str, Enum):
    UNKNOWN = "unknown"   # 信息不足，请继续缓冲
    TOOL = "tool"         # 工具调用（工具名非空），应执行
    NO_TOOL = "no_tool"   # 模型显式表示无需工具（如 {"tool": null}），需另行索取回答
    ANSWER = "answer"     # 最终回答正文，可直接展示


@dataclass(frozen=True)
class Classification:
    """判别结果。tool / arguments 仅在 verdict == TOOL 时有效。"""

    verdict: Verdict
    tool: str | None = None
    arguments: dict | None = None


def classify(buffer: str) -> Classification:
    """对当前累积文本给出判别结论。

    buffer 应当是「从头开始累积的完整文本」，而不是单个增量片段。
    """
    text = buffer.lstrip()
    if not text:
        return Classification(Verdict.UNKNOWN)

    # 剥离 ```json 围栏后再做判别
    stripped = _FENCE_RE.sub("", text)
    if not stripped:
        # 目前只收到了围栏本身，信息不足
        return Classification(Verdict.UNKNOWN)

    start = stripped.find("{")
    if start == -1:
        # 纯正文分支
        if len(stripped) >= MIN_ANSWER_CHARS:
            return Classification(Verdict.ANSWER)
        return Classification(Verdict.UNKNOWN)

    # JSON 候选分支：从第一个 { 起尝试取出配平的对象
    obj_text = _balanced_object(stripped, start)
    if obj_text is None:
        if len(stripped) >= MAX_TOOL_BUFFER:
            # fail-open：长时间收不到完整 JSON，按正文处理
            return Classification(Verdict.ANSWER)
        return Classification(Verdict.UNKNOWN)

    return _parse_payload(obj_text)


# ----------------------------------------------------------------------
def _balanced_object(text: str, start: int) -> str | None:
    """从 text[start] 处的 `{` 取出配平的 JSON 对象；未闭合时返回 None。

    需要正确跳过字符串内部的括号与转义，例如 {"query": "a}b{c"} / "say \\"hi\\""，
    否则会被误判为「对象已闭合」而提前解析出错。
    """
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _parse_payload(obj_text: str) -> Classification:
    """解析已配平的 JSON 对象，判断它是工具调用还是普通正文。"""
    try:
        data = json.loads(obj_text)
    except json.JSONDecodeError:
        # 括号配平但不是合法 JSON：按正文处理，避免无谓等待
        return Classification(Verdict.ANSWER)

    if not isinstance(data, dict):
        return Classification(Verdict.ANSWER)

    name = data.get("tool", data.get("name"))
    if name is None or str(name).strip().lower() in _NULLISH:
        return Classification(Verdict.NO_TOOL)

    args = data.get("arguments")
    if args is None:
        args = data.get("args")
    if not isinstance(args, dict):
        args = {}

    return Classification(Verdict.TOOL, tool=str(name).strip(), arguments=args)
