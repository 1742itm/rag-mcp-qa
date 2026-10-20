"""判别器单测：覆盖流式判别的全部结论分支与边界。

判别器是 ReAct 改造的关键部件——工具调用的 JSON 一旦被误判成回答，
就会直接泄露给用户，所以这里尽量把边界钉死。
"""
from __future__ import annotations

import json

import pytest

from app.mcp.discriminator import (
    MAX_TOOL_BUFFER,
    MIN_ANSWER_CHARS,
    Classification,
    Verdict,
    classify,
)

# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------


def v(buffer: str) -> Verdict:
    return classify(buffer).verdict


def tool_call(name: str = "knowledge_search", **arguments) -> str:
    return json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False)


PROSE = "根据参考资料，这个问题的答案可以从三个方面来说明。"


# ----------------------------------------------------------------------
# 一、信息不足：必须继续缓冲
# ----------------------------------------------------------------------


class TestUnknown:
    @pytest.mark.parametrize("buffer", ["", " ", "\n\n", "\t "])
    def test_空白输入不判定(self, buffer):
        assert v(buffer) is Verdict.UNKNOWN

    def test_极短正文不判定(self):
        assert v("短") is Verdict.UNKNOWN

    def test_未达窗口的正文不判定(self):
        short = "一" * (MIN_ANSWER_CHARS - 1)
        assert v(short) is Verdict.UNKNOWN

    def test_只收到围栏本身不判定(self):
        assert v("```") is Verdict.UNKNOWN

    def test_只收到半个围栏不判定(self):
        assert v("``") is Verdict.UNKNOWN


# ----------------------------------------------------------------------
# 二、纯正文 → ANSWER
# ----------------------------------------------------------------------


class TestAnswer:
    def test_达到窗口即判定为回答(self):
        assert v("一" * MIN_ANSWER_CHARS) is Verdict.ANSWER

    def test_正常中文回答(self):
        assert v(PROSE) is Verdict.ANSWER

    def test_回答带代码块围栏(self):
        # 围栏只影响判别，不影响展示
        code = "```python\n" + "print('hello world')\n" * 3 + "```"
        assert v(code) is Verdict.ANSWER

    def test_过短的围栏片段仍继续等待(self):
        # 剥离围栏后可判别长度不足窗口，属于「信息不足」，继续缓冲是正确行为
        assert v("```python\nprint(1)\n```") is Verdict.UNKNOWN

    def test_回答里出现单独的左花括号(self):
        # 花括号收不齐 → 在到达上限前保持 UNKNOWN，不会误判成回答
        assert v("下面给出示例配置：{" + "x" * 10) is Verdict.UNKNOWN


# ----------------------------------------------------------------------
# 三、工具调用 → TOOL
# ----------------------------------------------------------------------


class TestTool:
    def test_标准工具调用(self):
        c = classify(tool_call("knowledge_search", query="RAG 是什么"))
        assert c.verdict is Verdict.TOOL
        assert c.tool == "knowledge_search"
        assert c.arguments == {"query": "RAG 是什么"}

    def test_参数含数字与布尔值原样保留(self):
        c = classify('{"tool": "kb", "arguments": {"query": "x", "top_k": 3}}')
        assert c.verdict is Verdict.TOOL
        assert c.arguments == {"query": "x", "top_k": 3}

    def test_带_json围栏的调用(self):
        c = classify('```json\n{"tool": "kb", "arguments": {"query": "x"}}\n```')
        assert c.verdict is Verdict.TOOL
        assert c.tool == "kb"

    def test_围栏无换行(self):
        c = classify('```json{"tool": "kb", "arguments": {}}')
        assert c.verdict is Verdict.TOOL

    def test_字符串内的花括号不会破坏配平(self):
        c = classify('{"tool": "kb", "arguments": {"query": "a}b{c"}}')
        assert c.verdict is Verdict.TOOL
        assert c.arguments == {"query": "a}b{c"}

    def test_字符串内的转义引号不会破坏配平(self):
        raw = r'{"tool": "kb", "arguments": {"query": "say \"hi\""}}'
        c = classify(raw)
        assert c.verdict is Verdict.TOOL
        assert c.arguments == {"query": 'say "hi"'}

    def test_用name字段声明工具(self):
        c = classify('{"name": "kb", "args": {"query": "x"}}')
        assert c.verdict is Verdict.TOOL
        assert c.tool == "kb"
        assert c.arguments == {"query": "x"}

    def test_arguments类型异常时兜底为空字典(self):
        c = classify('{"tool": "kb", "arguments": "不是字典"}')
        assert c.verdict is Verdict.TOOL
        assert c.arguments == {}

    def test_缺少arguments字段(self):
        c = classify('{"tool": "kb"}')
        assert c.verdict is Verdict.TOOL
        assert c.arguments == {}

    # ---- 不完整 JSON：必须继续等 ----

    def test_未闭合的对象不判定(self):
        assert v('{"tool": "kb", "arguments": {"query": "x"') is Verdict.UNKNOWN

    def test_嵌套未闭合不判定(self):
        assert v('{"tool": "kb", "arguments": {') is Verdict.UNKNOWN

    def test_刚写出键名不判定(self):
        assert v('{"tool"') is Verdict.UNKNOWN

    def test_仅左花括号不判定(self):
        assert v("{") is Verdict.UNKNOWN


# ----------------------------------------------------------------------
# 四、显式「无需工具」→ NO_TOOL
# ----------------------------------------------------------------------


class TestNoTool:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"tool": null}',
            '{"tool": "null"}',
            '{"tool": "none"}',
            '{"tool": ""}',
            '{"tool": "  "}',
            '{"name": null}',
        ],
    )
    def test_空值形态判为无需工具(self, raw):
        assert v(raw) is Verdict.NO_TOOL

    def test_非工具结构的JSON也判为无需工具(self):
        # 模型输出了 JSON 但不是工具调用：不能当正文展示（会露出一坨 JSON），
        # 判为 NO_TOOL，由调用方改为再索取一次回答
        assert v('{"answer": "你好"}') is Verdict.NO_TOOL


# ----------------------------------------------------------------------
# 五、先天异常：fail-open 兜底
# ----------------------------------------------------------------------


class TestFailOpen:
    def test_括号配平但非法JSON判为回答(self):
        assert v('{"a": }') is Verdict.ANSWER

    def test_超长未闭合对象放弃判别(self):
        assert v("{" + "a" * (MAX_TOOL_BUFFER + 100)) is Verdict.ANSWER

    def test_上限边界_刚好达到即放弃(self):
        assert v("{" + "a" * (MAX_TOOL_BUFFER - 1)) is Verdict.ANSWER

    def test_上限边界_差一个字符仍等待(self):
        assert v("{" + "a" * (MAX_TOOL_BUFFER - 2)) is Verdict.UNKNOWN


# ----------------------------------------------------------------------
# 六、前置语场景（判别器存在的核心意义）
# ----------------------------------------------------------------------


class TestPreamble:
    def test_短前置语后跟工具调用_判为工具而非回答(self):
        raw = "好的，我来查一下。\n" + tool_call("kb", query="记忆机制")
        c = classify(raw)
        assert c.verdict is Verdict.TOOL
        assert c.arguments == {"query": "记忆机制"}

    def test_前置语达到窗口长度后跟工具调用(self):
        # 窗口内出现 `{` 就转入 JSON 分支，不会先判成回答
        preamble = "我先思考一下再决定" * 3
        assert len(preamble) >= MIN_ANSWER_CHARS
        raw = preamble + "\n" + tool_call("kb", query="x")
        assert v(raw) is Verdict.TOOL

    def test_超长前置语后跟工具调用_记录已知边界(self):
        # 已知边界（设计取舍）：前置语本身就超过 MIN_ANSWER_CHARS 时，
        # 判别器无法区分「长前置语」与「正式回答」，会先判为 ANSWER。
        # 这依赖提示词约束模型「调用工具时不要写前置语」，属于上游防线。
        long_preamble = "我先思考一下再决定" * 3
        assert len(long_preamble) >= MIN_ANSWER_CHARS
        assert v(long_preamble) is Verdict.ANSWER
        # 若随后又出现工具调用，已经判出的 ANSWER 无法回退（需由调用方 latch 语义兜底）
        assert v(long_preamble + "\n" + tool_call("kb", query="x")) is Verdict.TOOL

    def test_前置语后无工具调用_仍判为回答(self):
        raw = "好的，我先说明一下。" + PROSE
        assert v(raw) is Verdict.ANSWER


# ----------------------------------------------------------------------
# 七、增量流式用法（真实调用方式）
# ----------------------------------------------------------------------


class TestStreaming:
    def test_逐字符喂入工具调用_只在闭合那一刻判定(self):
        raw = tool_call("kb", query="RAG")
        verdicts = []
        for i in range(len(raw)):
            verdicts.append(v(raw[: i + 1]))

        # 除了最后一步，前面全部应当是 UNKNOWN
        assert all(x is Verdict.UNKNOWN for x in verdicts[:-1]), verdicts
        assert verdicts[-1] is Verdict.TOOL

    def test_逐字符喂入正文_只在达到窗口时判定一次(self):
        raw = PROSE
        hit = [
            i + 1
            for i in range(len(raw))
            if classify(raw[: i + 1]).verdict is Verdict.ANSWER
        ]
        assert hit, "应当判定出回答"
        # 首次判定发生在窗口处，且此后保持判定为回答
        assert hit[0] == MIN_ANSWER_CHARS
        assert hit[-1] == len(raw)

    def test_按token喂入_判定结果与整体一致(self):
        raw = tool_call("knowledge_search", query="技术趋势")

        def feed_by_tokens(text: str, size: int) -> Verdict:
            verdict = Verdict.UNKNOWN
            for i in range(0, len(text), size):
                verdict = classify(text[: i + size]).verdict
            return verdict

        for size in (1, 2, 3, 7):
            assert feed_by_tokens(raw, size) is Verdict.TOOL, f"token size={size}"


# ----------------------------------------------------------------------
# 八、不变的契约
# ----------------------------------------------------------------------


class TestContract:
    def test_判别是只读的_不修改原文本(self):
        raw = "```json\n" + tool_call("kb", query="x") + "\n```"
        before = str(raw)
        classify(raw)
        assert raw == before

    def test_同一输入重复判别结果稳定(self):
        raw = tool_call("kb", query="x")
        assert {classify(raw).verdict for _ in range(5)} == {Verdict.TOOL}

    def test_结果类型正确(self):
        c = classify(tool_call("kb", query="x"))
        assert isinstance(c, Classification)
        assert isinstance(c.arguments, dict)

    def test_非工具判定时不应带工具字段(self):
        assert classify(PROSE).tool is None
        assert classify('{"tool": null}').tool is None
