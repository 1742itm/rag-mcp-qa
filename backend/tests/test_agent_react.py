"""Agent ReAct 循环单测：用脚本化的假 LLM 精确控制每一轮的输出。

这些用例锁住的是「调用次数」与「输出不外泄」两个关键契约：
ReAct 相对旧的两段式方案省掉一次 LLM 往返，靠的就是
「模型作答的那一轮本身就是流式回答」，所以调用次数必须被钉死。
"""
from __future__ import annotations

import json

import pytest

from app.mcp.agent import MAX_TOOL_ROUNDS, Agent
from app.mcp.context import ContextManager
from app.mcp.protocol import Tool, ToolParameter
from app.mcp.registry import ToolRegistry

# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------


def tool_call(name: str = "kb_search", **arguments) -> str:
    return json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False)


def ref(doc_id: str, idx: int, score: float, text: str | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_index": idx,
        "filename": "文档.txt",
        "text": text or f"片段{doc_id}-{idx}",
        "score": score,
    }


class ScriptedLLM:
    """按脚本依次返回输出内容的假 LLM，并记录每次收到的消息序列。

    stream_chunks：每个元素是「一轮」；每轮是字符串列表（模拟逐块流式输出）。
    脚本用完后一律返回一段普通回答，避免测试卡死。
    """

    def __init__(self, turns: list[list[str]]):
        self.turns = list(turns)
        self.requests: list[list[dict[str, str]]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def system_prompts(self) -> list[str]:
        return [m[0]["content"] for m in self.requests]

    async def chat(self, messages, temperature=None):  # 兼容旧接口，ReAct 下不应被调用
        raise AssertionError("ReAct 模式下不应再调用非流式的 chat()")

    async def stream(self, messages, temperature=None):
        self.requests.append(list(messages))
        chunks = self.turns.pop(0) if self.turns else ["好的，这是回答。"]
        for chunk in chunks:
            yield chunk


def build_registry(results: dict[str, list[dict]], executed: list[str]) -> ToolRegistry:
    """假检索工具：返回约定与真实 knowledge_search 一致（含 summary）。"""
    reg = ToolRegistry()

    async def kb_search(query: str, top_k: int = 0):
        executed.append(query)
        hits = results.get(query, [])
        if not hits:
            return {
                "count": 0,
                "references": [],
                "summary": "未检索到相关内容（返回片段的相似度均低于相关性阈值 0.45）。",
            }
        return {
            "count": len(hits),
            "references": hits,
            "summary": f"检索到 {len(hits)} 个与问题相关的原文片段。",
        }

    reg.register(
        Tool(
            name="kb_search",
            description="知识库检索",
            parameters=[ToolParameter(name="query", type="string", description="检索词")],
            handler=kb_search,
        )
    )
    return reg


def add_doc_list_tool(reg: ToolRegistry, summary: str, executed: list[str]) -> None:
    """再加一个「非检索类」工具（无参数、不返回 references），用于验证多工具编排。"""

    async def doc_list():
        executed.append("doc_list")
        return {
            "count": 2,
            "documents": [{"filename": "a.md"}, {"filename": "b.md"}],
            "summary": summary,
        }

    reg.register(Tool(name="doc_list", description="文档清单", parameters=[], handler=doc_list))


async def collect(agent: Agent, question: str = "问题"):
    events = []
    async for ev in agent.run(question, []):
        events.append(ev)
    done = next(e["data"] for e in events if e["event"] == "done")
    answer = "".join(e["data"]["content"] for e in events if e["event"] == "delta")
    return events, done, answer


def run(agent: Agent, question: str = "问题"):
    import asyncio

    return asyncio.run(collect(agent, question))


def make(results: dict[str, list[dict]] | None = None, executed: list[str] | None = None):
    executed = executed if executed is not None else []
    return build_registry(results or {}, executed), executed


# ----------------------------------------------------------------------
# 一、直接作答：旧方案要 2 次调用，ReAct 只需 1 次
# ----------------------------------------------------------------------


class TestAnswerDirectly:
    def test_不经检索直接作答只调用一次(self):
        reg, _ = make()
        llm = ScriptedLLM([["这是", "直接给出的回答，不需要检索。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()))

        assert llm.call_count == 1, "ReAct 下直接作答只需一次调用"
        assert answer == "这是直接给出的回答，不需要检索。"
        assert done["rounds"] == 0
        assert done["references"] == []

    def test_直接作答时提示词里带了工具清单(self):
        reg, _ = make()
        llm = ScriptedLLM([["根据我的判断，这个问题无需查阅资料。"]])
        run(Agent(llm, reg, ContextManager()))
        assert "kb_search" in llm.system_prompts[0]

    def test_作答轮未被单独再请求一次(self):
        reg, _ = make()
        llm = ScriptedLLM([["这是一段足够长的、可以直接展示给用户的回答内容。"]])
        events, _, _ = run(Agent(llm, reg, ContextManager()))
        assert "answering" not in [e["data"].get("stage") for e in events if e["event"] == "status"]


# ----------------------------------------------------------------------
# 二、工具 JSON 绝不能泄露给用户（判别器 + latch 的组合契约）
# ----------------------------------------------------------------------


class TestNoLeak:
    def test_先输出工具调用再作答_工具JSON不出现在回答里(self):
        reg, executed = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                ["最终回答：", "资料显示答案为 A。"],
            ]
        )
        _, done, answer = run(Agent(llm, reg, ContextManager()))

        assert "kb_search" not in answer
        assert "{" not in answer
        assert executed == ["q1"]
        assert done["rounds"] == 1
        assert answer == "最终回答：资料显示答案为 A。"

    def test_工具调用前的寒暄也不会被当作回答(self):
        reg, executed = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM(
            [
                ["好的，我来查一下。\n", tool_call("kb_search", query="q1")],
                ["查到了，答案是 A。"],
            ]
        )
        _, done, answer = run(Agent(llm, reg, ContextManager()))
        assert executed == ["q1"]
        # 寒暄不是本次回答的正文，不应混进最终回答
        assert "我来查一下" not in answer

    def test_逐块流式输出工具调用也不会泄露(self):
        reg, executed = make({"q1": [ref("d1", 0, 0.9)]})
        raw = tool_call("kb_search", query="q1")
        llm = ScriptedLLM(
            [
                [raw[i : i + 3] for i in range(0, len(raw), 3)],  # 按 3 字符切片
                ["答案是 A。"],
            ]
        )
        _, _, answer = run(Agent(llm, reg, ContextManager()))
        assert executed == ["q1"]
        assert "{" not in answer and "tool" not in answer

    def test_未闭合的JSON会被丢弃而非展示(self):
        reg, executed = make()
        llm = ScriptedLLM(
            [
                ['{"tool": "kb_search", "arguments": {"query": "x"'],  # 从未闭合
                ["补充说明后给出的回答。"],
            ]
        )
        _, done, answer = run(Agent(llm, reg, ContextManager()))
        assert "{" not in answer
        assert done["rounds"] == 0
        assert llm.call_count == 2, "丢弃无效输出后应收尾轮重新索取回答"


# ----------------------------------------------------------------------
# 三、调用次数（ReAct 的核心收益，钉死不许回归）
# ----------------------------------------------------------------------


class TestCallBudget:
    def test_两轮检索共三次调用(self):
        reg, executed = make(
            {
                "q1": [ref("d1", 0, 0.9)],
                "q2": [ref("d2", 0, 0.8)],
            }
        )
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                [tool_call("kb_search", query="q2")],
                ["综合两轮资料，答案是 A 和 B。"],
            ]
        )
        _, done, answer = run(Agent(llm, reg, ContextManager()))

        # 旧的两段式方案需要 4 次（3 次路由 + 1 次作答），ReAct 只需 3 次
        assert llm.call_count == 3
        assert executed == ["q1", "q2"]
        assert done["rounds"] == 2
        assert answer == "综合两轮资料，答案是 A 和 B。"

    def test_模型在检索轮直接作答则不再多问一次(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                ["资料已足够，直接回答：A。"],
            ]
        )
        _, done, _ = run(Agent(llm, reg, ContextManager()))
        assert llm.call_count == 2
        assert done["rounds"] == 1

    def test_工具轮次被钳制在上限且收尾必有一次调用(self):
        reg, _ = make({f"q{i}": [ref(f"d{i}", 0, 0.9 - i * 0.1)] for i in range(1, 6)})
        llm = ScriptedLLM([[tool_call("kb_search", query=f"q{i}")] for i in range(1, 6)])
        _, done, _ = run(Agent(llm, reg, ContextManager()))

        assert done["rounds"] == MAX_TOOL_ROUNDS
        # 检索轮 MAX_TOOL_ROUNDS 次 + 收尾轮 1 次
        assert llm.call_count == MAX_TOOL_ROUNDS + 1


# ----------------------------------------------------------------------
# 四、收敛保障
# ----------------------------------------------------------------------


class TestConvergence:
    def test_收尾轮不暴露工具清单(self):
        reg, _ = make({f"q{i}": [ref(f"d{i}", 0, 0.5)] for i in range(1, 6)})
        llm = ScriptedLLM([[tool_call("kb_search", query=f"q{i}")] for i in range(1, 6)])
        run(Agent(llm, reg, ContextManager()))

        assert "kb_search" in llm.system_prompts[0], "检索轮应暴露工具"
        assert "kb_search" not in llm.system_prompts[-1], "收尾轮不应暴露工具"
        assert "不要调用任何工具" in llm.system_prompts[-1]

    def test_本轮无新片段则不再继续检索(self):
        # q2 返回的片段与 q1 完全重复 → 判定为无新内容，直接进入作答
        reg, executed = make({"q1": [ref("d1", 0, 0.9)], "q2": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                [tool_call("kb_search", query="q2")],
            ]
        )
        _, done, _ = run(Agent(llm, reg, ContextManager()))
        assert executed == ["q1", "q2"]
        assert done["rounds"] == 2
        assert len(done["references"]) == 1, "重复片段应被去重"
        assert llm.call_count == 3, "第二轮之后应直接收尾作答，不再尝试第三次检索"

    def test_模型显式表示无需工具时转入收尾轮(self):
        reg, _ = make()
        llm = ScriptedLLM(
            [
                ['{"tool": null}'],
                ["根据现有资料无法确定。"],
            ]
        )
        _, done, answer = run(Agent(llm, reg, ContextManager()))
        assert done["rounds"] == 0
        assert answer == "根据现有资料无法确定。"
        assert llm.call_count == 2

    def test_收尾轮仍产出工具JSON时给兜底提示(self):
        reg, _ = make()
        llm = ScriptedLLM(
            [
                ['{"tool": null}'],
                [tool_call("kb_search", query="x")],  # 收尾轮仍不听话
            ]
        )
        _, _, answer = run(Agent(llm, reg, ContextManager()))
        assert "请重试" in answer
        assert "{" not in answer


# ----------------------------------------------------------------------
# 五、Observation 注入与引用整理
# ----------------------------------------------------------------------


class TestObservation:
    def test_检索结果以Observation形式进入后续对话(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9, text="这是检索到的原文内容")]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                ["回答：A。"],
            ]
        )
        run(Agent(llm, reg, ContextManager()))

        second = llm.requests[1]
        contents = [m["content"] for m in second]
        assert any(c.startswith("Observation") for c in contents)
        assert any("这是检索到的原文内容" in c for c in contents)
        # 模型的原始工具调用输出也留在轨迹里
        assert any(m["role"] == "assistant" and "kb_search" in m["content"] for m in second)

    def test_Observation标注来源文档名且不带序号(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["回答。"]])
        run(Agent(llm, reg, ContextManager()))
        obs = [m["content"] for m in llm.requests[1] if m["content"].startswith("Observation")][0]
        assert "《文档.txt》" in obs
        assert "[1]" not in obs

    def test_跨轮片段去重且按相似度降序展示(self):
        reg, _ = make(
            {
                "q1": [ref("d1", 0, 0.70), ref("d2", 0, 0.40)],
                "q2": [ref("d2", 0, 0.40), ref("d3", 0, 0.90)],  # d2 与第一轮重复
            }
        )
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                [tool_call("kb_search", query="q2")],
                ["回答。"],
            ]
        )
        _, done, _ = run(Agent(llm, reg, ContextManager()))

        keys = [(r["doc_id"], r["chunk_index"]) for r in done["references"]]
        assert keys == [("d3", 0), ("d1", 0), ("d2", 0)], "应去重并按相似度降序"
        scores = [r["score"] for r in done["references"]]
        assert scores == sorted(scores, reverse=True)

    def test_展示用引用数量有上限(self):
        many = [ref("d", i, 0.5 + i * 0.001) for i in range(12)]
        reg, _ = make({"q1": many})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["回答。"]])
        _, done, _ = run(Agent(llm, reg, ContextManager()))
        assert len(done["references"]) == 8


# ----------------------------------------------------------------------
# 六、事件协议
# ----------------------------------------------------------------------


class TestEvents:
    def test_事件序列完整且内部事件不外泄(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["回答内容。"]])
        events, _, _ = run(Agent(llm, reg, ContextManager()))
        names = [e["event"] for e in events]

        assert names[0] == "status"
        assert names[-1] == "done"
        assert names.count("tool_call") == 1
        assert names.count("tool_result") == 1
        assert "_verdict" not in names, "内部判别事件不应外泄"
        assert "chat" not in names

    def test_状态事件按轮次递增(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)], "q2": [ref("d2", 0, 0.8)]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                [tool_call("kb_search", query="q2")],
                ["回答。"],
            ]
        )
        events, _, _ = run(Agent(llm, reg, ContextManager()))
        routing_rounds = [
            e["data"]["round"]
            for e in events
            if e["event"] == "status" and e["data"]["stage"] == "routing"
        ]
        assert routing_rounds == [1, 2, 3]

    def test_收尾轮带answering状态(self):
        # 模型只回了 {"tool": null}（没给回答）才会走收尾轮；
        # 若它在本轮直接作答，则不会有额外一轮，也就没有 answering 状态
        reg, _ = make()
        llm = ScriptedLLM([['{"tool": null}'], ["回答。"]])
        events, _, _ = run(Agent(llm, reg, ContextManager()))
        assert "answering" in [e["data"]["stage"] for e in events if e["event"] == "status"]

    def test_本轮直接作答则无收尾轮状态(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["回答。"]])
        events, _, _ = run(Agent(llm, reg, ContextManager()))
        stages = [e["data"]["stage"] for e in events if e["event"] == "status"]
        assert "answering" not in stages, "本轮已作答，不应再出现收尾轮"

    def test_工具调用与结果事件带上轮次与数量(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9), ref("d2", 0, 0.8)]})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["回答。"]])
        events, _, _ = run(Agent(llm, reg, ContextManager()))

        call = next(e["data"] for e in events if e["event"] == "tool_call")
        result = next(e["data"] for e in events if e["event"] == "tool_result")
        assert call["round"] == 1 and call["tool"] == "kb_search"
        assert call["arguments"] == {"query": "q1"}
        assert result["round"] == 1 and result["count"] == 2


# ----------------------------------------------------------------------
# 七、检索为空：必须如实告知模型「没找到」而不是默默作答
# ----------------------------------------------------------------------


class TestEmptyRetrieval:
    def test_检索结果全被阈值过滤时注入未找到说明(self):
        reg, _ = make({"q1": []})  # 工具返回 0 条（相似度均低于阈值）
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["知识库中未找到相关内容。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()), "知识库里没有的问题")

        obs = [m["content"] for m in llm.requests[1] if m["content"].startswith("Observation")]
        assert obs, "必须注入一条「无结果」Observation，否则模型会凭空作答"
        assert "未检索到相关内容" in obs[0]
        assert "阈值" in obs[0]
        assert done["rounds"] == 1, "工具确实被调用过，轮次应计入"
        assert done["references"] == []
        assert answer == "知识库中未找到相关内容。"

    def test_检索为空时仍会走收尾轮(self):
        reg, _ = make({"q1": []})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["未找到。"]])
        events, _, _ = run(Agent(llm, reg, ContextManager()))
        stages = [e["data"]["stage"] for e in events if e["event"] == "status"]
        assert "answering" in stages
        assert llm.call_count == 2, "一轮检索 + 一轮收尾作答"

    def test_片段全部重复时说明无新增而非未找到(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)], "q2": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                [tool_call("kb_search", query="q2")],
                ["回答。"],
            ]
        )
        _, done, _ = run(Agent(llm, reg, ContextManager()))

        obs = [m["content"] for m in llm.requests[2] if m["content"].startswith("Observation")]
        assert any("重复" in o for o in obs), "重复片段应说明「无新增信息」，不能被误报成未找到"
        assert not any("未检索到相关内容" in o for o in obs)
        assert len(done["references"]) == 1

    def test_工具执行失败时注入失败说明(self):
        reg = ToolRegistry()

        async def boom(query: str = ""):
            raise RuntimeError("检索炸了")

        reg.register(
            Tool(
                name="kb_search",
                description="知识库检索",
                parameters=[ToolParameter(name="query", type="string", description="检索词")],
                handler=boom,
            )
        )
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["兜底回答。"]])
        _, _, answer = run(Agent(llm, reg, ContextManager()))

        obs = [m["content"] for m in llm.requests[1] if m["content"].startswith("Observation")]
        assert any("失败" in o for o in obs)
        assert answer == "兜底回答。"

    def test_未调用工具时不会注入任何Observation(self):
        reg, _ = make()
        llm = ScriptedLLM([["你好，有什么可以帮你的？"]])
        run(Agent(llm, reg, ContextManager()), "你好")
        assert not [m for m in llm.requests[0] if m["content"].startswith("Observation")]


# ----------------------------------------------------------------------
# 八、多工具编排：Agent 不写死工具形态，靠 references/summary 约定分发
# ----------------------------------------------------------------------


class TestMultiTool:
    def test_非检索类工具结果用工具自述讲给模型(self):
        reg, executed = make()
        add_doc_list_tool(reg, "知识库共有 2 份文档：\n《a.md》\n《b.md》", executed)
        llm = ScriptedLLM([[tool_call("doc_list")], ["知识库里有两份文档：a.md 和 b.md。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()))

        obs = [m["content"] for m in llm.requests[1] if m["content"].startswith("Observation")]
        assert obs, "非检索类工具的结果同样要注入 Observation"
        assert "工具返回" in obs[0]
        assert "《a.md》" in obs[0]
        assert done["references"] == [], "非检索类工具不应产生引用来源"
        assert answer == "知识库里有两份文档：a.md 和 b.md。"

    def test_非检索类工具成功执行后继续编排而不立即收尾(self):
        reg, executed = make({"q1": [ref("d1", 0, 0.9)]})
        add_doc_list_tool(reg, "知识库共有 2 份文档。", executed)
        llm = ScriptedLLM(
            [
                [tool_call("doc_list")],
                [tool_call("kb_search", query="q1")],
                ["综合文档清单与检索结果，答案是 A。"],
            ]
        )
        _, done, answer = run(Agent(llm, reg, ContextManager()))

        assert executed == ["doc_list", "q1"], "应当先列清单再检索，两个工具都被执行"
        assert done["tool_calls"] == ["doc_list", "kb_search"]
        assert done["rounds"] == 2
        assert len(done["references"]) == 1
        assert llm.call_count == 3

    def test_同一工具调用重复发起则提前终止(self):
        reg, executed = make()
        add_doc_list_tool(reg, "知识库共有 2 份文档。", executed)
        llm = ScriptedLLM(
            [
                [tool_call("doc_list")],
                [tool_call("doc_list")],  # 与上一轮完全相同 → 空转
                ["回答。"],
            ]
        )
        _, done, _ = run(Agent(llm, reg, ContextManager()))

        assert executed == ["doc_list"], "重复的调用不应被再次执行"
        assert done["rounds"] == 1
        assert llm.call_count == 3, "第二次重复调用被拦下，转入收尾作答"

    def test_参数不同则不算重复(self):
        reg, executed = make({"q1": [ref("d1", 0, 0.9)], "q2": [ref("d2", 0, 0.8)]})
        llm = ScriptedLLM(
            [
                [tool_call("kb_search", query="q1")],
                [tool_call("kb_search", query="q2")],
                ["回答。"],
            ]
        )
        _, done, _ = run(Agent(llm, reg, ContextManager()))
        assert executed == ["q1", "q2"]
        assert done["rounds"] == 2

    def test_工具失败时不再继续空转(self):
        reg = ToolRegistry()

        async def boom(query: str = ""):
            raise RuntimeError("炸了")

        reg.register(
            Tool(
                name="kb_search",
                description="检索",
                parameters=[ToolParameter(name="query", type="string", description="检索词")],
                handler=boom,
            )
        )
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["兜底回答。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()))

        obs = [m["content"] for m in llm.requests[1] if m["content"].startswith("Observation")]
        assert any("执行失败" in o for o in obs)
        assert done["rounds"] == 1
        assert answer == "兜底回答。"

    def test_无参数工具不会因参数兜底而误传(self):
        # registry 对模型传错的参数做兜底裁剪；无参数工具应收到空 kwargs
        reg = ToolRegistry()
        received: list[dict] = []

        async def no_arg_tool():
            received.append({"called": True})
            return {"count": 0, "summary": "无参数工具被调用。"}

        reg.register(Tool(name="no_arg", description="无参数", parameters=[], handler=no_arg_tool))
        llm = ScriptedLLM([[tool_call("no_arg")], ["回答。"]])
        run(Agent(llm, reg, ContextManager()))

        assert received == [{"called": True}]


# ----------------------------------------------------------------------
# 九、异常与边界
# ----------------------------------------------------------------------


class TestRobustness:
    def test_工具执行失败不中断对话(self):
        reg = ToolRegistry()

        async def boom(query: str = ""):
            raise RuntimeError("检索炸了")

        reg.register(
            Tool(
                name="kb_search",
                description="知识库检索",
                parameters=[ToolParameter(name="query", type="string", description="检索词")],
                handler=boom,
            )
        )
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["兜底回答。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()))
        assert answer == "兜底回答。", "工具异常时应继续给出回答"
        assert done["rounds"] == 1

    def test_模型请求不存在的工具时转为收尾作答(self):
        reg, _ = make()
        llm = ScriptedLLM([[tool_call("不存在的工具", query="q1")], ["兜底回答。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()))
        assert done["rounds"] == 1, "仍会记一次调用，但工具返回 error"
        assert answer == "兜底回答。"

    def test_没有可用工具时直接作答(self):
        reg = ToolRegistry()
        llm = ScriptedLLM([["无工具也能回答。"]])
        _, done, answer = run(Agent(llm, reg, ContextManager()))
        assert answer == "无工具也能回答。"
        assert llm.call_count == 1

    def test_空的历史记录不影响运行(self):
        reg, _ = make({"q1": [ref("d1", 0, 0.9)]})
        llm = ScriptedLLM([[tool_call("kb_search", query="q1")], ["回答。"]])
        _, done, _ = run(Agent(llm, reg, ContextManager()))
        assert done["rounds"] == 1
