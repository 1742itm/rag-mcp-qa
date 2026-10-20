"""Agent 编排：ReAct 风格的单次调用循环。

与「先路由决策、再单独作答」的两段式不同，这里让模型在**同一次流式调用**里
自己决定这一轮输出的是「工具调用」还是「最终回答」：

    流式接收 → 判别器分流
      ├─ TOOL     → 执行工具，把 Observation 追加进对话，进入下一轮
      ├─ ANSWER   → 已边收边展示完，直接收尾
      └─ NO_TOOL  → 模型没给出回答，进入收尾轮重新索取

为什么能省一次 LLM 往返：旧方案每轮都要一次**独立的非流式决策调用**，
最后还要再来一次作答调用；这里模型作答的那一轮本身就是流式回答，
所以 N 轮工具调用总共只需 N+1 次调用。

工具是通用的：Agent 不关心具体有哪些工具，也不写死「结果里必须有片段」。
它只按约定的字段形态（`references` / `summary`）把结果讲给模型，
因此新增工具无需改动本文件。

收敛保障：
  1. 调用阶段最多 MAX_TOOL_ROUNDS 轮；
  2. 同一工具调用签名重复 → 提前终止；
  3. 检索类工具没带回新片段 → 提前终止；
  4. 收尾轮**不暴露工具清单**，模型无法再产出工具调用，必然收敛。
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from .context import ContextManager
from .discriminator import Classification, Verdict, classify
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

# 检索阶段允许的最大工具调用轮数
MAX_TOOL_ROUNDS = 3
# 对外展示与落库的引用片段上限（按相似度降序取前 N 条），避免列表与存储膨胀
MAX_CONTEXT_REFERENCES = 8


class Agent:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        context_manager: ContextManager,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.context = context_manager

    async def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行一轮问答（内部可能包含多轮检索），产出 SSE 事件字典。"""
        history = history or []
        tools_prompt = self.registry.prompt_description()
        transcript: list[dict[str, str]] = []          # 模型原始输出 + Observation
        references: list[dict[str, Any]] = []          # 累积引用片段（已去重）
        seen_chunks: set[tuple[Any, Any]] = set()      # 已收录片段的 (doc_id, chunk_index)
        issued_calls: set[str] = set()                 # 已发起过的工具调用签名（防重复）
        called_tools: list[str] = []
        rounds = 0
        answered = False

        # ---- 检索阶段：每轮都可能「调用工具」或「直接作答」----
        while rounds < MAX_TOOL_ROUNDS:
            yield {"event": "status", "data": {"stage": "routing", "round": rounds + 1}}

            messages = self.context.build_react_messages(
                question, history, tools_prompt, transcript, MAX_TOOL_ROUNDS
            )
            classification, raw = None, ""
            async for ev in self._consume(messages):
                if ev["event"] == "delta":
                    yield ev
                else:
                    classification, raw = ev["data"]["classification"], ev["data"]["raw"]

            if classification is None:
                break

            if classification.verdict is Verdict.ANSWER:
                # 回答已随流展示完毕，无需再请求一次
                answered = True
                break

            if classification.verdict is not Verdict.TOOL:
                # NO_TOOL：模型表示无须工具却没给出回答，交给收尾轮索取
                logger.info("模型未自行作答（%s），转入收尾轮", classification.verdict.value)
                break

            # ---- 执行工具调用 ----
            signature = json.dumps(
                {"tool": classification.tool, "arguments": classification.arguments},
                sort_keys=True,
                ensure_ascii=False,
            )
            if signature in issued_calls:
                # 同一调用重复发起说明模型没有进展，再循环只是空转
                logger.info("模型重复发起同一工具调用（%s），结束编排", classification.tool)
                break
            issued_calls.add(signature)

            rounds += 1
            yield {
                "event": "tool_call",
                "data": {
                    "tool": classification.tool,
                    "arguments": classification.arguments,
                    "round": rounds,
                },
            }
            result = await self.registry.execute(classification.tool, classification.arguments)
            called_tools.append(classification.tool or "")
            fresh = self._fresh_references(result, seen_chunks)

            yield {
                "event": "tool_result",
                "data": {
                    "tool": classification.tool,
                    # 工具自报的结果条数更准确；检索类工具因阈值/去重可能少于它
                    "count": result.get("count", len(fresh)) if isinstance(result, dict) else len(fresh),
                    "fresh": len(fresh),
                    "round": rounds,
                },
            }

            transcript.append({"role": "assistant", "content": raw})
            transcript.append(self._build_observation(result, fresh))

            if fresh:
                references.extend(fresh)
                continue

            if isinstance(result, dict) and result.get("error"):
                # 工具本身坏了，重试大概率还是坏，直接转入作答并如实告知
                logger.info("第 %d 轮工具执行失败，结束编排转入作答", rounds)
                break

            if self._is_retrieval(result):
                # 检索类工具没带回新片段（相似度都低于阈值，或与已有片段重复）：
                # 再检索也无收益，转入收尾作答
                logger.info("第 %d 轮未获得可用片段，结束检索进入作答", rounds)
                break

            # 非检索类工具（文档清单 / 读取正文）成功执行，属于有效进展，交由模型决定下一步
            logger.info("第 %d 轮执行了非检索工具 %s，继续编排", rounds, classification.tool)

        # ---- 收尾阶段：不暴露工具，确保拿到回答 ----
        if not answered:
            yield {"event": "status", "data": {"stage": "answering", "round": rounds}}
            messages = self.context.build_react_messages(
                question, history, None, transcript, MAX_TOOL_ROUNDS
            )
            classification, raw = None, ""
            async for ev in self._consume(messages):
                if ev["event"] == "delta":
                    yield ev
                else:
                    classification, raw = ev["data"]["classification"], ev["data"]["raw"]

            if classification is not None and classification.verdict is not Verdict.ANSWER:
                logger.warning("收尾轮仍未直接作答（%s），返回兜底提示", classification.verdict.value)
                yield {
                    "event": "delta",
                    "data": {"content": "（模型未按要求直接作答，请重试）"},
                }

        yield {
            "event": "done",
            "data": {
                "references": self._rank_and_cap(references),
                "tool_calls": called_tools,
                "rounds": rounds,
            },
        }

    # ------------------------------------------------------------------
    async def _consume(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[dict[str, Any]]:
        """订阅一次模型输出，边收边判别。

        - 可安全展示的正文以 `delta` 事件产出；
        - 最后以一个内部 `_verdict` 事件给出本轮结论（由 run 消费，不转发给前端）。

        判别器保证「工具调用 JSON」不会被当作正文流出；一旦判定为回答即进入 latch，
        后续文本直接透传、不再回头判别（避免展示到一半又被改判成工具调用）。
        """
        buffer = ""
        latched = False

        async for chunk in self.llm.stream(messages):
            buffer += chunk
            if latched:
                yield {"event": "delta", "data": {"content": chunk}}
                continue
            if classify(buffer).verdict is Verdict.ANSWER:
                latched = True
                yield {"event": "delta", "data": {"content": buffer}}

        if latched:
            classification = Classification(Verdict.ANSWER)
        else:
            classification = classify(buffer)
            if classification.verdict is Verdict.UNKNOWN:
                if "{" in buffer:
                    # 出现了 JSON 候选却始终没闭合：不展示半截 JSON，
                    # 判为本轮未产出有效结果，交收尾轮重新索取回答
                    logger.info("模型输出了未闭合的 JSON，已丢弃并转入收尾轮")
                    classification = Classification(Verdict.NO_TOOL)
                else:
                    # 很短的正文，流已结束：直接展示
                    yield {"event": "delta", "data": {"content": buffer}}
                    classification = Classification(Verdict.ANSWER)

        yield {
            "event": "_verdict",
            "data": {"classification": classification, "raw": buffer},
        }

    # ------------------------------------------------------------------
    def _build_observation(self, result: Any, fresh: list[dict[str, Any]]) -> dict[str, str]:
        """按工具结果的形态，选一种方式把它讲给模型听。

        - 工具失败            → 说明失败原因
        - 检索到新片段        → 结构化片段（带来源与相似度）
        - 检索类但无新片段     → 工具自述 + 「与之前重复」的补充
        - 非检索类工具        → 直接用工具自述（如文档清单、正文片段）
        """
        if isinstance(result, dict) and result.get("error"):
            return self.context.build_no_result_observation(f"工具执行失败：{result['error']}")

        if fresh:
            return self.context.build_observation(fresh)

        if self._is_retrieval(result):
            summary = str(result.get("summary") or "").strip() or "本轮没有获得新的相关内容。"
            if result.get("references"):
                summary = f"{summary} 但这些片段与之前已获得的内容重复，没有新增信息。"
            return self.context.build_no_result_observation(summary)

        summary = result.get("summary") if isinstance(result, dict) else None
        return self.context.build_tool_observation(str(summary or "工具没有返回可用内容。"))

    @staticmethod
    def _is_retrieval(result: Any) -> bool:
        """带 references 字段的即为检索类工具，其片段要纳入引用来源。"""
        return isinstance(result, dict) and isinstance(result.get("references"), list)

    @staticmethod
    def _fresh_references(
        result: Any,
        seen_chunks: set[tuple[Any, Any]],
    ) -> list[dict[str, Any]]:
        """取出本轮结果中尚未收录过的片段，并登记去重标记。"""
        if not isinstance(result, dict):
            return []
        refs = result.get("references")
        if not isinstance(refs, list):
            return []

        fresh: list[dict[str, Any]] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            key = (ref.get("doc_id"), ref.get("chunk_index"))
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            fresh.append(ref)
        return fresh

    @staticmethod
    def _rank_and_cap(
        references: list[dict[str, Any]],
        limit: int = MAX_CONTEXT_REFERENCES,
    ) -> list[dict[str, Any]]:
        """按相似度降序排列并截断，供前端展示与落库。"""
        ranked = sorted(references, key=lambda r: r.get("score") or 0.0, reverse=True)
        return ranked[:limit]
