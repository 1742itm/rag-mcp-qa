"""MCP 上下文管理：统一拼装喂给 LLM 的消息序列。

ReAct 模式下只有一个提示词入口（`build_react_messages`），
由 `tools_prompt` 是否传入切换两种形态：

  - 传入工具清单  → 「可以调用工具」的循环提示词（检索阶段使用）
  - 不传工具清单  → 「不要再调用工具，直接作答」的收尾提示词

收尾轮刻意**不暴露工具**：模型看不到工具清单，就无法再产出工具调用，
从而保证必然收敛、一定能拿到回答。
"""
from __future__ import annotations

from typing import Any

MAX_HISTORY_MESSAGES = 20

# ---- 检索阶段：告知工具与 ReAct 格式约定 ----
REACT_SYSTEM = """你是一个严谨的知识库问答助手，具备调用工具检索资料的能力。

可用工具：
{tools}

【工作方式】你可以进行多轮检索，最多 {max_rounds} 轮：
1. 需要查阅资料时：**只输出一行 JSON**，格式为
{{"tool": "工具名", "arguments": {{"参数名": "参数值"}}}}
   JSON 必须是你这一轮的**全部**输出——前后不要有任何文字、说明、寒暄，也不要加代码块围栏。
2. 资料已足够，或问题本身不需要查资料时：**直接输出回答正文**，不要输出任何 JSON。
3. 【先检索再作答】只要问题涉及知识库里的文档内容（面试题、资料、术语、方案、数据等），
   就必须先调用工具检索，再依据检索结果回答。
   即使用户的问法像是在问你自己（如「你平时是怎么查看日志的」「你怎么做性能优化的」），
   也应理解为在问文档里对应的内容，不得跳过检索直接凭常识作答；
   只有纯寒暄或与文档无关的闲聊（如「你好」「你是谁」）才可以直接回答。
4. 每次工具调用后，你会收到一条以 `Observation:` 开头的消息，内含检索到的原文片段；
   据此判断能否作答，或需要换关键词、换角度继续检索。

【回答要求】
1. 优先依据 Observation 中的检索结果回答，不要编造资料里不存在的内容；
2. 若 Observation 提示「未检索到相关内容」，说明知识库中没有匹配的资料，
   此时应如实说明，不要凭常识编造并伪装成来自文档；
3. 若资料不足以回答，明确说明「根据现有资料无法确定」，不要臆测；
4. 回答使用简洁的中文，可适当分点，便于阅读；
5. 可以提到信息来源的文档名，但不要虚构来源。"""

# ---- 收尾阶段：不暴露工具，强制直接作答 ----
FINAL_ANSWER_SYSTEM = """你是一个严谨的知识库问答助手。

【重要】本次**不要调用任何工具，也不要输出任何 JSON**，直接输出回答正文。

【回答要求】
1. 若对话中已有检索结果（Observation），优先据此回答，不要编造资料里不存在的内容；
2. 若 Observation 提示「未检索到相关内容」，说明知识库中没有匹配的资料，
   此时应如实说明，不要凭常识编造并伪装成来自文档；
3. 若对话中完全没有检索结果，说明问题与知识库无关或无需查资料，可正常回答；
4. 回答使用简洁的中文，可适当分点，便于阅读；
5. 可以提到信息来源的文档名，但不要虚构来源。"""


class ContextManager:
    """构建 LLM 消息序列。"""

    def __init__(self, max_history: int = MAX_HISTORY_MESSAGES) -> None:
        self.max_history = max_history

    def _trim(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        return history[-self.max_history :] if history else []

    def _base_messages(
        self,
        system: str,
        question: str,
        history: list[dict[str, str]],
        transcript: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """公共骨架：system → 历史 → 本轮问题 → 本轮的工具调用/观测轨迹。"""
        messages = [{"role": "system", "content": system}]
        messages.extend(self._trim(history))
        messages.append({"role": "user", "content": question})
        if transcript:
            messages.extend(transcript)
        return messages

    def build_react_messages(
        self,
        question: str,
        history: list[dict[str, str]],
        tools_prompt: str | None = None,
        transcript: list[dict[str, str]] | None = None,
        max_rounds: int = 3,
    ) -> list[dict[str, str]]:
        """构建一轮 ReAct 消息。

        tools_prompt 为空表示收尾轮——不暴露工具，模型无法再调用工具。
        transcript 为前面各轮「模型原始输出 + Observation」的累积。
        """
        if tools_prompt:
            system = REACT_SYSTEM.format(tools=tools_prompt, max_rounds=max_rounds)
        else:
            system = FINAL_ANSWER_SYSTEM
        return self._base_messages(system, question, history, transcript)

    def build_observation(self, fragments: list[dict[str, Any]]) -> dict[str, str]:
        """把本轮检索到的片段格式化成一条 Observation 消息。

        只标注来源文档名、不编号，避免模型引用出与前端列表对不上的序号。
        命中段带相似度；为补齐长答案而扩展出来的相邻段没有相似度，
        必须显式标注，否则模型会把它们当成同样相关的内容。
        """
        blocks = []
        for ref in fragments:
            source = ref.get("filename") or "未知文档"
            score = ref.get("score")
            head = f"来源《{source}》"
            if score is not None:
                head += f"（相似度 {score}）"
            elif ref.get("expanded"):
                head += "（命中片段的相邻上下文）"
            blocks.append(f"{head}\n{(ref.get('text') or '').strip()}")
        content = "Observation（本轮检索结果）:\n" + "\n\n".join(blocks)
        return {"role": "user", "content": content}

    def build_no_result_observation(self, reason: str) -> dict[str, str]:
        """构造「本轮没有拿到可用资料」的 Observation。

        必须有这条消息：否则模型只看到问题就直接作答，
        会把常识性内容讲得像是从文档里查到的——这正是相似度阈值要防的幻觉。
        """
        content = (
            "Observation（本轮检索结果）:\n"
            f"{reason}\n"
            "请据此如实回答：若问题需要依据知识库内容而检索没有返回结果，"
            "应说明「知识库中未找到相关内容」，不要凭空编造。"
        )
        return {"role": "user", "content": content}

    def build_tool_observation(self, content: str) -> dict[str, str]:
        """包装非检索类工具（如文档清单、读取正文）的返回，供模型据此作答。"""
        return {"role": "user", "content": f"Observation（工具返回）:\n{content}"}


context_manager = ContextManager()
