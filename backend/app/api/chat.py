"""对话接口：SSE 流式问答。

使用 GET + query 参数，是为了让前端可以直接用原生 EventSource 订阅。
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from ..db.sqlite import get_db
from ..llm.client import LLMError
from ..mcp import build_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse(event: str, data: dict) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@router.get("/stream")
async def chat_stream(
    message: str = Query(..., min_length=1, description="用户问题"),
    conversation_id: str | None = Query(None, description="会话 ID，为空则新建"),
) -> EventSourceResponse:
    db = get_db()
    conv = db.ensure_conversation(conversation_id, message)
    db.add_message(conv["id"], "user", message)
    # 历史消息中排除刚写入的这条用户消息
    history = db.history_pairs(conv["id"])[:-1]
    agent = build_agent()

    async def event_generator():
        yield _sse("meta", {"conversation_id": conv["id"], "title": conv["title"]})

        answer = ""
        references: list[dict] = []
        tool_calls: list[str] = []
        rounds = 0
        try:
            async for ev in agent.run(message, history):
                name, data = ev["event"], ev["data"]
                if name == "delta":
                    answer += data.get("content", "")
                    yield _sse("delta", data)
                elif name == "done":
                    references = data.get("references", [])
                    tool_calls = data.get("tool_calls", [])
                    rounds = data.get("rounds", 0)
                else:
                    yield _sse(name, data)
        except LLMError as exc:
            logger.warning("LLM 调用失败: %s", exc)
            yield _sse("error", {"message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("对话生成异常")
            yield _sse("error", {"message": f"服务异常：{exc}"})
            return

        saved = db.add_message(conv["id"], "assistant", answer, references)
        yield _sse(
            "done",
            {
                "message_id": saved["id"],
                "conversation_id": conv["id"],
                "title": conv["title"],
                "references": references,
                "tool_calls": tool_calls,
                "rounds": rounds,
            },
        )

    return EventSourceResponse(event_generator(), ping=15000)