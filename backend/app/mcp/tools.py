"""内置 MCP 工具。

三个工具覆盖三类不同的能力，用来体现「注册即用、模型自主选择」：
    knowledge_search     语义检索（有参数、会返回可引用的片段）
    list_documents       文档清单（无参数、返回元信息）
    get_document_content 按文件名读正文（必填字符串参数 + 可选区间）

统一返回约定：
    summary      模型面向的一行/一段描述（必填），Agent 会把它包成 Observation
    references   仅检索类工具提供；Agent 据此收集「引用来源」，并做跨轮去重
    count        命中数量，供前端展示
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..config import settings
from ..db.sqlite import get_db
from ..rag.store import get_store
from .protocol import Tool, ToolParameter
from .registry import registry

# get_document_content 默认一次返回的片段数，避免动辄把整篇文档灌进上下文
DEFAULT_DOC_CHUNK_LIMIT = 5

BUILTIN_TOOL_NAMES = ("knowledge_search", "list_documents", "get_document_content")


async def knowledge_search(query: str, top_k: int = 0) -> dict:
    """在知识库中做语义检索，返回最相关的原文片段。

    检索是 CPU/IO 密集操作，放到线程池执行避免阻塞事件循环。
    命中片段还会向后补齐相邻上下文：一道长答案常被切成多段，
    只召回命中段会让模型只答出其中一部分（缺头少尾）。
    """
    store = get_store()
    k = int(top_k) if top_k and int(top_k) > 0 else settings.top_k
    hits = await asyncio.to_thread(store.search, query, k)

    if not hits:
        return {
            "count": 0,
            "references": [],
            "summary": (
                f"未检索到相关内容（返回片段的相似度均低于相关性阈值 {settings.min_score}）。"
            ),
        }

    contexts = await asyncio.to_thread(_expand_neighbors, store, hits)
    return {
        "count": len(hits),
        "references": contexts,
        "summary": (
            f"检索到 {len(hits)} 个与问题相关的原文片段，"
            f"并补齐其相邻上下文，共提供 {len(contexts)} 段原文。"
        ),
    }


def _expand_neighbors(store: Any, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把命中片段扩展成「命中段 + 相邻上下文段」，合并去重并截断。

    - 按命中顺序（相似度降序）逐条扩展，保证高相关片段排在最前；
    - 扩展段没有相似度可言，score 置 None，并标 expanded=True，
      这样 Agent 拼 Observation 时能区分「命中段」与「上下文段」，不会编造分数；
    - 总片段数受 max_context_chunks 约束，避免长文档把上下文灌满。
    """
    window = max(int(settings.context_window), 0)
    cap = max(int(settings.max_context_chunks), 1)

    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for hit in hits:
        hit_key = (hit.get("doc_id"), hit.get("chunk_index"))
        rows = store.chunk_window(hit["filename"], hit["chunk_index"], window, window)
        for row in rows:
            key = (row.get("doc_id"), row.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "text": row.get("text"),
                    "doc_id": row.get("doc_id"),
                    "filename": row.get("filename"),
                    "chunk_index": row.get("chunk_index"),
                    "score": hit.get("score") if key == hit_key else None,
                    "expanded": key != hit_key,
                }
            )
            if len(merged) >= cap:
                return merged
    return merged


async def list_documents() -> dict:
    """列出知识库里已上传的文档清单（文件名、片段数、上传时间）。"""
    docs = get_db().list_documents()
    if not docs:
        return {"count": 0, "documents": [], "summary": "知识库当前没有任何文档。"}

    lines = [
        f"《{d['filename']}》（{d['chunks']} 个片段，上传于 {d['created_at']}）" for d in docs
    ]
    return {
        "count": len(docs),
        "documents": docs,
        "summary": f"知识库共有 {len(docs)} 份文档：\n" + "\n".join(lines),
    }


async def get_document_content(filename: str, start: int = 0, limit: int = 0) -> dict:
    """读取指定文档的片段正文，可按片段序号取一个区间。

    用户明确点名某份文档（或需要展开某文档的内容）时使用；
    只是想找「哪段话能回答问题」应改用 knowledge_search。
    """
    store = get_store()
    offset = max(int(start or 0), 0)
    size = int(limit) if limit and int(limit) > 0 else DEFAULT_DOC_CHUNK_LIMIT

    matched, chunks = await asyncio.to_thread(store.document_chunks, filename, offset, size)

    if matched is None:
        return {
            "count": 0,
            "summary": f"知识库中没有找到名为「{filename}」的文档，无法读取其内容。",
        }
    if not chunks:
        return {
            "count": 0,
            "summary": f"《{matched}》在片段序号 {offset} 之后没有更多内容了。",
        }

    blocks = [f"[片段 {c['chunk_index']}]\n{c['text']}" for c in chunks]
    return {
        "count": len(chunks),
        "filename": matched,
        "chunks": chunks,
        "summary": (
            f"《{matched}》从片段 {chunks[0]['chunk_index']} 起共取到 {len(chunks)} 段正文：\n\n"
            + "\n\n".join(blocks)
        ),
    }


def _build_tools() -> list[Tool]:
    return [
        Tool(
            name="knowledge_search",
            description=(
                "在用户已上传的文档知识库中做语义检索，返回与问题最相关的原文片段。"
                "当问题涉及具体文档内容、需要事实依据时，优先使用这个工具。"
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="检索用的自然语言问题或关键词",
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="返回片段数量，默认由服务端配置决定",
                    required=False,
                ),
            ],
            handler=knowledge_search,
        ),
        Tool(
            name="list_documents",
            description=(
                "列出知识库里已上传的文档清单（文件名、片段数、上传时间）。"
                "当用户询问「知识库有哪些文档」「有几份文档」「有没有某某文档」时使用，不要用它检索内容。"
            ),
            parameters=[],
            handler=list_documents,
        ),
        Tool(
            name="get_document_content",
            description=(
                "按文件名读取指定文档的正文片段，可按片段序号取一个区间。"
                "当用户明确点名某份文档（如「看看 README.md 讲了什么」）或需要展开某文档内容时使用；"
                "若只是想找哪段话能回答问题，应使用 knowledge_search。"
            ),
            parameters=[
                ToolParameter(
                    name="filename",
                    type="string",
                    description="文档文件名，可与实际文件名部分匹配（如「README」可匹配 README.md）",
                ),
                ToolParameter(
                    name="start",
                    type="integer",
                    description="起始片段序号，默认 0（从开头读）",
                    required=False,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description=f"读取片段数，默认 {DEFAULT_DOC_CHUNK_LIMIT}",
                    required=False,
                ),
            ],
            handler=get_document_content,
        ),
    ]


def register_builtin_tools() -> None:
    """注册内置工具（幂等）。新增能力只需在这里多注册一个 Tool，Agent 无需改动。"""
    if all(registry.get(name) for name in BUILTIN_TOOL_NAMES):
        return
    for tool in _build_tools():
        registry.register(tool)
