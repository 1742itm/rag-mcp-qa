"""Chroma 本地向量库封装。

说明：向量由我们自己的 Embedder 计算后显式写入，
因此 Chroma 不会去下载它的默认 ONNX 模型，真正做到离线、轻量。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings
from .embedder import BaseEmbedder, get_embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge"


class VectorStore:
    def __init__(self, persist_dir: str, embedder: BaseEmbedder | None = None) -> None:
        # embedder 可注入，便于测试用确定性的哈希向量替代真实模型
        self.embedder = embedder or get_embedder(
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_dim,
        )
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量库就绪: %s (embedder=%s, dim=%d)",
                    persist_dir, self.embedder.name, self.embedder.dim)

    # ------------------------------------------------------------------
    def add_document(self, doc_id: str, filename: str, chunks: list[str]) -> int:
        """写入一个文档的所有切片，返回切片数。"""
        if not chunks:
            return 0
        embeddings = self.embedder.embed(chunks)
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索，返回带相似度与出处的结果列表。

        相似度低于 min_score 的片段会被丢弃：宁可不给资料，也不要把无关片段
        当作"参考"喂给模型——那只会把噪声变成新的幻觉。
        min_score 传 None 时取配置默认值；传负数可拿到完整结果（评测/排查用）。
        """
        top_k = top_k or settings.top_k
        threshold = self._default_threshold() if min_score is None else min_score
        if self.collection.count() == 0:
            return []

        embedding = self.embedder.embed_one(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        dropped = 0
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            score = round(1.0 - float(dist), 4)
            if score < threshold:
                dropped += 1
                continue
            hits.append(
                {
                    "text": text,
                    "doc_id": (meta or {}).get("doc_id"),
                    "filename": (meta or {}).get("filename"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "score": score,
                }
            )

        if dropped:
            logger.info(
                "检索「%s」命中 %d 条，其中 %d 条相似度低于阈值 %.2f 已丢弃",
                query, dropped + len(hits), dropped, threshold,
            )
        return hits

    def _default_threshold(self) -> float:
        """默认阈值随向量化器切换。

        哈希向量与语义向量的分数尺度完全不同（差约 0.25），用同一个阈值必然有一方失效，
        因此降级模式单独取一个更低的阈值。
        """
        if self.embedder.name == "hash":
            return settings.min_score_hash
        return settings.min_score

    def document_chunks(
        self,
        filename: str,
        start: int = 0,
        limit: int | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """按文件名取出该文档的片段（按片段序号升序）。

        先做精确匹配；未命中时退化为「文件名包含关系」的模糊匹配，
        这样用户说「README 文档」也能命中实际文件名 README.md。
        返回 (实际匹配到的文件名, 片段列表)；完全找不到时文件名为 None。
        """
        if self.collection.count() == 0:
            return None, []

        result = self.collection.get(
            where={"filename": filename}, include=["documents", "metadatas"]
        )
        matched = filename
        if not result["ids"]:
            matched = self._match_filename(filename)
            if matched is None:
                return None, []
            result = self.collection.get(
                where={"filename": matched}, include=["documents", "metadatas"]
            )

        rows: list[dict[str, Any]] = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            rows.append(
                {
                    "text": text,
                    "filename": (meta or {}).get("filename"),
                    "doc_id": (meta or {}).get("doc_id"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                }
            )
        rows.sort(key=lambda r: r["chunk_index"] or 0)

        offset = max(int(start or 0), 0)
        if limit is None or limit <= 0:
            return matched, rows[offset:]
        return matched, rows[offset : offset + limit]

    def chunk_window(
        self,
        filename: str,
        chunk_index: int,
        before: int = 0,
        after: int = 0,
    ) -> list[dict[str, Any]]:
        """取某片段所在文档内的相邻上下文（含自身），按片段序号升序返回。

        长答案常被切成多段：只把命中的那几段喂给模型，回答就会缺头少尾。
        扩展相邻段可以把连续的一整块答案补齐，同时不引入其他文档的内容。
        """
        if self.collection.count() == 0:
            return []

        lo = max(int(chunk_index) - max(int(before), 0), 0)
        hi = int(chunk_index) + max(int(after), 0)
        result = self.collection.get(
            where={
                "$and": [
                    {"filename": filename},
                    {"chunk_index": {"$gte": lo}},
                    {"chunk_index": {"$lte": hi}},
                ]
            },
            include=["documents", "metadatas"],
        )

        rows: list[dict[str, Any]] = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            rows.append(
                {
                    "text": text,
                    "filename": (meta or {}).get("filename"),
                    "doc_id": (meta or {}).get("doc_id"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                }
            )
        rows.sort(key=lambda r: r["chunk_index"] or 0)
        return rows

    def _match_filename(self, keyword: str) -> str | None:
        """在已有文件名里做不区分大小写的包含匹配。"""
        key = (keyword or "").strip().lower()
        if not key:
            return None
        metas = self.collection.get(include=["metadatas"]).get("metadatas") or []
        names = sorted({(m or {}).get("filename") for m in metas if (m or {}).get("filename")})
        for name in names:  # 优先「文件名包含关键词」
            if key in name.lower():
                return name
        for name in names:  # 再退一步「关键词包含文件名」
            if name.lower() in key:
                return name
        return None

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


_store: VectorStore | None = None
_lock = threading.Lock()


def get_store() -> VectorStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = VectorStore(str(settings.chroma_dir))
    return _store


def warm_up() -> float:
    """预热向量化模型，返回耗时秒数。

    必须在服务启动时调用。模型加载与 torch 的首次推理都是惰性的，
    如果留到第一个用户请求才做，那一次请求要多等十几秒（实测约 16s）——
    表现为「首字延迟很高」，但其实是冷启动，与检索、编排都无关。
    """
    started = time.perf_counter()
    store = get_store()
    store.embedder.embed_one("预热")  # 触发首次推理，与集合是否为空无关
    return time.perf_counter() - started