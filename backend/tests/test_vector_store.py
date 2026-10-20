"""向量库相似度阈值单测。

用确定性的哈希向量 + 临时 Chroma 目录，完全离线、结果可复现，
因此可以对「阈值过滤」这类边界做精确断言。
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.rag.embedder import HashEmbedder
from app.rag.store import VectorStore

CHUNKS = [
    "RAG 通过先检索后生成的方式降低大模型幻觉。",
    "短期记忆存放在内存中，长期记忆存放在 PostgreSQL 与 Milvus。",
    "代码质量使用 ruff 与 mypy strict 做静态检查。",
]


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(str(tmp_path / "chroma"), embedder=HashEmbedder(dim=512))


class StubSemanticEmbedder(HashEmbedder):
    """只用于测试：借用哈希向量实现，但伪装成语义模型，验证默认阈值按向量化器切换。"""

    name = "sentence-transformers"


@pytest.fixture
def loaded(store) -> VectorStore:
    store.add_document("doc1", "文档.txt", CHUNKS)
    return store


def all_scores(store: VectorStore, query: str, top_k: int = 3) -> list[float]:
    """取未过滤时的原始分数（min_score 传负数即不过滤）。"""
    return [hit["score"] for hit in store.search(query, top_k=top_k, min_score=-1.0)]


class TestThreshold:
    def test_关闭阈值时返回全部候选(self, loaded):
        hits = loaded.search("大模型幻觉", top_k=3, min_score=-1.0)
        assert len(hits) == 3
        assert all("score" in h for h in hits)

    def test_结果按相似度降序(self, loaded):
        scores = all_scores(loaded, "大模型幻觉")
        assert scores == sorted(scores, reverse=True)

    def test_阈值高于最高分时结果为空(self, loaded):
        assert loaded.search("大模型幻觉", top_k=3, min_score=0.999) == []

    def test_被过滤片段不混入结果(self, loaded):
        for threshold in (0.0, 0.05, 0.1, 0.2, 0.3):
            expected = [s for s in all_scores(loaded, "大模型幻觉") if s >= threshold]
            kept = loaded.search("大模型幻觉", top_k=3, min_score=threshold)
            assert [h["score"] for h in kept] == expected, f"阈值 {threshold} 过滤结果不符"

    def test_阈值抬高一条即可排除最高分(self, loaded):
        best = all_scores(loaded, "大模型幻觉")[0]
        kept = loaded.search("大模型幻觉", top_k=3, min_score=best)
        excluded = loaded.search("大模型幻觉", top_k=3, min_score=round(best + 0.0001, 4))
        assert all(h["score"] >= best for h in kept)
        assert all(h["score"] > best for h in excluded)
        assert len(excluded) < len(kept), "略高于最高分的阈值应至少排除一条"

    def test_默认阈值取自配置_哈希模式(self, loaded, monkeypatch):
        best = all_scores(loaded, "大模型幻觉")[0]
        monkeypatch.setattr(settings, "min_score_hash", round(best + 0.0001, 4))
        assert loaded.search("大模型幻觉", top_k=3) == [], "未显式传参时应使用配置默认阈值"

    def test_显式传参覆盖配置默认值(self, loaded, monkeypatch):
        monkeypatch.setattr(settings, "min_score_hash", 0.999)
        monkeypatch.setattr(settings, "min_score", 0.999)
        assert loaded.search("大模型幻觉", top_k=3, min_score=-1.0), "显式传参应覆盖配置"

    def test_默认阈值随向量化器切换(self, tmp_path, monkeypatch):
        # 两个向量化器的分数尺度不同，必须各用各的阈值，否则降级模式会被全部过滤
        monkeypatch.setattr(settings, "min_score", 0.99)
        monkeypatch.setattr(settings, "min_score_hash", -1.0)

        semantic = VectorStore(str(tmp_path / "sem"), embedder=StubSemanticEmbedder(dim=512))
        semantic.add_document("d1", "文档.txt", CHUNKS)
        hashed = VectorStore(str(tmp_path / "hash"), embedder=HashEmbedder(dim=512))
        hashed.add_document("d1", "文档.txt", CHUNKS)

        assert semantic.search("大模型幻觉", top_k=3) == [], "语义向量化器应使用 min_score"
        assert len(hashed.search("大模型幻觉", top_k=3)) == 3, "哈希向量化器应使用 min_score_hash"

    def test_哈希模式默认阈值不会让降级模式失效(self, loaded):
        # 降级模式的可用性保障：默认阈值下，与文本有字符重叠的查询仍应拿到结果
        assert loaded.search("大模型幻觉", top_k=3), "降级模式不应因阈值而完全失效"

    def test_top_k仍然生效(self, loaded):
        assert len(loaded.search("记忆", top_k=2, min_score=-1.0)) == 2

    def test_空知识库检索返回空(self, store):
        assert store.search("任何问题", min_score=-1.0) == []


class TestMetadata:
    """去重依赖 (doc_id, chunk_index)，阈值过滤不能破坏这些字段。"""

    def test_命中片段带完整元数据(self, loaded):
        hits = loaded.search("记忆", top_k=3, min_score=-1.0)
        assert hits
        for hit in hits:
            assert hit["doc_id"] == "doc1"
            assert hit["filename"] == "文档.txt"
            assert isinstance(hit["chunk_index"], int)
            assert isinstance(hit["text"], str) and hit["text"]

    def test_删除文档后不再被检索到(self, loaded):
        loaded.delete_document("doc1")
        assert loaded.search("记忆", top_k=3, min_score=-1.0) == []


class TestChunkWindow:
    """相邻上下文读取：长答案补齐的基础能力，必须严格限定在同一文档内。"""

    @pytest.fixture
    def many(self, store) -> VectorStore:
        store.add_document("doc1", "文档.txt", [f"片段{i}内容" for i in range(10)])
        store.add_document("doc2", "其他.txt", [f"别的{i}内容" for i in range(10)])
        return store

    def test_含自身且按序号升序(self, many):
        rows = many.chunk_window("文档.txt", 5, before=2, after=2)
        assert [r["chunk_index"] for r in rows] == [3, 4, 5, 6, 7]

    def test_前后窗口可分别控制(self, many):
        assert [r["chunk_index"] for r in many.chunk_window("文档.txt", 5, 0, 3)] == [5, 6, 7, 8]
        assert [r["chunk_index"] for r in many.chunk_window("文档.txt", 5, 3, 0)] == [2, 3, 4, 5]

    def test_窗口为0时只返回自身(self, many):
        assert [r["chunk_index"] for r in many.chunk_window("文档.txt", 4)] == [4]

    def test_起点边界不会越界(self, many):
        assert [r["chunk_index"] for r in many.chunk_window("文档.txt", 0, 3, 1)] == [0, 1]

    def test_终点边界不会越界(self, many):
        assert [r["chunk_index"] for r in many.chunk_window("文档.txt", 9, 1, 3)] == [8, 9]

    def test_不串到其他文档(self, many):
        rows = many.chunk_window("文档.txt", 5, 10, 10)
        assert len(rows) == 10, "只应取到本文档的片段"
        assert {r["filename"] for r in rows} == {"文档.txt"}

    def test_返回完整元数据(self, many):
        row = many.chunk_window("文档.txt", 2)[0]
        assert row["doc_id"] == "doc1"
        assert row["filename"] == "文档.txt"
        assert row["text"] == "片段2内容"

    def test_空知识库返回空(self, store):
        assert store.chunk_window("文档.txt", 0, 2, 2) == []
