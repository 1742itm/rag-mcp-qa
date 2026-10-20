"""内置工具单测：用真实工具函数 + 可注入的临时依赖，完全离线、不加载模型。

工具是 MCP 编排的「能力层」，它们的返回约定（summary / references）被 Agent 直接消费，
所以这里重点验证约定字段与边界，而不只是「能不能跑」。
"""
from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.db.sqlite import Database
from app.mcp import tools as tools_module
from app.mcp.registry import ToolRegistry
from app.rag.embedder import HashEmbedder
from app.rag.store import VectorStore

DOC = "文档.txt"
CHUNKS = [
    "RAG 通过先检索后生成的方式降低大模型幻觉。",
    "短期记忆存放在内存中，长期记忆存放在 PostgreSQL 与 Milvus。",
    "代码质量使用 ruff 与 mypy strict 做静态检查。",
    "相似度阈值用于过滤噪声，低于阈值的结果直接丢弃。",
]


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def store(tmp_path) -> VectorStore:
    s = VectorStore(str(tmp_path / "chroma"), embedder=HashEmbedder(dim=512))
    s.add_document("doc1", DOC, CHUNKS)
    return s


@pytest.fixture
def wired(db, store, monkeypatch):
    """把工具里的 get_db / get_store 指向临时依赖，避免碰真实数据。"""
    monkeypatch.setattr(tools_module, "get_db", lambda: db)
    monkeypatch.setattr(tools_module, "get_store", lambda: store)
    return db, store


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
class TestListDocuments:
    def test_空知识库(self, wired):
        result = run(tools_module.list_documents())
        assert result["count"] == 0
        assert result["documents"] == []
        assert "没有任何文档" in result["summary"]

    def test_列出文档清单(self, wired):
        db, _ = wired
        db.add_document("a.md", "/tmp/a.md", 3)
        db.add_document("b.pdf", "/tmp/b.pdf", 7)

        result = run(tools_module.list_documents())
        assert result["count"] == 2
        assert {"filename", "chunks", "created_at"} <= set(result["documents"][0])
        assert "a.md" in result["summary"] and "b.pdf" in result["summary"]
        assert "3 个片段" in result["summary"]

    def test_不返回references字段(self, wired):
        # 非检索类工具不应带 references，否则 Agent 会把它当检索结果去做去重
        assert "references" not in run(tools_module.list_documents())


class TestGetDocumentContent:
    def test_按文件名精确读取(self, wired):
        result = run(tools_module.get_document_content(DOC))
        assert result["count"] == settings.top_k or result["count"] > 0
        assert result["filename"] == DOC
        assert "[片段 0]" in result["summary"]
        assert CHUNKS[0] in result["summary"]

    def test_片段按序号升序(self, wired):
        result = run(tools_module.get_document_content(DOC, limit=4))
        assert [c["chunk_index"] for c in result["chunks"]] == [0, 1, 2, 3]

    def test_可按区间读取(self, wired):
        result = run(tools_module.get_document_content(DOC, start=2, limit=1))
        assert result["count"] == 1
        assert result["chunks"][0]["chunk_index"] == 2
        assert CHUNKS[2] in result["summary"]

    def test_默认只取有限片段(self, wired):
        _, store = wired
        long_doc = "很长的文档.md"
        store.add_document("doc2", long_doc, [f"第 {i} 段内容" for i in range(50)])

        result = run(tools_module.get_document_content(long_doc))
        assert result["count"] == tools_module.DEFAULT_DOC_CHUNK_LIMIT

    def test_文件名可部分匹配(self, wired):
        # 用户说「文档」也应命中实际文件名「文档.txt」
        result = run(tools_module.get_document_content("文档"))
        assert result["filename"] == DOC
        assert result["count"] > 0

    def test_文件名不区分大小写(self, wired):
        _, store = wired
        store.add_document("doc3", "README.md", ["说明文档内容"])

        result = run(tools_module.get_document_content("readme"))
        assert result["filename"] == "README.md"

    def test_找不到文档时如实说明(self, wired):
        result = run(tools_module.get_document_content("不存在的文档.md"))
        assert result["count"] == 0
        assert "没有找到" in result["summary"]
        assert "不存在的文档.md" in result["summary"]

    def test_起点超出范围时如实说明(self, wired):
        result = run(tools_module.get_document_content(DOC, start=999))
        assert result["count"] == 0
        assert "没有更多内容" in result["summary"]

    def test_负数区间被规整(self, wired):
        result = run(tools_module.get_document_content(DOC, start=-5, limit=-1))
        # 负数起点规整为 0，非正 limit 回退到默认值；文档本身只有 4 段
        assert result["count"] == len(CHUNKS)
        assert result["chunks"][0]["chunk_index"] == 0
        assert "references" not in result


class TestKnowledgeSearch:
    def test_命中时返回片段与可引用结构(self, wired):
        result = run(tools_module.knowledge_search("大模型幻觉"))

        assert result["count"] > 0
        assert result["references"], "检索类工具必须返回 references 供引用展示"
        hit = result["references"][0]
        assert {"text", "filename", "doc_id", "chunk_index", "score"} <= set(hit)
        assert "检索到" in result["summary"]

    def test_全部低于阈值时如实说明且引用为空(self, wired, monkeypatch):
        monkeypatch.setattr(settings, "min_score", 0.999)
        monkeypatch.setattr(settings, "min_score_hash", 0.999)

        result = run(tools_module.knowledge_search("大模型幻觉"))
        assert result["count"] == 0
        assert result["references"] == []
        assert "未检索到相关内容" in result["summary"]

    def test_top_k参数生效(self, wired, monkeypatch):
        # 关掉阈值以单独验证 top_k——否则低于阈值的片段会被过滤掉，数量对不上
        monkeypatch.setattr(settings, "min_score_hash", -1.0)
        result = run(tools_module.knowledge_search("记忆", top_k=2))
        assert result["count"] == 2

    def test_top_k为0时用配置默认值(self, wired, monkeypatch):
        monkeypatch.setattr(settings, "min_score_hash", -1.0)
        result = run(tools_module.knowledge_search("记忆", top_k=0))
        assert result["count"] == min(settings.top_k, len(CHUNKS))

    def test_阈值会削减实际返回数量(self, wired, monkeypatch):
        # 相似度阈值与 top_k 是两个独立的闸门：这里验证阈值确实在削减结果
        monkeypatch.setattr(settings, "min_score_hash", -1.0)
        without_threshold = run(tools_module.knowledge_search("记忆", top_k=4))["count"]

        monkeypatch.setattr(settings, "min_score_hash", 0.999)
        with_threshold = run(tools_module.knowledge_search("记忆", top_k=4))["count"]

        assert with_threshold == 0 <= without_threshold

    def test_空知识库检索返回空(self, db, tmp_path, monkeypatch):
        empty = VectorStore(str(tmp_path / "empty"), embedder=HashEmbedder(dim=512))
        monkeypatch.setattr(tools_module, "get_db", lambda: db)
        monkeypatch.setattr(tools_module, "get_store", lambda: empty)

        result = run(tools_module.knowledge_search("任何问题"))
        assert result["count"] == 0
        assert "未检索到相关内容" in result["summary"]


class TestNeighborExpansion:
    """相邻上下文扩展：长答案被切成多段时，只召回命中段会让回答缺头少尾。

    这里用「首段独一无二、其余为填充」的长文档，保证命中段是确定的第 5 段。
    """

    KEY_CHUNK = "向量数据库用来存储文本的语义向量，这一段是独一无二的关键内容。"
    QUERY = "向量数据库存储文本的语义向量"

    @pytest.fixture
    def long_store(self, db, tmp_path, monkeypatch):
        chunks = [f"第{i}段：与主题无关的填充内容，仅用于占位。" for i in range(10)]
        chunks[5] = self.KEY_CHUNK
        store = VectorStore(str(tmp_path / "long"), embedder=HashEmbedder(dim=512))
        store.add_document("doc9", "长文档.md", chunks)

        monkeypatch.setattr(tools_module, "get_db", lambda: db)
        monkeypatch.setattr(tools_module, "get_store", lambda: store)
        monkeypatch.setattr(settings, "min_score_hash", -1.0)  # 关掉阈值，只验证窗口逻辑
        return store

    def test_命中片段会补齐相邻上下文(self, long_store):
        result = run(tools_module.knowledge_search(self.QUERY, top_k=1))

        assert result["count"] == 1, "count 仍应是命中片段数"
        assert [r["chunk_index"] for r in result["references"]] == [3, 4, 5, 6, 7]
        assert "补齐其相邻上下文" in result["summary"]

    def test_命中段保留相似度_扩展段不带分数(self, long_store):
        refs = run(tools_module.knowledge_search(self.QUERY, top_k=1))["references"]

        hit = next(r for r in refs if r["chunk_index"] == 5)
        assert hit["expanded"] is False and isinstance(hit["score"], float)
        assert self.KEY_CHUNK == hit["text"]

        for ref in refs:
            if ref["chunk_index"] != 5:
                assert ref["expanded"] is True
                assert ref["score"] is None, "扩展段不能编造相似度"

    def test_窗口大小取自配置(self, long_store, monkeypatch):
        monkeypatch.setattr(settings, "context_window", 1)
        refs = run(tools_module.knowledge_search(self.QUERY, top_k=1))["references"]
        assert [r["chunk_index"] for r in refs] == [4, 5, 6]

    def test_窗口为0时退化为原行为(self, long_store, monkeypatch):
        monkeypatch.setattr(settings, "context_window", 0)
        refs = run(tools_module.knowledge_search(self.QUERY, top_k=1))["references"]
        assert [r["chunk_index"] for r in refs] == [5]

    def test_总片段数受上限约束(self, long_store, monkeypatch):
        monkeypatch.setattr(settings, "max_context_chunks", 3)
        refs = run(tools_module.knowledge_search(self.QUERY, top_k=1))["references"]
        assert len(refs) == 3

    def test_多条命中的扩展区间不会重复(self, long_store, monkeypatch):
        monkeypatch.setattr(settings, "min_score_hash", -1.0)
        refs = run(tools_module.knowledge_search(self.QUERY, top_k=4))["references"]

        keys = [(r["doc_id"], r["chunk_index"]) for r in refs]
        assert len(keys) == len(set(keys)), "合并时必须按 (doc_id, chunk_index) 去重"
        assert len(refs) <= settings.max_context_chunks
        assert 5 in [r["chunk_index"] for r in refs]

    def test_上下文不跨文档(self, long_store):
        chunks = [f"第{i}段：与主题无关的填充内容，仅用于占位。" for i in range(6)]
        long_store.add_document("doc10", "另一份.md", chunks)

        refs = run(tools_module.knowledge_search(self.QUERY, top_k=1))["references"]
        assert {r["filename"] for r in refs} == {"长文档.md"}


# ----------------------------------------------------------------------
class TestRegistration:
    def test_三个内置工具都已注册(self):
        registry = ToolRegistry()
        for tool in tools_module._build_tools():
            registry.register(tool)

        names = {t.name for t in registry.list()}
        assert names == set(tools_module.BUILTIN_TOOL_NAMES)
        assert len(names) == 3

    def test_工具名与处理器一一对应(self):
        by_name = {t.name: t for t in tools_module._build_tools()}
        assert by_name["knowledge_search"].handler is tools_module.knowledge_search
        assert by_name["list_documents"].handler is tools_module.list_documents
        assert by_name["get_document_content"].handler is tools_module.get_document_content

    def test_无参数工具声明为空(self):
        by_name = {t.name: t for t in tools_module._build_tools()}
        assert by_name["list_documents"].parameters == []
        assert by_name["list_documents"].schema()["parameters"]["required"] == []

    def test_每个工具都生成了可用的JSONSchema(self):
        for tool in tools_module._build_tools():
            schema = tool.schema()
            assert schema["name"] and schema["description"]
            assert schema["parameters"]["type"] == "object"

    def test_工具描述能提示模型如何区分(self):
        prompt = "\n".join(t.to_prompt() for t in tools_module._build_tools())
        # 描述必须让模型能分辨「查清单」「读正文」「语义检索」三种意图
        assert "清单" in prompt
        assert "knowledge_search" in prompt and "get_document_content" in prompt
        assert "优先使用" in prompt
