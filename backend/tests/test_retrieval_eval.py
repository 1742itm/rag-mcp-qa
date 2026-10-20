"""检索评测脚本的单测。

评测本身需要真实向量模型与耗时，不纳入默认 pytest；这里只测它的**纯计算部分**
（指标、阈值扫描、建议阈值）与评测集的完整性，全部离线、毫秒级。
"""
from __future__ import annotations

import pytest

from app.config import settings
from evals.retrieval_eval import (
    CORPUS,
    NEGATIVE,
    POSITIVE,
    SCAN_THRESHOLDS,
    recall_at_k,
    recommend_threshold,
    scan_thresholds,
)


class TestRecallAtK:
    def test_全部相关片段都被召回(self):
        assert recall_at_k([("a", 0), ("b", 0)], (("a", 0), ("b", 0))) == 1.0

    def test_部分召回(self):
        assert recall_at_k([("a", 0), ("c", 0)], (("a", 0), ("b", 0))) == 0.5

    def test_完全未召回(self):
        assert recall_at_k([("c", 0)], (("a", 0),)) == 0.0

    def test_没有标注时返回0(self):
        assert recall_at_k([("a", 0)], ()) == 0.0

    def test_重复命中不重复计数(self):
        assert recall_at_k([("a", 0), ("a", 0)], (("a", 0),)) == 1.0

    def test_超出召回列表不影响分母(self):
        assert recall_at_k([], (("a", 0), ("b", 0))) == 0.0


class TestScanThresholds:
    def test_每个阈值一行(self):
        rows = scan_thresholds([0.9], [0.1], [0.2, 0.5, 0.8])
        assert len(rows) == 3
        assert [r["threshold"] for r in rows] == [0.2, 0.5, 0.8]

    def test_正确标记完美分离(self):
        rows = scan_thresholds([0.9, 0.6], [0.1, 0.2], [0.3, 0.7, 0.95])
        assert rows[0]["separated"] is True, "0.3：正样本全保留且负样本无误收"
        assert rows[1]["separated"] is False, "0.7：漏掉 0.6 那个正样本"
        assert rows[2]["separated"] is False, "0.95：正样本一条都不剩"

    def test_阈值越高保留越少(self):
        rows = scan_thresholds([0.2, 0.4, 0.6, 0.8], [0.1, 0.3], [0.1, 0.3, 0.5, 0.7, 0.9])
        kept = [r["positive_kept"] for r in rows]
        assert kept == sorted(kept, reverse=True)

    def test_阈值越高误收越少(self):
        rows = scan_thresholds([0.9], [0.1, 0.3, 0.5], [0.1, 0.2, 0.4, 0.6])
        leaked = [r["negative_leaked"] for r in rows]
        assert leaked == sorted(leaked, reverse=True)

    def test_统计总数正确(self):
        rows = scan_thresholds([0.5, 0.6], [0.1, 0.2, 0.3], [0.4])
        assert rows[0]["positive_total"] == 2
        assert rows[0]["negative_total"] == 3


class TestRecommendThreshold:
    def test_可分离时取空隙中值(self):
        rec = recommend_threshold([0.6, 0.8], [0.2, 0.3])
        assert rec["separable"] is True
        assert rec["negative_max"] == pytest.approx(0.3)
        assert rec["positive_min"] == pytest.approx(0.6)
        assert rec["gap"] == pytest.approx(0.3)
        assert rec["recommended"] == pytest.approx(0.45)

    def test_分数重叠时判定不可分离(self):
        rec = recommend_threshold([0.5, 0.6], [0.2, 0.55])
        assert rec["separable"] is False
        assert "重叠" in rec["reason"]

    def test_恰好相接视为不可分离(self):
        rec = recommend_threshold([0.5], [0.5])
        assert rec["separable"] is False

    def test_样本不足时判定不可分离(self):
        assert recommend_threshold([], [0.1])["separable"] is False
        assert recommend_threshold([0.9], [])["separable"] is False

    def test_建议值必然落在空隙内(self):
        rec = recommend_threshold([0.9, 0.58], [0.1, 0.31])
        assert rec["negative_max"] < rec["recommended"] < rec["positive_min"]


class TestDataset:
    """评测集自身的完整性——标注写错会让评测结果失去意义。"""

    def test_正样本的标注都指向真实存在的片段(self):
        for item in POSITIVE:
            for doc, idx in item.relevant:
                assert doc in CORPUS, f"标注指向未知文档：{doc}"
                assert 0 <= idx < len(CORPUS[doc]), f"{doc} 的片段序号越界：{idx}"

    def test_查询不重复且非空(self):
        texts = [q.text for q in POSITIVE] + NEGATIVE
        assert all(t.strip() for t in texts)
        assert len(texts) == len(set(texts)), "存在重复的评测问题"

    def test_正负样本都不为空(self):
        assert POSITIVE, "缺少正样本"
        assert NEGATIVE, "缺少负样本"

    def test_语料不为空且每个文档都有片段(self):
        assert CORPUS
        for name, chunks in CORPUS.items():
            assert chunks, f"{name} 没有片段"

    def test_扫描范围覆盖当前配置的阈值(self):
        # 否则扫描表里根本看不到真实配置落在哪一档，标定就失去参照
        assert min(SCAN_THRESHOLDS) <= settings.min_score <= max(SCAN_THRESHOLDS)
        assert min(SCAN_THRESHOLDS) <= settings.min_score_hash <= max(SCAN_THRESHOLDS)
