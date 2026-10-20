"""检索评测：标定相似度阈值 + 衡量检索质量。

为什么需要它
------------
「相关性阈值」这类参数不能凭直觉设。本项目最初按直觉取 0.35，
实测发现 8 个无关问题里仍有 2 个漏过；标定后才发现 0.45 才是分界点。

这个脚本把标定过程固化下来：
  1. 用内置语料（自带标注，不依赖你上传的文档）建立隔离的向量库；
  2. 分别测「正样本（知识库确有答案）」与「负样本（与知识库无关）」的分数分布；
  3. 扫描候选阈值，给出建议值，便于复核或随语料/模型变化重新标定。

运行
----
    cd backend
    python -m evals.retrieval_eval                # 用配置的向量模型
    python -m evals.retrieval_eval --provider hash # 离线降级模式，秒级完成
    python -m evals.retrieval_eval --json          # 输出机器可读结果

换向量模型或换语料后，请重跑本脚本重新标定阈值——它跟模型、语料、切分方式都有关。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.rag.embedder import get_embedder  # noqa: E402
from app.rag.store import VectorStore  # noqa: E402

# ----------------------------------------------------------------------
# 内置评测语料：自带答案标注，与用户上传的文档完全隔离
# ----------------------------------------------------------------------
Corpus = dict[str, list[str]]

CORPUS: Corpus = {
    "rag.md": [
        "RAG 检索增强生成：先从知识库检索相关片段，再把片段注入上下文交给模型生成，从而降低幻觉。",
        "向量检索基于语义相似度：把文本转成向量，语义越相近的两段文本，向量距离越近。",
        "相似度阈值用于过滤噪声：相似度低于阈值的结果视为与问题无关，直接丢弃而不注入上下文。",
    ],
    "memory.md": [
        "短期记忆存放在内存中，保存当前会话的对话历史与工具调用结果，用滑动窗口管理。",
        "长期记忆存放在 PostgreSQL 与向量库中，保存用户画像、历史摘要与高频问答对。",
        "会话结束时提取关键信息并更新长期记忆，完成短期记忆到长期记忆的转化。",
    ],
    "quality.md": [
        "代码质量依赖 ruff 统一代码风格，并使用 mypy strict 做静态类型检查。",
        "安全扫描使用 bandit 工具，同时把单个函数的圈复杂度限制在 10 以内。",
        "合并请求至少需要一人评审，核心模块则要求两人评审后才能合入。",
    ],
    "react.md": [
        "ReAct 模式让模型逐步推理、边做边看，可以中途调整策略。",
        "Plan-and-Execute 模式先制定完整计划再执行，计划一旦失败需要整体重做。",
        "ReAct 适合意图多变、需要交互的场景，提前规划则适合步骤明确、可预见的任务。",
    ],
}


@dataclass(frozen=True)
class PositiveQuery:
    """正样本：问题应当命中指定的片段。"""

    text: str
    relevant: tuple[tuple[str, int], ...]  # (文件名, 片段序号)


POSITIVE: list[PositiveQuery] = [
    PositiveQuery("RAG 是怎么降低大模型幻觉的？", (("rag.md", 0),)),
    PositiveQuery("相似度阈值有什么作用？", (("rag.md", 2),)),
    PositiveQuery("短期记忆存放在哪里？", (("memory.md", 0),)),
    PositiveQuery("怎么把短期记忆转成长期记忆？", (("memory.md", 2),)),
    PositiveQuery("项目用什么工具做静态检查？", (("quality.md", 0),)),
    PositiveQuery("代码复杂度有什么限制？", (("quality.md", 1),)),
    PositiveQuery(
        "ReAct 和 Plan-and-Execute 有什么区别？",
        (("react.md", 0), ("react.md", 1), ("react.md", 2)),
    ),
]

# 负样本：与语料完全无关，理想情况下一条都不应命中
NEGATIVE: list[str] = [
    "红烧肉怎么做才好吃？",
    "北京今天的天气怎么样？",
    "怎么治疗感冒？",
    "珠穆朗玛峰有多高？",
    "如何在家种植番茄？",
    "2024 年奥运会金牌榜第一是哪个国家？",
]

# 阈值扫描的候选值
SCAN_THRESHOLDS = [round(0.05 * i, 2) for i in range(3, 16)]  # 0.15 ~ 0.75


# ----------------------------------------------------------------------
# 纯计算：不碰向量库，便于单测
# ----------------------------------------------------------------------
def recall_at_k(retrieved: list[tuple[str, int]], relevant: tuple[tuple[str, int], ...]) -> float:
    """命中的相关片段数 / 全部相关片段数。"""
    if not relevant:
        return 0.0
    hit = len(set(retrieved) & set(relevant))
    return hit / len(relevant)


def scan_thresholds(
    pos_top1: list[float],
    neg_top1: list[float],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """扫描候选阈值，统计「正样本保留」与「负样本误收」。"""
    thresholds = thresholds or SCAN_THRESHOLDS
    rows = []
    for t in thresholds:
        kept = sum(1 for s in pos_top1 if s >= t)
        leaked = sum(1 for s in neg_top1 if s >= t)
        rows.append(
            {
                "threshold": t,
                "positive_kept": kept,
                "positive_total": len(pos_top1),
                "negative_leaked": leaked,
                "negative_total": len(neg_top1),
                "separated": kept == len(pos_top1) and leaked == 0,
            }
        )
    return rows


def recommend_threshold(pos_top1: list[float], neg_top1: list[float]) -> dict:
    """在「最高负样本分」与「最低正样本分」的空隙内取中值。

    两类分数有重叠时无法用单一阈值分离，返回 separable=False。
    """
    if not pos_top1 or not neg_top1:
        return {"separable": False, "reason": "样本不足"}
    lo, hi = max(neg_top1), min(pos_top1)
    if lo >= hi:
        return {
            "separable": False,
            "reason": "正负样本分数有重叠",
            "negative_max": round(lo, 4),
            "positive_min": round(hi, 4),
        }
    return {
        "separable": True,
        "negative_max": round(lo, 4),
        "positive_min": round(hi, 4),
        "gap": round(hi - lo, 4),
        "recommended": round((lo + hi) / 2, 3),
    }


# ----------------------------------------------------------------------
# 评测主流程
# ----------------------------------------------------------------------
def _build_store(provider: str, persist_dir: Path) -> VectorStore:
    embedder = get_embedder(provider, settings.embedding_model, settings.embedding_dim)
    store = VectorStore(str(persist_dir), embedder=embedder)
    store.reset()  # 每次评测都从干净状态开始，避免旧数据干扰
    for name, chunks in CORPUS.items():
        store.add_document(name, name, chunks)
    return store


def evaluate(provider: str | None = None, k: int | None = None) -> dict:
    provider = provider or settings.embedding_provider
    k = k or settings.top_k

    # 评测库放在 data/ 下（已被 .gitignore 忽略），与线上知识库 chroma/ 相互隔离
    store = _build_store(provider, settings.data_dir / "eval_chroma")

    positive_rows = []
    pos_top1: list[float] = []
    pos_recall: list[float] = []
    pos_hit1 = 0

    for item in POSITIVE:
        hits = store.search(item.text, top_k=k, min_score=-1.0)
        retrieved = [(h["filename"], h["chunk_index"]) for h in hits]
        scores = [h["score"] for h in hits]
        recall = recall_at_k(retrieved, item.relevant)
        hit1 = bool(set(retrieved[:1]) & set(item.relevant))

        pos_top1.append(scores[0] if scores else 0.0)
        pos_recall.append(recall)
        pos_hit1 += int(hit1)
        positive_rows.append(
            {
                "query": item.text,
                "top1": scores[0] if scores else None,
                "scores": scores,
                "recall_at_k": round(recall, 4),
                "hit_at_1": hit1,
            }
        )

    negative_rows = []
    neg_top1: list[float] = []
    for text in NEGATIVE:
        hits = store.search(text, top_k=k, min_score=-1.0)
        scores = [h["score"] for h in hits]
        neg_top1.append(scores[0] if scores else 0.0)
        negative_rows.append(
            {"query": text, "top1": scores[0] if scores else None, "scores": scores}
        )

    # 按当前配置阈值实际跑一遍，看负样本是否仍会误收
    leaked_now = sum(1 for text in NEGATIVE if store.search(text, top_k=k))

    return {
        "provider": store.embedder.name,
        "dim": store.embedder.dim,
        "chunks": store.count(),
        "k": k,
        "configured_min_score": settings.min_score,
        "configured_min_score_hash": settings.min_score_hash,
        "positive": positive_rows,
        "negative": negative_rows,
        "summary": {
            "recall_at_k": round(statistics.mean(pos_recall), 4),
            "hit_at_1": f"{pos_hit1}/{len(POSITIVE)}",
            "positive_top1_min": round(min(pos_top1), 4),
            "positive_top1_median": round(statistics.median(pos_top1), 4),
            "negative_top1_max": round(max(neg_top1), 4),
            "negative_top1_median": round(statistics.median(neg_top1), 4),
            "negative_leaked_with_current_config": leaked_now,
        },
        "scan": scan_thresholds(pos_top1, neg_top1),
        "recommendation": recommend_threshold(pos_top1, neg_top1),
    }


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------
def print_report(result: dict) -> None:
    print("=" * 92)
    print(f"向量化器={result['provider']} 维度={result['dim']} 语料片段={result['chunks']} top_k={result['k']}")
    print("=" * 92)

    print(f"\n【正样本】应命中的问题（共 {len(result['positive'])} 条）")
    print(f"{'Top1':>8}{'Recall@k':>11}{'Hit@1':>8}  {'全部分数':<38} 问题")
    print("-" * 92)
    for row in result["positive"]:
        scores = str([round(s, 4) for s in row["scores"]])
        print(
            f"{row['top1']:>8.4f}{row['recall_at_k']:>11.4f}"
            f"{'  是' if row['hit_at_1'] else '  否':>8}  {scores:<38} {row['query']}"
        )

    print(f"\n【负样本】与语料无关的问题（共 {len(result['negative'])} 条）")
    print(f"{'Top1':>8}  {'全部分数':<38} 问题")
    print("-" * 92)
    for row in result["negative"]:
        scores = str([round(s, 4) for s in row["scores"]])
        print(f"{row['top1']:>8.4f}  {scores:<38} {row['query']}")

    s = result["summary"]
    print("\n【汇总】")
    print(f"  Recall@{result['k']} 平均：{s['recall_at_k']}")
    print(f"  Hit@1：{s['hit_at_1']}")
    print(f"  正样本 Top1：min={s['positive_top1_min']}  中位={s['positive_top1_median']}")
    print(f"  负样本 Top1：max={s['negative_top1_max']}  中位={s['negative_top1_median']}")
    print(
        f"  当前配置阈值下负样本误收：{s['negative_leaked_with_current_config']}"
        f"/{len(result['negative'])}"
    )

    print("\n【阈值扫描】")
    print(f"{'阈值':>6}{'正样本保留':>12}{'负样本误收':>12}   说明")
    print("-" * 60)
    for row in result["scan"]:
        mark = "  <-- 完美分离" if row["separated"] else ""
        print(
            f"{row['threshold']:>6.2f}"
            f"{row['positive_kept']:>8}/{row['positive_total']:<3}"
            f"{row['negative_leaked']:>8}/{row['negative_total']:<3}   {mark}"
        )

    rec = result["recommendation"]
    print("\n【建议】")
    if rec.get("separable"):
        print(
            f"  负样本最高 {rec['negative_max']} < 正样本最低 {rec['positive_min']}，"
            f"空隙 {rec['gap']}"
        )
        print(f"  建议 MIN_SCORE = {rec['recommended']}")
        current = result["configured_min_score"]
        flag = "一致" if abs(current - rec["recommended"]) < 0.035 else "不一致，建议复核"
        print(f"  当前配置 MIN_SCORE = {current}（{flag}）")
    else:
        print(f"  正负样本分数有重叠（{rec.get('reason')}），无法用单一阈值分离。")
        print("  可考虑：加入重排模型、改用混合检索，或补充更多负样本后再评估。")


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评测与相似度阈值标定")
    parser.add_argument(
        "--provider",
        choices=["sentence-transformers", "hash"],
        default=None,
        help="向量化器；默认取 .env 配置。用 hash 可离线秒级跑通",
    )
    parser.add_argument("--k", type=int, default=None, help="检索条数，默认取 TOP_K")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON 结果")
    args = parser.parse_args()

    result = evaluate(args.provider, args.k)
    print_report(result)

    if args.json:
        print("\n【JSON】")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
