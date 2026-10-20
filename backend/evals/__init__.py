"""检索评测包：用于标定相似度阈值、衡量检索质量。

与 `tests/` 下的单测不同，评测需要真实向量模型、耗时更长、结果依赖语料，
因此不纳入默认 `pytest` 运行，需显式执行：

    cd backend
    python -m evals.retrieval_eval
"""
