"""向量化：优先使用 sentence-transformers，缺失时自动降级为内置哈希向量。

降级设计的意义：保证「拿到代码就能跑」——
即使所在环境无法联网下载 HuggingFace 模型，检索链路依然完整可用。
"""
from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np

# 顶层导入 config：确保 HF_ENDPOINT 在 sentence-transformers 被导入前已写入环境变量
from ..config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """向量化接口。"""

    name: str = "base"
    dim: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(BaseEmbedder):
    """零依赖降级方案：字符 n-gram 哈希 + L2 归一化。

    不是语义模型，但对「关键词高度重合」的检索场景仍有可用精度，
    且完全离线、确定性输出，适合作为兜底。
    """

    name = "hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _grams(self, text: str) -> list[str]:
        text = text.lower().strip()
        grams = list(text)
        grams += [text[i : i + 2] for i in range(len(text) - 1)]
        return grams

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for gram in self._grams(text):
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的语义向量（首次使用会自动下载模型）。"""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，加快启动

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # 兼容不同版本的 sentence-transformers
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32).tolist()


_lock = threading.Lock()
_cached: BaseEmbedder | None = None


def get_embedder(provider: str, model_name: str, hash_dim: int = 512) -> BaseEmbedder:
    """按配置返回向量化器，单例缓存，失败自动降级。"""
    global _cached
    if _cached is not None:
        return _cached

    with _lock:
        if _cached is not None:
            return _cached

        if provider == "sentence-transformers":
            try:
                embedder = SentenceTransformerEmbedder(model_name)
                logger.info("已加载向量模型 %s (dim=%d)", model_name, embedder.dim)
            except Exception as exc:
                logger.warning(
                    "加载 sentence-transformers 失败（%s），降级为内置哈希向量。"
                    "如需语义检索请安装并联网下载模型。",
                    exc,
                )
                embedder = HashEmbedder(hash_dim)
        else:
            embedder = HashEmbedder(hash_dim)

        _cached = embedder
        return _cached