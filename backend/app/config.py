"""全局配置：从 backend/.env 读取，全部带默认值，开箱即用。"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 目录
BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 服务 ----
    app_name: str = "AI Agent 文档问答系统"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- LLM（兼容 OpenAI 接口）----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_timeout: int = 120

    # ---- 向量化 ----
    # sentence-transformers: 使用本地模型（需联网首次下载）
    # hash: 内置确定性哈希向量（零依赖、离线可用）
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # 仅 hash 降级模式使用
    # HuggingFace 镜像地址，国内下载模型更稳定；置空则使用官方源
    hf_endpoint: str = "https://hf-mirror.com"

    # ---- RAG 参数 ----
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    # 检索相似度下限：低于该值的片段视为与问题无关，直接丢弃而不注入上下文。
    # 0.45 由实测标定：本知识库上「相关问题」Top1 最低 0.547、「无关问题」Top1 最高 0.402，
    # 取两者之间可完美分离；凭直觉设的 0.35 会有 2/8 的无关问题漏过。
    min_score: float = 0.45
    # 哈希降级模式（EMBEDDING_PROVIDER=hash）专用阈值：哈希向量衡量的是字符重叠度而非语义，
    # 分数整体低约 0.25（相关对实测 0.31~0.45，无关对 -0.04~0.05），
    # 沿用 0.45 会把所有结果都过滤掉、使降级模式彻底失效。
    min_score_hash: float = 0.15
    # 命中片段向前后各扩展的相邻片段数：一份长答案常被切成多段，
    # 只召回命中段会让模型「答一半」（实测第 41 题 5 段只召回 3 段，缺两节内容）。
    context_window: int = 2
    # 单次检索注入上下文的总片段上限（含扩展段），避免长文档把上下文灌满
    max_context_chunks: int = 12

    # ---- 存储路径 ----
    data_dir: Path = BASE_DIR / "data"
    upload_dir: Path = BASE_DIR / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

# 必须在任何 huggingface_hub / sentence-transformers 导入之前生效：
# huggingface_hub 在模块导入时就读取这两个变量，晚设置会退回官方源（国内不可达）
if settings.hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
# 模型缓存也放进项目目录，避免污染本机其他位置（外部已设置 HF_HOME 时以其为准）
os.environ.setdefault("HF_HOME", str(settings.data_dir / "hf_cache"))

# 确保目录存在
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)