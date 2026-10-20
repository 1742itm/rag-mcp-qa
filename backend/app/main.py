"""FastAPI 应用入口。

启动：uvicorn app.main:app --reload --port 8000  （在 backend/ 目录、已激活 venv）
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, conversations, documents, system
from .config import settings
from .db.sqlite import init_db
from .mcp import register_builtin_tools
from .rag.store import warm_up as warm_up_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings.db_path)
    register_builtin_tools()

    # 预热向量模型：模型加载与首次推理很慢（实测约 16s）。
    # 放在启动时做，避免第一个用户请求把这段时间算进「首字延迟」。
    logger.info("正在预热向量模型（首次运行需下载模型，可能较慢）…")
    duration = await asyncio.to_thread(warm_up_retrieval)
    logger.info("向量模型预热完成，耗时 %.1fs", duration)

    logger.info("%s 启动完成 -> http://%s:%d/docs", settings.app_name, settings.host, settings.port)
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(conversations.router)
app.include_router(chat.router)


@app.get("/")
async def root() -> dict:
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)