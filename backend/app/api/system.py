"""系统信息接口：健康检查与运行状态。"""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..db.sqlite import get_db
from ..mcp.registry import registry
from ..rag.store import get_store

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health() -> dict:
    """返回服务状态，便于前端展示与排查配置问题。"""
    store = get_store()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "llm": {
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "configured": bool(settings.llm_api_key)
            or "localhost" in settings.llm_base_url
            or "127.0.0.1" in settings.llm_base_url,
        },
        "embedding": {
            "provider": store.embedder.name,
            "configured_provider": settings.embedding_provider,
            "model": getattr(store.embedder, "model_name", None),
            "dim": store.embedder.dim,
        },
        "tools": [t.name for t in registry.list()],
        "documents": len(get_db().list_documents()),
        "chunks": store.count(),
    }