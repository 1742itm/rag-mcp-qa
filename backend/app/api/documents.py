"""文档接口：上传解析入库、列表查询、删除。"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import settings
from ..db.sqlite import get_db
from ..rag.loader import SUPPORTED_SUFFIXES, load_text, split_text
from ..rag.store import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """上传文档 → 解析 → 切片 → 向量化入库。"""
    filename = file.filename or "unnamed"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix}，仅支持 PDF / TXT / MD",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件超过 20MB 限制")

    doc_id = uuid.uuid4().hex
    dest = settings.upload_dir / f"{doc_id}{suffix}"
    dest.write_bytes(content)

    try:
        text = await asyncio.to_thread(load_text, dest)
    except Exception as exc:
        logger.exception("解析文档失败")
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"文档解析失败：{exc}") from exc

    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="文档解析后没有可用文本（可能是扫描版 PDF）")

    try:
        store = await asyncio.to_thread(get_store)
        count = await asyncio.to_thread(store.add_document, doc_id, filename, chunks)
    except Exception as exc:
        logger.exception("向量化入库失败")
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"向量化失败：{exc}") from exc

    doc = get_db().add_document(filename, str(dest), count, doc_id)
    logger.info("文档入库完成: %s (%d 切片)", filename, count)
    return doc


@router.get("")
async def list_documents() -> list[dict]:
    return get_db().list_documents()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    db = get_db()
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    store = await asyncio.to_thread(get_store)
    await asyncio.to_thread(store.delete_document, doc_id)
    db.delete_document(doc_id)

    path = Path(doc["path"])
    if path.exists():
        path.unlink(missing_ok=True)
    return {"ok": True, "id": doc_id}