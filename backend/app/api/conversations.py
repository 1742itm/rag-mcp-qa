"""会话接口：列表、新建、查看消息、删除。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db.sqlite import get_db

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = "新对话"


@router.get("")
async def list_conversations() -> list[dict]:
    return get_db().list_conversations()


@router.post("")
async def create_conversation(payload: ConversationCreate) -> dict:
    return get_db().create_conversation(payload.title)


@router.get("/{conv_id}/messages")
async def list_messages(conv_id: str) -> list[dict]:
    if not get_db().get_conversation(conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return get_db().list_messages(conv_id)


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str) -> dict:
    if not get_db().delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "id": conv_id}