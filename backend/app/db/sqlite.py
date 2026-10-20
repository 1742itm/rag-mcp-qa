"""SQLite 存储层：文档元数据、会话与消息历史。

单文件、零中间件，配合 Chroma 目录一起构成「轻量化部署」。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id          TEXT PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    chunks      INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    "references"    TEXT,
                    created_at      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id);
                """
            )
            self._conn.commit()

    # ---------------------- 文档 ----------------------
    def add_document(
        self, filename: str, path: str, chunks: int, doc_id: str | None = None
    ) -> dict[str, Any]:
        doc_id = doc_id or uuid.uuid4().hex
        created = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO documents (id, filename, path, chunks, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, path, chunks, created),
            )
            self._conn.commit()
        return {
            "id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "created_at": created,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, filename, chunks, created_at FROM documents "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, filename, path, chunks, created_at FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ---------------------- 会话 ----------------------
    def create_conversation(self, title: str = "新对话") -> dict[str, Any]:
        conv_id = uuid.uuid4().hex
        created = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conv_id, title, created, created),
            )
            self._conn.commit()
        return {"id": conv_id, "title": title, "created_at": created}

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, created_at FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def ensure_conversation(self, conv_id: str | None, first_message: str) -> dict[str, Any]:
        """会话不存在则创建，标题取首条消息前 20 字。"""
        if conv_id:
            conv = self.get_conversation(conv_id)
            if conv:
                return conv
        title = (first_message or "新对话").strip().replace("\n", " ")[:20] or "新对话"
        return self.create_conversation(title)

    def rename_conversation(self, conv_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conv_id),
            )
            self._conn.commit()

    def delete_conversation(self, conv_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            self._conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def _touch(self, conv_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conv_id)
            )
            self._conn.commit()

    # ---------------------- 消息 ----------------------
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        references: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        created = _now()
        payload = json.dumps(references, ensure_ascii=False) if references else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, \"references\", created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, payload, created),
            )
            self._conn.commit()
            msg_id = cur.lastrowid
        self._touch(conversation_id)
        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "references": references or [],
            "created_at": created,
        }

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, \"references\", created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["references"] = json.loads(item["references"]) if item["references"] else []
            messages.append(item)
        return messages

    def history_pairs(self, conversation_id: str, limit: int = 20) -> list[dict[str, str]]:
        """转成 LLM 需要的 {role, content} 序列。"""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.list_messages(conversation_id)[-limit:]
        ]

    def close(self) -> None:
        self._conn.close()


_db: Database | None = None


def init_db(db_path: Path) -> Database:
    global _db
    _db = Database(db_path)
    logger.info("SQLite 就绪: %s", db_path)
    return _db


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("数据库尚未初始化，请先调用 init_db()")
    return _db