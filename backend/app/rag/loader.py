"""文档加载与切片：负责把 PDF / TXT / MD 变成可向量化的文本块。"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# 针对中文优化的切分优先级
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_text(path: Path) -> str:
    """读取文件纯文本。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md"}:
        return _load_plain(path)
    raise ValueError(f"不支持的文件类型: {suffix}")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _load_plain(path: Path) -> str:
    for encoding in ("utf-8", "gbk", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """切分为带重叠的文本块，避免语义被硬切断。"""
    text = (text or "").strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
    )
    chunks = [c.strip() for c in splitter.split_text(text)]
    return [c for c in chunks if c]