import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from rag_parser.api.schemas import ChunkingType
from rag_parser.chunkers.base import BaseChunker
from rag_parser.chunkers.hybrid import HybridRagChunker
from rag_parser.core.config import settings
from rag_parser.core.interfaces import (
    BaseParser,
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from rag_parser.parsers.csv_parser import CsvParser
from rag_parser.parsers.markitdown_parser import (
    SUPPORTED_EXTENSIONS as MARKITDOWN_EXTENSIONS,
    MarkItDownParser,
)
from rag_parser.parsers.xlsx_parser import XlsxParser

logger = logging.getLogger(__name__)

PARSERS: Dict[str, type] = {".csv": CsvParser, ".tsv": CsvParser}
PARSERS.update({extension: XlsxParser for extension in (".xlsx", ".xls")})
PARSERS.update({extension: MarkItDownParser for extension in MARKITDOWN_EXTENSIONS})

TEXT_STRATEGIES: Dict[ChunkingType, str] = {
    ChunkingType.FIXED: "token",
    ChunkingType.SENTENCE: "sentence",
    ChunkingType.SEMANTIC: "semantic",
}

MIME_ALLOWLIST = {
    "text/plain",
    "text/csv",
    "text/markdown",
    "text/html",
    "text/x-csv",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
    "application/octet-stream",
    "inode/x-empty",
}


def supported_extensions() -> list:
    return sorted(PARSERS.keys())


def get_parser(filename: str) -> BaseParser:
    extension = Path(filename).suffix.lower()
    parser_class = PARSERS.get(extension)
    if parser_class is None:
        raise UnsupportedFormatError(extension, settings.supported_extensions_set)
    return parser_class()


def get_chunker(
    chunking_type: ChunkingType,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    source: str = "document",
    threshold: Optional[float] = None,
    table_chunk_size: Optional[int] = None,
) -> Optional[BaseChunker]:
    if chunking_type == ChunkingType.NONE:
        return None
    strategy = TEXT_STRATEGIES.get(ChunkingType(chunking_type))
    if strategy is None:
        raise ValueError(f"Unknown chunking type: {chunking_type}")
    if overlap is not None and chunk_size is not None and overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    return HybridRagChunker(
        text_strategy=strategy,  # type: ignore[arg-type]
        chunk_size=chunk_size,
        overlap=overlap,
        source=source,
        table_chunk_size=table_chunk_size,
        threshold=threshold,
    )


def sniff_mime(data: bytes) -> Optional[str]:
    if not data:
        return None
    try:
        import magic
    except ImportError:
        return None
    try:
        return magic.from_buffer(data, mime=True)
    except Exception:  # noqa: BLE001 - libmagic may be unavailable
        return None


def validate_upload(filename: str, data: bytes) -> None:
    if not filename or not filename.strip():
        raise DocumentParseError("Filename is required")
    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeError(len(data), settings.max_file_size_bytes)
    if not data:
        raise DocumentParseError("Uploaded file is empty")
    extension = Path(filename).suffix.lower()
    if extension not in PARSERS:
        raise UnsupportedFormatError(extension, settings.supported_extensions_set)
    mime = sniff_mime(data[:4096])
    if mime and mime not in MIME_ALLOWLIST:
        logger.warning("Unexpected mime type %s for %s", mime, filename)
        if extension in (".pdf", ".docx", ".xlsx", ".xls", ".xlsm"):
            raise DocumentParseError(
                f"File content does not match extension {extension} (detected {mime})",
                filename=filename,
            )


def save_upload(filename: str, data: bytes) -> Path:
    temp_dir = settings.ensure_temp_dir()
    safe_name = Path(filename).name or "upload"
    target = temp_dir / f"{uuid.uuid4().hex}_{safe_name}"
    target.write_bytes(data)
    return target


class DocumentStore:
    """In-memory store of parsed results (single user MVP, no database)."""

    def __init__(self, max_items: Optional[int] = None) -> None:
        self._items: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_items = max_items or settings.DOC_STORE_MAX_ITEMS

    def add(self, document: Dict[str, Any]) -> str:
        document_id = document.get("document_id") or uuid.uuid4().hex
        document = {**document, "document_id": document_id}
        with self._lock:
            self._items[document_id] = document
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return document_id

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._items.get(document_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


document_store = DocumentStore()
