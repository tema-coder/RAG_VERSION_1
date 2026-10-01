from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentError(Exception):
    """Base error raised by parsers and chunkers."""

    def __init__(self, message: str, *, filename: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.filename = filename


class UnsupportedFormatError(DocumentError):
    def __init__(self, extension: str, supported: Optional[List[str]] = None) -> None:
        message = f"Unsupported file format: {extension!r}"
        if supported:
            message += f". Supported: {', '.join(sorted(supported))}"
        super().__init__(message)
        self.extension = extension


class DocumentParseError(DocumentError):
    def __init__(self, message: str, *, filename: Optional[str] = None) -> None:
        super().__init__(message, filename=filename)


class FileTooLargeError(DocumentError):
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        super().__init__(
            f"File size {size_bytes} bytes exceeds limit of {max_bytes} bytes"
        )
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


@dataclass
class Chunk:
    index: int
    text: str
    tokens: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_for_embedding: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParseResult:
    document_id: str
    filename: str
    format: str
    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: Optional[List[Chunk]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = [chunk.to_dict() for chunk in self.chunks or []]
        return payload


class BaseParser(ABC):
    format: str = "unknown"
    extensions: tuple = ()

    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        raise NotImplementedError

    def _new_document_id(self) -> str:
        import uuid

        return uuid.uuid4().hex


class BaseChunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> List[Chunk]:
        raise NotImplementedError
