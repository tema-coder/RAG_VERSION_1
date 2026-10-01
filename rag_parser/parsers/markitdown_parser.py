import re
from pathlib import Path
from typing import Any, List

from markitdown import MarkItDown
from markitdown._exceptions import (
    FileConversionException,
    MarkItDownException,
    MissingDependencyException,
    UnsupportedFormatException,
)

from rag_parser.core.interfaces import BaseParser, DocumentParseError, ParseResult

SUPPORTED_EXTENSIONS: List[str] = [
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".txt",
    ".md",
]

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

SIGNATURES = {
    ".pdf": (b"%PDF", "PDF"),
    ".docx": (b"PK\x03\x04", "ZIP (OOXML)"),
}

STRUCTURED_FORMATS = (".pdf", ".docx")

_converter: Any = None


def get_converter() -> MarkItDown:
    global _converter
    if _converter is None:
        _converter = MarkItDown()
    return _converter


class MarkItDownParser(BaseParser):
    format = "markitdown"

    def __init__(self, converter: Any = None) -> None:
        self.converter = converter if converter is not None else get_converter()

    def supported_extensions(self) -> List[str]:
        return list(SUPPORTED_EXTENSIONS)

    def parse(self, file_path: Path) -> ParseResult:
        path = Path(file_path)
        if not path.exists():
            raise DocumentParseError(f"File not found: {path}", filename=path.name)

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise DocumentParseError(
                f"Cannot read file: {exc}", filename=path.name
            ) from exc

        if size_bytes == 0:
            raise DocumentParseError("File is empty", filename=path.name)
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise DocumentParseError(
                f"File is too large for conversion: {size_bytes} bytes",
                filename=path.name,
            )

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentParseError(
                f"Unsupported extension {extension!r} for MarkItDown parser. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                filename=path.name,
            )

        self._validate_container(path, extension)

        try:
            result = self.converter.convert(str(path))
        except (FileConversionException, MissingDependencyException) as exc:
            raise DocumentParseError(
                f"MarkItDown failed to convert the file: {exc}",
                filename=path.name,
            ) from exc
        except UnsupportedFormatException as exc:
            raise DocumentParseError(
                f"MarkItDown does not support this file: {exc}", filename=path.name
            ) from exc
        except MarkItDownException as exc:
            raise DocumentParseError(
                f"MarkItDown failed to convert the file: {exc}", filename=path.name
            ) from exc
        except Exception as exc:  # noqa: BLE001 - converters raise assorted errors
            raise DocumentParseError(
                f"Failed to convert the file: {exc}", filename=path.name
            ) from exc

        markdown_text = self._clean(result.text_content or "")
        if extension in STRUCTURED_FORMATS and self._looks_unconverted(
            path, markdown_text
        ):
            raise DocumentParseError(
                f"MarkItDown returned the raw file content for {extension}: "
                "the file is not a valid document",
                filename=path.name,
            )
        if not markdown_text:
            raise DocumentParseError(
                "Converted document contains no text", filename=path.name
            )

        lines = markdown_text.splitlines()
        return ParseResult(
            document_id=self._new_document_id(),
            filename=path.name,
            format=self._detect_format(extension),
            markdown=markdown_text,
            metadata={
                "format": self._detect_format(extension),
                "extension": extension,
                "size_bytes": size_bytes,
                "lines": len(lines),
                "characters": len(markdown_text),
                "converter": "markitdown",
            },
        )

    def _looks_unconverted(self, path: Path, markdown_text: str) -> bool:
        """True when MarkItDown fell back to its plain-text converter.

        A broken container is dumped verbatim, so the converted text equals the
        file content and no real conversion happened.
        """
        if not markdown_text:
            return False
        try:
            raw = path.read_bytes().decode("utf-8", errors="ignore").strip()
        except OSError:
            return False
        return bool(raw) and raw == markdown_text.strip()

    def _validate_container(self, path: Path, extension: str) -> None:
        """Rejects files whose bytes do not match their extension.

        MarkItDown falls back to its plain-text converter when a structured
        format is broken, which would silently return binary noise as Markdown.
        """
        expected = SIGNATURES.get(extension)
        if expected is None:
            return
        signature, description = expected
        with path.open("rb") as handle:
            header = handle.read(len(signature))
        if header != signature:
            raise DocumentParseError(
                f"File content does not match the {extension} extension "
                f"(expected a {description} container)",
                filename=path.name,
            )

    def _clean(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\f", "\n\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _detect_format(self, extension: str) -> str:
        if extension in (".html", ".htm"):
            return "html"
        if extension == ".md":
            return "markdown"
        if extension == ".txt":
            return "text"
        return extension.lstrip(".")
