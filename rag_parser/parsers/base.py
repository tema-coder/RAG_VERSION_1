import csv
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_parser.core.interfaces import DocumentParseError, ParseResult

MAX_CELL_LENGTH = 512
TABLE_CELL_LIMIT = 5000


def read_bytes(file_path: Path) -> bytes:
    path = Path(file_path)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DocumentParseError(f"Cannot read file: {exc}", filename=path.name) from exc


def decode_text(data: bytes) -> tuple:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def read_text_file(file_path: Path) -> tuple:
    path = Path(file_path)
    if not path.exists():
        raise DocumentParseError(f"File not found: {path}", filename=path.name)
    data = read_bytes(path)
    if not data:
        raise DocumentParseError("File is empty", filename=path.name)
    text, encoding = decode_text(data)
    return text, encoding, len(data)


def escape_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|").strip()
    if len(text) > MAX_CELL_LENGTH:
        text = text[:MAX_CELL_LENGTH] + "…"
    return text


def normalize_row(row: Sequence[Any], width: Optional[int] = None) -> List[str]:
    cells = [escape_cell(value) for value in row]
    if width is not None:
        cells = (cells + [""] * width)[:width]
    return cells


def rows_to_markdown_table(
    rows: Iterable[Sequence[Any]],
    max_rows: int = TABLE_CELL_LIMIT,
) -> tuple:
    materialized: List[List[str]] = []
    width = 0
    for row in rows:
        if isinstance(row, (list, tuple)):
            width = max(width, len(row))
    for index, row in enumerate(rows):
        if index >= max_rows:
            break
        materialized.append(normalize_row(row, width))
    if not materialized:
        return "", 0, 0

    header = materialized[0]
    body = materialized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines), len(materialized), width


def build_result(
    file_path: Path,
    fmt: str,
    markdown: str,
    metadata: Dict[str, Any],
) -> ParseResult:
    path = Path(file_path)
    return ParseResult(
        document_id="",
        filename=path.name,
        format=fmt,
        markdown=markdown,
        metadata=metadata,
    )


def detect_dialect(sample: str) -> str:
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        for candidate in (";", "\t", ",", "|"):
            if candidate in sample:
                return candidate
        return ","


def parse_csv_rows(text: str, delimiter: str) -> List[List[str]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [row for row in reader]
