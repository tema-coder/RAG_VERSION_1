import re
from typing import List, Optional

import structlog

from rag_parser.chunkers.base import BaseChunker, Chunk, count_tokens

logger = structlog.get_logger(__name__)

CHUNKER_NAME = "table"

_SEPARATOR_CELL_RE = re.compile(r"^:?-{1,}:?$")


def _split_row(line: str) -> List[str]:
    """Splits a Markdown table line into cells.

    A trailing ``|`` produces an empty tail element which is filtered out.
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_row(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(
        _SEPARATOR_CELL_RE.match(cell.replace(" ", "")) for cell in cells
    )


def _is_empty_row(line: str) -> bool:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    return all(not cell.strip() for cell in _split_row(line))


def _is_markdown_table(block: str) -> bool:
    """Detects a Markdown table by the presence of a separator row."""
    lines = [line for line in block.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return any(_is_separator_row(line) for line in lines)


def _clean_table_header(block: str) -> str:
    """Removes leading empty placeholder rows (``| | |``) and the separator after them.

    The first remaining data row is promoted to the header: a fresh separator
    row is inserted right after it.
    """
    lines = [line for line in block.strip().splitlines() if line.strip()]
    index = 0
    while index < len(lines) and _is_empty_row(lines[index]):
        index += 1
    dropped_separator = False
    if index > 0 and index < len(lines) and _is_separator_row(lines[index]):
        index += 1
        dropped_separator = True
    cleaned = lines[index:]
    if dropped_separator and cleaned:
        width = max(len(_split_row(cleaned[0])), 1)
        separator = "|" + "|".join(["---"] * width) + "|"
        cleaned = [cleaned[0], separator, *cleaned[1:]]
    return "\n".join(cleaned)


def table_to_natural_language(table_markdown: str) -> str:
    """Converts a Markdown table into natural-language lines for embeddings.

    Each data row becomes ``Header: value, Header: value, ...``. Rows whose
    cell count does not match the header are skipped with a warning.
    """
    lines = [line for line in table_markdown.strip().splitlines() if line.strip()]
    rows = [line for line in lines if not _is_separator_row(line)]
    if not rows:
        return ""
    headers = _split_row(rows[0])
    sentences: List[str] = []
    for line in rows[1:]:
        cells = _split_row(line)
        if len(cells) != len(headers):
            logger.warning(
                "table_row_skipped",
                reason="cell count does not match header",
                expected=len(headers),
                actual=len(cells),
                row=line[:200],
            )
            continue
        sentences.append(
            ", ".join(f"{header}: {value}" for header, value in zip(headers, cells))
        )
    return "\n".join(sentences)


def _count_data_rows(table_markdown: str) -> int:
    lines = [line for line in table_markdown.strip().splitlines() if line.strip()]
    rows = [line for line in lines if not _is_separator_row(line)]
    return max(len(rows) - 1, 0)


class TableChunkerRAG(BaseChunker):
    """Table chunker: **ровно одна строка данных на чанк** (ТЗ 3.3).

    Каждый чанк содержит заголовок таблицы, разделитель и ОДНУ строку
    данных. Количество строк не настраивается: это жёсткое требование
    бизнес-логики (поиск по конкретной строке таблицы), а не параметр
    качества, поэтому параметра ``chunk_size`` здесь нет вовсе.

    Используется chonkie ``TableChunker(tokenizer="row", chunk_size=1)``:
    заголовок сам прикрепляется к каждому чанку. Если установленная версия
    chonkie не поддерживает ``tokenizer="row"``, используется ручной
    fallback с тем же инвариантом (по одной строке на чанк).
    """

    name = CHUNKER_NAME

    #: Единственное допустимое число строк данных в чанке таблицы.
    ROWS_PER_CHUNK: int = 1

    def __init__(self, source: str = "document") -> None:
        self.source = source
        self._fallback = False
        try:
            from chonkie import TableChunker as _ChonkieTableChunker

            self.chunker = _ChonkieTableChunker(
                tokenizer="row", chunk_size=self.ROWS_PER_CHUNK
            )
        except Exception:  # noqa: BLE001 - old chonkie without row tokenizers
            logger.warning("table_chunker_fallback", reason="row tokenizer unsupported")
            self.chunker = None
            self._fallback = True

    def _fallback_chunks(self, table_markdown: str) -> List[str]:
        """Ручная нарезка: шапка + ровно одна строка данных.

        Используется только если chonkie не даёт ``tokenizer="row"``.
        """
        lines = [
            line for line in table_markdown.strip().splitlines() if line.strip()
        ]
        rows = [line for line in lines if not _is_separator_row(line)]
        if not rows:
            return []
        header, data = rows[0], rows[1:]
        separator = "|" + "|".join(["---"] * len(_split_row(header))) + "|"
        if not data:
            return []
        return [
            "\n".join([header, separator, row]) for row in data
        ]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []
        cleaned = _clean_table_header(text)
        if not cleaned or not _is_markdown_table(cleaned):
            return []
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        if self._fallback or self.chunker is None:
            raw_texts = self._fallback_chunks(cleaned)
        else:
            raw_texts = [raw.text for raw in self.chunker.chunk(cleaned)]
        chunks: List[Chunk] = []
        for index, table_text in enumerate(raw_texts):
            natural = table_to_natural_language(table_text)
            chunks.append(
                Chunk(
                    index=index,
                    text=table_text,
                    tokens=count_tokens(table_text),
                    source=source,
                    metadata={
                        "chunk_index": index,
                        "chunker": self.name,
                        "source_type": "markdown_table",
                        "data_rows": _count_data_rows(table_text),
                        "rows_per_chunk": self.ROWS_PER_CHUNK,
                        **extra,
                    },
                    text_for_embedding=natural or None,
                )
            )
        return chunks
