"""PDF/изображения → Markdown через Docling.

Тонкая обёртка над ``docling_converter``: здесь нет классификаторов,
постраничного рендера, EasyOCR-по-кропам, PyMuPDF-фолбэков и маршрутизации.
Docling — единственный механизм извлечения содержимого PDF и изображений.

Единый интерфейс: ``parse_pdf(path) -> markdown_str``.
Возврат ``run_pipeline`` — ``PdfPipelineResult`` с Markdown и метаданными
для существующего RAG-пайплайна (``DedocParser._parse_via_docling``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from rag_parser.parsers import docling_converter

logger = logging.getLogger(__name__)


@dataclass
class PdfPipelineResult:
    """Результат работы PDF/изображение-пайплайна."""

    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(path: Path) -> PdfPipelineResult:
    """Конвертирует PDF или изображение в Markdown через Docling.

    Ошибки Docling не скрываются и не заменяются результатом другого
    парсера — вызывающий код получает ``DoclingError``.
    """
    source = Path(path)
    result = docling_converter.convert_to_markdown(source)
    metadata: Dict[str, Any] = {
        "source_pipeline": "docling",
        "input_format": result.source,
        "pages": result.page_count,
    }
    metadata.update(result.metadata)
    logger.info(
        "docling_pipeline_done",
        extra={
            "file": source.name,
            "format": result.source,
            "markdown_len": len(result.markdown),
            "pages": result.page_count,
        },
    )
    return PdfPipelineResult(markdown=result.markdown, metadata=metadata)


def parse_pdf(path: str | Path) -> str:
    """Единый интерфейс: ``parse_pdf(path) -> markdown_str``."""
    return run_pipeline(Path(path)).markdown


__all__ = ["PdfPipelineResult", "run_pipeline", "parse_pdf"]
