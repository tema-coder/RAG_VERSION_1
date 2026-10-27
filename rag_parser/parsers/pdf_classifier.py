"""Классификатор PDF: текстовый или скан.

Страница считается «текстовой», если на ней не меньше
``PDF_TEXT_MIN_CHARS_PER_PAGE`` осмысленных символов в текстовом слое.
Документ считается сканом, если доля текстовых страниц меньше
``PDF_TEXT_PAGES_MIN_RATIO``.

Промпт: «Используй лёгкую библиотеку, не требующую Java и tesseract
(PyMuPDF или pdfminer, что уже есть в окружении).» — используем PyMuPDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag_parser.parsers.pdf_text_parser import PdfTextParser

logger = logging.getLogger(__name__)


@dataclass
class PdfClassification:
    """Результат классификации PDF."""

    pdf_type: str  # "text" | "scan"
    total_pages: int
    text_pages: List[int]  # номера текстовых страниц (0-based)
    scan_pages: List[int]  # номера страниц без текстового слоя
    chars_per_page: List[int]  # осмысленные символы по страницам

    @property
    def text_pages_ratio(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return len(self.text_pages) / self.total_pages


def classify_pdf(
    path: Path,
    *,
    min_chars_per_page: int = 50,
    text_pages_min_ratio: float = 0.5,
) -> PdfClassification:
    """Классифицирует PDF как текстовый или скан.

    :param path: путь к PDF.
    :param min_chars_per_page: минимальное число осмысленных символов
        на странице, чтобы считать её «текстовой» (старт 50).
    :param text_pages_min_ratio: минимальная доля текстовых страниц,
        чтобы считать документ текстовым (старт 0.5).
    :return: ``PdfClassification`` с типом, списками страниц и статистикой.
    """
    chars = PdfTextParser.get_text_layer_chars(path)
    total = len(chars)
    text_pages: List[int] = []
    scan_pages: List[int] = []
    for i, count in enumerate(chars):
        if count >= min_chars_per_page:
            text_pages.append(i)
        else:
            scan_pages.append(i)

    if total == 0:
        pdf_type = "scan"  # пустой PDF — считаем сканом
    else:
        ratio = len(text_pages) / total
        pdf_type = "text" if ratio >= text_pages_min_ratio else "scan"

    result = PdfClassification(
        pdf_type=pdf_type,
        total_pages=total,
        text_pages=text_pages,
        scan_pages=scan_pages,
        chars_per_page=chars,
    )
    logger.info(
        "pdf_classified",
        extra={
            "file": path.name,
            "pdf_type": pdf_type,
            "total_pages": total,
            "text_pages": len(text_pages),
            "scan_pages": len(scan_pages),
            "ratio": result.text_pages_ratio,
        },
    )
    return result


__all__ = ["PdfClassification", "classify_pdf"]
