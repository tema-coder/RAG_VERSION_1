"""Классификатор PDF: текстовый или скан.

Страница считается «текстовой», если выполняются ВСЕ условия:
* не меньше ``PDF_TEXT_MIN_CHARS_PER_PAGE`` осмысленных символов
  в текстовом слое;
* текст занимает не меньше ``PDF_TEXT_MIN_AREA_RATIO`` площади страницы
  (отсев страниц с одним колонтитулом);
* текстовый слой адекватный: доля читаемых символов не ниже
  ``PDF_TEXT_MIN_VALID_CHAR_RATIO`` (отсев кракозябр и битого OCR-слоя).

Документ считается сканом, если доля текстовых страниц меньше
``PDF_TEXT_PAGES_MIN_RATIO``. Решение принимается по документу целиком;
пороги вынесены в конфиг (``core.config.Settings``).

Смешанные случаи (страницы-картинки внутри текстового PDF и мусорный
слой) попадают в ``scan_pages`` и маршрутизируются в OCR-ветку.

Промпт: «Используй лёгкую библиотеку, не требующую Java и tesseract
(PyMuPDF или pdfminer, что уже есть в окружении).» — используем PyMuPDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag_parser.parsers.pdf_text_parser import (
    PdfTextParser,
    _valid_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class PdfClassification:
    """Результат классификации PDF."""

    pdf_type: str  # "text" | "scan"
    total_pages: int
    text_pages: List[int]  # номера текстовых страниц (0-based)
    scan_pages: List[int]  # страницы без адекватного текстового слоя
    chars_per_page: List[int]  # осмысленные символы по страницам
    area_ratios: List[float] = None  # доля площади текста по страницам
    valid_ratios: List[float] = None  # доля читаемых символов

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
    min_area_ratio: float = 0.002,
    min_valid_char_ratio: float = 0.7,
) -> PdfClassification:
    """Классифицирует PDF как текстовый или скан.

    :param path: путь к PDF.
    :param min_chars_per_page: минимальное число осмысленных символов
        на странице, чтобы считать её «текстовой» (старт 50).
    :param text_pages_min_ratio: минимальная доля текстовых страниц,
        чтобы считать документ текстовым (старт 0.5).
    :param min_area_ratio: минимальная доля площади текста на странице.
    :param min_valid_char_ratio: минимальная доля читаемых символов.
    :return: ``PdfClassification`` с типом, списками страниц и статистикой.
    """
    chars = PdfTextParser.get_text_layer_chars(path)
    area_ratios = PdfTextParser.get_text_layer_area_ratios(path)
    valid_ratios = PdfTextParser.get_text_layer_valid_ratios(path)
    text_pages: List[int] = []
    scan_pages: List[int] = []
    for i, count in enumerate(chars):
        area = area_ratios[i] if i < len(area_ratios) else 0.0
        valid = valid_ratios[i] if i < len(valid_ratios) else 0.0
        if (
            count >= min_chars_per_page
            and area >= min_area_ratio
            and valid >= min_valid_char_ratio
        ):
            text_pages.append(i)
        else:
            scan_pages.append(i)

    total = len(chars)
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
        area_ratios=area_ratios,
        valid_ratios=valid_ratios,
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
