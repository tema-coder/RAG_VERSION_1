"""Код №1: парсер чистого текста для текстовых PDF (PyMuPDF).

Извлекает текст постранично через ``fitz`` (PyMuPDF) — лёгкая библиотека,
не требующая Java и tesseract. Используется для страниц текстового PDF
БЕЗ таблиц: даёт плоский текст без ``#``-разметки (заголовки потом найдёт
``header_promoter`` / ``priority_rules``).

Промпт ``promt_for_parse_pdf.txt``: «Код №1: парсер чистого текста для
текстовых PDF (работает корректно, не переписывать логику, только
адаптировать интерфейс)». В репозитории готового кода не было — написан
с нуля на доступном PyMuPDF.

Интерфейс:
    >>> from rag_parser.parsers.pdf_text_parser import PdfTextParser
    >>> pages = PdfTextParser.extract_pages(Path("doc.pdf"), pages=[0, 1, 2])
    >>> # → {0: "текст страницы 0", 1: "текст страницы 1", ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class PdfTextParser:
    """Постраничное извлечение текста из текстового PDF через PyMuPDF."""

    def __init__(self, *, min_chars_per_page: int = 1) -> None:
        """ :param min_chars_per_page: минимальное число символов на странице,
            чтобы считать её «текстовой» (для классификатора). """
        self.min_chars_per_page = min_chars_per_page

    @staticmethod
    def extract_pages(
        path: Path,
        pages: Optional[Sequence[int]] = None,
    ) -> Dict[int, str]:
        """Извлекает текст из указанных страниц PDF.

        :param path: путь к PDF.
        :param pages: номера страниц (0-based). ``None`` — все страницы.
        :return: ``{page_number: text}`` — текст по страницам.
        """
        import fitz  # PyMuPDF

        result: Dict[int, str] = {}
        doc = fitz.open(str(path))
        try:
            total = len(doc)
            page_indices = pages if pages is not None else range(total)
            for pno in page_indices:
                if pno < 0 or pno >= total:
                    continue
                page = doc[pno]
                text = page.get_text("text") or ""
                # Нормализуем переносы: заменяем \r\n на \n, убираем \f.
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                text = text.replace("\f", "\n")
                result[pno] = text.strip()
        finally:
            doc.close()
        return result

    @staticmethod
    def get_text_layer_chars(path: Path) -> List[int]:
        """Число осмысленных символов на каждой странице (для классификатора).

        «Осмысленные» — после удаления пробелов/переносов/пунктуации.
        Используется классификатором PDF (текстовый vs скан).
        """
        import fitz

        result: List[int] = []
        doc = fitz.open(str(path))
        try:
            for page in doc:
                text = page.get_text("text") or ""
                # Считаем только буквы/цифры (не пробелы/переносы).
                meaningful = sum(1 for c in text if c.isalnum())
                result.append(meaningful)
        finally:
            doc.close()
        return result
