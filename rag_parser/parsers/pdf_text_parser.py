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
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Маркеры битого текстового слоя (ToUnicode CMap не задан и т.п.).
_CID_MARKER_RE = re.compile(r"\(cid:\d+\)|\ufffd")

#: Управляющие символы (кроме \t\n\r) — признак мусорного слоя.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Гласные (ru+en): если буквенного текста много, а гласных нет — мусор.
_VOWELS = set("аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯAEIOUY")


def _is_readable_char(ch: str) -> bool:
    """Читаемый ли символ (кириллица/латиница/цифры/пунктуация)."""
    if ch.isalnum() and ord(ch) < 0xE000:  # без Private Use Area
        return True
    if ch.isspace():
        return True
    return ch in ".,;:!?\"'()[]{}«»—–-+=*/%№<>_~"


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
    def extract_from_doc(
        doc,
        pages: Optional[Sequence[int]] = None,
    ) -> Dict[int, str]:
        """Извлекает текст из указанных страниц уже открытого документа.

        Не переоткрывает файл — вызывается постранично из пайплайна.
        :param doc: открытый ``fitz.Document``.
        :param pages: номера страниц (0-based). ``None`` — все страницы.
        :return: ``{page_number: text}``.
        """
        total = len(doc)
        page_indices = pages if pages is not None else range(total)
        result: Dict[int, str] = {}
        for pno in page_indices:
            if pno < 0 or pno >= total:
                continue
            page = doc[pno]
            text = page.get_text("text") or ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = text.replace("\f", "\n")
            result[pno] = text.strip()
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

    @staticmethod
    def get_text_layer_area_ratios(path: Path) -> List[float]:
        """Доля площади текстовых блоков от площади страницы.

        Отсев страниц с формальным текстом, занимающим крошечную часть
        листа (колонтитул / номер страницы). Абзацы считаем по блокам
        ``page.get_text("blocks")`` (``block_type == 0`` — текст).
        """
        import fitz

        result: List[float] = []
        doc = fitz.open(str(path))
        try:
            for page in doc:
                page_area = float(page.rect.width * page.rect.height)
                if page_area <= 0:
                    result.append(0.0)
                    continue
                text_area = 0.0
                for block in page.get_text("blocks") or []:
                    if block and block[6] == 0:  # 0 — текстовый блок
                        x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
                        text_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
                result.append(min(1.0, text_area / page_area))
        finally:
            doc.close()
        return result

    @staticmethod
    def get_text_layer_valid_ratios(path: Path) -> List[float]:
        """Доля читаемых символов в текстовом слое каждой страницы.

        Отсев мусорного слоя: кракозябры (PUA, управляющие символы),
        маркеры ``(cid:N)``/``\\ufffd``, битый OCR-слой (нет гласных
        при длинном «буквенном» тексте). 1.0 для пустой страницы.
        """
        import fitz

        result: List[float] = []
        doc = fitz.open(str(path))
        try:
            for page in doc:
                text = page.get_text("text") or ""
                result.append(_valid_ratio(text))
        finally:
            doc.close()
        return result


def _valid_ratio(text: str) -> float:
    """Доля читаемых символов текста (1.0 — если текст пустой)."""
    total = len(text)
    if total == 0:
        return 1.0
    readable = sum(1 for c in text if _is_readable_char(c))
    ratio = readable / total
    if ratio < 1.0:
        # Битые маркеры и управляющие символы — сразу мусорный слой.
        pass
    letters = sum(1 for c in text if c.isalpha())
    if letters >= 30:
        vowels = sum(1 for c in text if c in _VOWELS)
        # Меньше 1.5% гласных среди букв — битый OCR/шифрованный слой.
        if vowels / letters < 0.015:
            return 0.0
    if _CID_MARKER_RE.search(text) or _CONTROL_CHARS_RE.search(text):
        return 0.0
    return ratio


__all__ = [
    "PdfTextParser",
    "page_metrics",
]


def page_metrics(text: str, text_area_ratio: float) -> Tuple[int, float, float]:
    """Пакет метрик страницы: (осмысленные символы, доля площади, валидность)."""
    meaningful = sum(1 for c in text if c.isalnum())
    return meaningful, text_area_ratio, _valid_ratio(text)
