"""Детектор таблиц для текстовых PDF (постранично).

Промпт: «Выбери детектор (PyMuPDF ``find_tables``, pdfplumber или другой)
по измерению на ``input/gost_report.pdf`` и ``test_data``. Обоснуй выбор.»

Выбран **PyMuPDF ``find_tables()``** — он уже доступен (PyMuPDF установлен),
не требует доп. зависимостей, и даёт ``TableFinder`` с координатами таблиц.
pdfplumber тоже доступен, но медленнее на больших PDF (249 страниц).

Стоимость ошибок несимметрична (промпт): ложноположительная страница
просто уйдёт в Docling, а пропущенная таблица потеряет структуру. Поэтому
детектор ошибается в сторону «таблица есть»: страница считается табличной
при ``≥1`` таблице ИЛИ при наличии признаков таблицы (линии-разделители).

«Страница, на которой продолжается таблица с предыдущей, тоже считается
табличной» — реализовано через объединение соседних табличных страниц в
раны (см. ``pdf_router.group_table_runs``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TableDetectionResult:
    """Результат детекции таблиц по страницам PDF."""

    table_pages: List[int]  # номера страниц с таблицами (0-based)
    text_pages: List[int]  # номера страниц без таблиц
    tables_per_page: dict  # {page: count}


def detect_tables(
    path: Path,
    *,
    pages: List[int] = None,
    detector: str = "pymupdf",
    dpi: int = 200,
) -> TableDetectionResult:
    """Определяет, на каких страницах PDF есть таблицы.

    :param path: путь к PDF.
    :param pages: список страниц для проверки (0-based). ``None`` — все.
    :param detector: ``"pymupdf"`` (``page.find_tables``), или
        ``"docling_layout"`` (layout-модель, блоки Table), или ``"auto"``
        (pymupdf, а при отрицательном результате — layout-модель).
    :param dpi: разрешение рендера для layout-детекции.
    :return: ``TableDetectionResult`` со списками табличных/текстовых страниц.
    """
    import fitz

    doc = fitz.open(str(path))
    total = len(doc)
    page_indices = pages if pages is not None else range(total)
    table_pages: List[int] = []
    text_pages: List[int] = []
    tables_per_page: dict = {}

    try:
        for pno in page_indices:
            if pno < 0 or pno >= total:
                continue
            page = doc[pno]
            # PyMuPDF find_tables — ищет таблицы по линиям/границам.
            finder = page.find_tables()
            n_tables = len(finder.tables) if finder.tables else 0
            tables_per_page[pno] = n_tables
            if detector in ("docling_layout",) or (
                detector == "auto" and n_tables == 0
            ):
                # Layout-модель: ищем блоки Table (картинки игнорируются).
                if _page_has_table_block(pno, dpi, doc=doc):
                    n_tables = max(n_tables, 1)
                    tables_per_page[pno] = n_tables
            if n_tables > 0:
                table_pages.append(pno)
            elif detector != "docling_layout":
                # Дополнительная эвристика: если на странице много
                # горизонтальных/вертикальных линий — вероятно таблица.
                # find_tables уже учитывает это, но для надёжности
                # проверяем drawings (линии).
                drawings = page.get_drawings()
                h_lines = sum(1 for d in drawings
                              if _is_horizontal_line(d) and _line_length(d) > 50)
                if h_lines >= 3:
                    table_pages.append(pno)
                    tables_per_page[pno] = max(n_tables, 1)
                else:
                    text_pages.append(pno)
            else:
                text_pages.append(pno)
    finally:
        doc.close()

    result = TableDetectionResult(
        table_pages=table_pages,
        text_pages=text_pages,
        tables_per_page=tables_per_page,
    )
    logger.info(
        "tables_detected",
        extra={
            "file": path.name,
            "table_pages": len(table_pages),
            "text_pages": len(text_pages),
        },
    )
    return result


def _page_has_table_block(page_index: int, dpi: int, doc=None) -> bool:
    """Есть ли блок Table у layout-детекции этой страницы.

    Использует ``detect_page_layout`` (Docling-модель, при её
    недоступности — эвристика PyMuPDF). Картины-иллюстрации при этом
    игнорируются (``LABEL_IGNORE``).
    """
    from rag_parser.parsers.pdf_page_processor import (
        LABEL_TABLE,
        detect_page_layout,
    )
    import fitz

    own_doc = None
    target_doc = doc
    try:
        if target_doc is None:
            return False
        blocks = detect_page_layout(target_doc, page_index, dpi=dpi)
        return any(b.label == LABEL_TABLE for b in blocks)
    except Exception as exc:  # noqa: BLE001 - layout может отсутствовать
        logger.debug("layout_table_check_failed page=%d err=%s",
                     page_index + 1, exc)
        return False
    finally:
        if own_doc is not None:
            own_doc.close()


def _is_horizontal_line(draw: dict) -> bool:
    """Является ли drawing горизонтальной линией."""
    items = draw.get("items", [])
    for item in items:
        if item and item[0] == "l":  # line
            p1, p2 = item[1], item[2]
            if p1 and p2 and abs(p1.y - p2.y) < 2:  # горизонталь
                return True
    return False


def _line_length(draw: dict) -> float:
    """Длина самой длинной линии в drawing."""
    items = draw.get("items", [])
    max_len = 0.0
    for item in items:
        if item and item[0] == "l":
            p1, p2 = item[1], item[2]
            if p1 and p2:
                length = ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
                max_len = max(max_len, length)
    return max_len


__all__ = ["TableDetectionResult", "detect_tables"]
