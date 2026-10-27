"""Маршрутизатор PDF: определяет, какие страницы куда идут.

Три потока:
1. **text_parser** (Код №1) — страницы текстового PDF без таблиц.
2. **docling** (без OCR) — страницы текстового PDF с таблицами (раны).
3. **docling_ocr** (с OCR) — страницы без текстового слоя внутри
   текстового PDF; весь скан целиком.

Соседние табличные страницы объединяются в непрерывные «раны» и отдаются
в Docling одним вызовом (многостраничные таблицы не рвутся).

Сборка Markdown: результаты склеиваются строго в порядке номеров страниц
оригинала. Маркеры страниц в итоговый MD не попадают.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag_parser.parsers.pdf_classifier import PdfClassification
from rag_parser.parsers.pdf_table_detector import TableDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class PageAssignment:
    """Назначение одной страницы источнику."""

    page: int  # 0-based
    source: str  # "text_parser" | "docling" | "docling_ocr"


@dataclass
class TableRun:
    """Непрерывный диапазон табличных страниц (ран)."""

    pages: List[int]  # упорядоченные номера страниц
    source: str = "docling"  # "docling" | "docling_ocr"


@dataclass
class RoutingPlan:
    """План маршрутизации PDF."""

    assignments: List[PageAssignment] = field(default_factory=list)
    table_runs: List[TableRun] = field(default_factory=list)
    ocr_pages: List[int] = field(default_factory=list)  # страницы без слоя

    @property
    def text_parser_pages(self) -> List[int]:
        return [a.page for a in self.assignments if a.source == "text_parser"]

    @property
    def docling_pages(self) -> List[int]:
        return [a.page for a in self.assignments if a.source == "docling"]

    @property
    def docling_ocr_pages(self) -> List[int]:
        return [a.page for a in self.assignments if a.source == "docling_ocr"]


def group_table_runs(table_pages: List[int]) -> List[TableRun]:
    """Объединяет соседние табличные страницы в раны.

    «Соседние» = подряд идущие номера (0, 1, 2 → один ран [0, 1, 2]).
    Разрыв (0, 2) → два рана [0], [2].
    """
    if not table_pages:
        return []
    sorted_pages = sorted(set(table_pages))
    runs: List[TableRun] = []
    current: List[int] = [sorted_pages[0]]
    for p in sorted_pages[1:]:
        if p == current[-1] + 1:
            current.append(p)
        else:
            runs.append(TableRun(pages=current))
            current = [p]
    runs.append(TableRun(pages=current))
    return runs


def build_routing_plan(
    classification: PdfClassification,
    table_detection: Optional[TableDetectionResult] = None,
) -> RoutingPlan:
    """Строит план маршрутизации на основе классификации и детекции таблиц.

    Логика:
    * **Скан** (pdf_type="scan") — весь файл в Docling OCR. Код №1 не
      вызывается.
    * **Текстовый PDF** — постраничная детекция таблиц:
      - страницы без таблиц → text_parser (Код №1);
      - страницы с таблицами → docling (без OCR), объединённые в раны;
      - страницы без текстового слоя (scan_pages внутри текстового PDF)
        → docling_ocr, чтобы текст не терялся молча.
    """
    plan = RoutingPlan()

    if classification.pdf_type == "scan":
        # Весь файл — скан, всё в Docling OCR.
        plan.assignments = [
            PageAssignment(page=p, source="docling_ocr")
            for p in range(classification.total_pages)
        ]
        plan.ocr_pages = list(range(classification.total_pages))
        return plan

    # Текстовый PDF.
    table_pages = set(table_detection.table_pages) if table_detection else set()
    scan_pages = set(classification.scan_pages)

    for pno in range(classification.total_pages):
        if pno in scan_pages:
            # Страница без текстового слоя внутри текстового PDF → OCR.
            plan.assignments.append(PageAssignment(page=pno, source="docling_ocr"))
            plan.ocr_pages.append(pno)
        elif pno in table_pages:
            # Страница с таблицами → Docling без OCR.
            plan.assignments.append(PageAssignment(page=pno, source="docling"))
        else:
            # Обычная текстовая страница → Код №1.
            plan.assignments.append(PageAssignment(page=pno, source="text_parser"))

    # Раны табличных страниц (только docling, не ocr).
    docling_only = [a.page for a in plan.assignments if a.source == "docling"]
    plan.table_runs = group_table_runs(docling_only)

    return plan


# --------------------------------------------------------------------------- #
# Сборка Markdown в порядке страниц
# --------------------------------------------------------------------------- #

@dataclass
class PageResult:
    """Результат обработки одной страницы или рана."""

    pages: List[int]  # номера страниц
    source: str  # "text_parser" | "docling" | "docling_ocr"
    markdown: str


def assemble_markdown(results: List[PageResult]) -> str:
    """Склеивает результаты в порядке номеров страниц.

    Промпт: «результаты всех источников склеиваются строго в порядке номеров
    страниц оригинала. … Маркеры страниц в итоговый MD не попадают. …
    Склейка абзаца через границу страницы, если страница оборвалась без
    завершающей пунктуации, а следующая начинается со строчной, нужна
    по возможности и не блокирует остальное.»
    """
    # Сортируем по первой странице каждого результата.
    sorted_results = sorted(results, key=lambda r: r.pages[0] if r.pages else 0)
    parts: List[str] = []
    for r in sorted_results:
        md = r.markdown.strip()
        if not md:
            continue
        if parts:
            # Склейка абзаца через границу страницы: если предыдущий кусок
            # не заканчивается пунктуацией, а следующий начинается со
            # строчной — склеиваем через пробел (без \n\n).
            prev = parts[-1].rstrip()
            if prev and prev[-1] not in ".!?;:»)" and md and md[0].islower():
                parts[-1] = prev + " " + md
                continue
        parts.append(md)
    return "\n\n".join(parts).strip()


__all__ = [
    "PageAssignment",
    "PageResult",
    "RoutingPlan",
    "TableRun",
    "assemble_markdown",
    "build_routing_plan",
    "group_table_runs",
]
