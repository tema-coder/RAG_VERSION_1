"""Оркестратор PDF-пайплайна: classify → route → convert → assemble.

Промпт ``promt_for_parse_pdf.txt``, целевой пайплайн:
1. Классификация PDF: текстовый или скан.
2. Скан → весь файл в Docling (OCR, ru+en).
3. Текстовый PDF → детекция таблиц по страницам.
4. Маршрутизация: без таблиц → Код №1; с таблицами → Docling (без OCR);
   без слоя → Docling OCR.
5. Сборка Markdown в порядке страниц.
6. MD → Dedoc → заголовки → чанкинг (через существующий .md-путь).

Отказоустойчивость:
* Docling упал на табличных страницах → откат на Код №1, warning,
  ``metadata["lost_tables"]``.
* Docling упал на скане → ``DoclingError`` с ``X-Error-Type`` (не 500).

Наблюдаемость: в метаданные — тип PDF, списки страниц по источникам,
список Docling-ранов, время по этапам.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_parser.parsers.docling_converter import (
    DoclingError,
    DoclingResult,
    convert as docling_convert,
)
from rag_parser.parsers.pdf_classifier import PdfClassification, classify_pdf
from rag_parser.parsers.pdf_router import (
    PageResult,
    RoutingPlan,
    assemble_markdown,
    build_routing_plan,
)
from rag_parser.parsers.pdf_table_detector import TableDetectionResult, detect_tables
from rag_parser.parsers.pdf_text_parser import PdfTextParser

logger = logging.getLogger(__name__)


@dataclass
class PdfPipelineResult:
    """Результат работы PDF-пайплайна."""

    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(
    path: Path,
    *,
    min_chars_per_page: int = 50,
    text_pages_min_ratio: float = 0.5,
    ocr_engine: str = "easyocr",
    ocr_langs: tuple = ("ru", "en"),
) -> PdfPipelineResult:
    """Основной пайплайн обработки PDF.

    :return: ``PdfPipelineResult`` с Markdown и метаданными наблюдаемости.
    :raises DoclingError: если скан и Docling упал (не 500, а понятная ошибка).
    """
    timings: Dict[str, float] = {}
    metadata: Dict[str, Any] = {"source_pipeline": "pdf_router"}

    # --- Этап 1: классификация ---
    t0 = time.perf_counter()
    classification = classify_pdf(
        path,
        min_chars_per_page=min_chars_per_page,
        text_pages_min_ratio=text_pages_min_ratio,
    )
    timings["classify"] = time.perf_counter() - t0
    metadata["pdf_type"] = classification.pdf_type
    metadata["total_pages"] = classification.total_pages
    metadata["text_pages"] = classification.text_pages
    metadata["scan_pages"] = classification.scan_pages

    # --- Этап 2: детекция таблиц (только для текстовых PDF) ---
    table_detection: Optional[TableDetectionResult] = None
    if classification.pdf_type == "text":
        t0 = time.perf_counter()
        table_detection = detect_tables(path)
        timings["table_detect"] = time.perf_counter() - t0

    # --- Этап 3: план маршрутизации ---
    plan = build_routing_plan(classification, table_detection)
    metadata["routing"] = {
        "text_parser_pages": plan.text_parser_pages,
        "docling_pages": plan.docling_pages,
        "docling_ocr_pages": plan.docling_ocr_pages,
        "table_runs": [r.pages for r in plan.table_runs],
    }

    # --- Этап 4: выполнение плана ---
    results: List[PageResult] = []
    docling_runs: List[Dict[str, Any]] = []
    lost_tables: List[int] = []

    if classification.pdf_type == "scan":
        # Скан — весь файл в Docling OCR.
        t0 = time.perf_counter()
        try:
            dr = docling_convert(
                path, with_ocr=True, ocr_engine=ocr_engine, ocr_langs=ocr_langs
            )
            results.append(PageResult(
                pages=list(range(classification.total_pages)),
                source="docling_ocr", markdown=dr.markdown))
            docling_runs.append({"source": "docling_ocr", "pages": dr.page_range})
        except DoclingError:
            raise  # скан без Docling — ошибка с понятным типом
        timings["docling_ocr"] = time.perf_counter() - t0
    else:
        # Текстовый PDF — три потока.

        # 4a. Код №1: текстовые страницы без таблиц.
        text_pages = plan.text_parser_pages
        if text_pages:
            t0 = time.perf_counter()
            text_map = PdfTextParser.extract_pages(path, pages=text_pages)
            for pno, text in text_map.items():
                if text.strip():
                    results.append(PageResult(pages=[pno], source="text_parser",
                                              markdown=text))
            timings["text_parser"] = time.perf_counter() - t0

        # 4b. Docling (без OCR): табличные раны.
        for run in plan.table_runs:
            t0 = time.perf_counter()
            try:
                dr = docling_convert(path, pages=run.pages, with_ocr=False)
                results.append(PageResult(pages=run.pages, source="docling",
                                          markdown=dr.markdown))
                docling_runs.append({"source": "docling", "pages": dr.page_range})
            except DoclingError as exc:
                # Fallback: откат на Код №1 для этих страниц.
                logger.warning("docling_table_fallback pages=%s err=%s",
                               run.pages, exc)
                lost_tables.extend(run.pages)
                text_map = PdfTextParser.extract_pages(path, pages=run.pages)
                for pno, text in text_map.items():
                    if text.strip():
                        results.append(PageResult(pages=[pno], source="text_parser",
                                                  markdown=text))
            timings[f"docling_run_{run.pages[0]}"] = time.perf_counter() - t0

        # 4c. Docling OCR: страницы без текстового слоя.
        ocr_pages = plan.docling_ocr_pages
        if ocr_pages:
            t0 = time.perf_counter()
            try:
                dr = docling_convert(
                    path, pages=ocr_pages, with_ocr=True,
                    ocr_engine=ocr_engine, ocr_langs=ocr_langs)
                results.append(PageResult(pages=ocr_pages, source="docling_ocr",
                                          markdown=dr.markdown))
                docling_runs.append({"source": "docling_ocr", "pages": dr.page_range})
            except DoclingError as exc:
                logger.warning("docling_ocr_fallback pages=%s err=%s",
                               ocr_pages, exc)
                # Страницы без слоя и без OCR — текст теряется молча,
                # но не валим весь документ.
                lost_tables.extend(ocr_pages)
            timings["docling_ocr_pages"] = time.perf_counter() - t0

    # --- Этап 5: сборка Markdown ---
    t0 = time.perf_counter()
    markdown = assemble_markdown(results)
    timings["assemble"] = time.perf_counter() - t0

    metadata["docling_runs"] = docling_runs
    metadata["lost_tables"] = lost_tables
    metadata["timings"] = timings

    logger.info(
        "pdf_pipeline_done",
        extra={
            "file": path.name,
            "pdf_type": classification.pdf_type,
            "results": len(results),
            "markdown_len": len(markdown),
            "lost_tables": len(lost_tables),
        },
    )
    return PdfPipelineResult(markdown=markdown, metadata=metadata)


__all__ = ["PdfPipelineResult", "run_pipeline"]
