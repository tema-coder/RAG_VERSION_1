"""Оркестратор PDF-пайплайна (ТЗ promt_for_agent.txt).

Единый интерфейс: ``parse_pdf(path) -> markdown_str`` (см. ниже).

Шаги:
1. **Шаг 1** — классификация: текстовый PDF или скан
   (``classify_pdf``: символы + доля площади текста + адекватность слоя).
   Смешанные страницы (картинки в текстовом PDF, мусорный слой) идут в
   OCR-ветку.
2. **Шаг 2 (текстовый PDF)** — постраничная детекция таблиц:
   * без таблиц → быстрый PyMuPDF → Markdown (без EasyOCR/TableFormer);
   * с таблицами → рендер в изображение → layout → EasyOCR (текст) +
     TableFormer (таблицы) → Markdown в порядке чтения.
3. **Шаг 3 (скан)** — те же страницы через layout + EasyOCR + TableFormer.
4. **Шаг 4** — итоговый Markdown (не исходный PDF) уходит в dedoc
   (это делает ``DedocParser._parse_pdf_via_pipeline``).

Модели (EasyOCR, layout, TableFormer) — process-wide singletons
(``pdf_models``), создаются один раз и переиспользуются.

Отказоустойчивость: падение TableFormer/EasyOCR на блоке — fallback
внутри страницы; падение обработки страницы целиком — откат на
PyMuPDF (если есть слой), страница не теряется.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_parser.parsers.pdf_classifier import PdfClassification, classify_pdf
from rag_parser.parsers.pdf_page_processor import process_complex_page
from rag_parser.parsers.pdf_router import (
    PageResult,
    assemble_markdown,
    build_routing_plan,
)
from rag_parser.parsers.pdf_table_detector import detect_tables
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
    min_area_ratio: float = 0.002,
    min_valid_char_ratio: float = 0.7,
    easyocr_langs: Tuple[str, ...] = ("ru", "en"),
    render_dpi: int = 200,
) -> PdfPipelineResult:
    """Основной пайплайн обработки PDF (ТЗ шаги 1–3).

    Ошибка любого этапа не теряет страницу: внутри страниц действует
    многоуровневый fallback (layout → OCR → текстовый слой).
    :return: ``PdfPipelineResult`` с Markdown и метаданными наблюдаемости.
    """
    timings: Dict[str, float] = {}
    metadata: Dict[str, Any] = {"source_pipeline": "pdf_page_router"}

    # --- Этап 1: классификация (шаг 1 ТЗ) ---
    t0 = time.perf_counter()
    classification = classify_pdf(
        path,
        min_chars_per_page=min_chars_per_page,
        text_pages_min_ratio=text_pages_min_ratio,
        min_area_ratio=min_area_ratio,
        min_valid_char_ratio=min_valid_char_ratio,
    )
    timings["classify"] = time.perf_counter() - t0
    metadata["pdf_type"] = classification.pdf_type
    metadata["total_pages"] = classification.total_pages
    metadata["text_pages"] = classification.text_pages
    metadata["scan_pages"] = classification.scan_pages

    import fitz
    doc = fitz.open(str(path))
    try:
        # --- Этап 2: детекция таблиц (только для текстовых PDF) ---
        table_detection = None
        if classification.pdf_type == "text":
            t0 = time.perf_counter()
            table_detection = detect_tables(path)
            timings["table_detect"] = time.perf_counter() - t0

        # --- Этап 3: план маршрутизации ---
        plan = build_routing_plan(classification, table_detection)
        metadata["routing"] = {
            "text_parser_pages": plan.text_parser_pages,
            "table_pages": plan.docling_pages,
            "ocr_pages": plan.docling_ocr_pages,
            "table_runs": [r.pages for r in plan.table_runs],
        }

        # --- Этап 4: постраничная обработка (шаги 2 и 3 ТЗ) ---
        results: List[PageResult] = []
        failed_pages: List[int] = []
        t_start = time.perf_counter()
        for pno in range(classification.total_pages):
            t0 = time.perf_counter()
            md, route = _process_single_page(
                doc, pno, plan=plan, easyocr_langs=easyocr_langs,
                render_dpi=render_dpi,
            )
            timings[f"page_{pno}"] = time.perf_counter() - t0
            if route.startswith(("fallback", "empty")) or not md.strip():
                failed_pages.append(pno)
            if md.strip():
                results.append(PageResult(pages=[pno], source=route, markdown=md))
        timings["process_all_pages"] = time.perf_counter() - t_start

        # --- Этап 5: сборка Markdown в порядке страниц ---
        t0 = time.perf_counter()
        markdown = assemble_markdown(results)
        timings["assemble"] = time.perf_counter() - t0
    finally:
        doc.close()

    metadata["page_routes"] = [r.source for r in sorted(
        results, key=lambda r: r.pages[0])]
    metadata["failed_pages"] = failed_pages
    metadata["timings"] = timings
    logger.info(
        "pdf_pipeline_done",
        extra={
            "file": path.name,
            "pdf_type": classification.pdf_type,
            "markdown_len": len(markdown),
            "failed_pages": len(failed_pages),
        },
    )
    return PdfPipelineResult(markdown=markdown, metadata=metadata)


def _process_single_page(
    doc, pno: int, *,
    plan, easyocr_langs: Tuple[str, ...], render_dpi: int,
) -> Tuple[str, str]:
    """Маршрут + обработка одной страницы. Возвращает ``(md, маршрут)``."""
    if plan.scan_all:
        logger.info("page %d: scan → layout+easyocr+tableformer", pno + 1)
        return _complex_page(doc, pno, has_text_layer=False,
                             easyocr_langs=easyocr_langs, render_dpi=render_dpi)
    assignments = {a.page: a.source for a in plan.assignments}
    source = assignments.get(pno, "text_parser")
    if source == "text_parser":
        # Страница без таблиц: быстрый PyMuPDF, без EasyOCR/TableFormer.
        logger.info("page %d: text-only → pymupdf", pno + 1)
        text = PdfTextParser.extract_from_doc(doc, pages=[pno])
        md = text.get(pno, "").strip()
        return md, "text-only→pymupdf"
    if source == "docling":  # страница с таблицами
        logger.info(
            "page %d: table → layout+easyocr+tableformer", pno + 1
        )
        return _complex_page(doc, pno, has_text_layer=True,
                             easyocr_langs=easyocr_langs, render_dpi=render_dpi)
    # docling_ocr — страница-картинка внутри текстового PDF.
    logger.info("page %d: scan → layout+easyocr+tableformer", pno + 1)
    return _complex_page(doc, pno, has_text_layer=False,
                         easyocr_langs=easyocr_langs, render_dpi=render_dpi)


def _complex_page(
    doc, pno: int, *, has_text_layer: bool,
    easyocr_langs: Tuple[str, ...], render_dpi: int,
) -> Tuple[str, str]:
    """Обработка страницы через layout+OCR+TableFormer c fallback'ами."""
    try:
        md, route = process_complex_page(
            doc, pno, dpi=render_dpi, langs=easyocr_langs,
            has_text_layer=has_text_layer,
        )
        if md.strip():
            return md, route
        logger.warning("page %d: complex processing produced no text", pno + 1)
    except Exception as exc:  # noqa: BLE001 - страница не должна теряться
        logger.warning(
            "page %d: complex processing failed: %s", pno + 1, exc
        )
    # Последний fallback: текстовый слой (если он вообще есть).
    if has_text_layer:
        try:
            text = PdfTextParser.extract_from_doc(doc, pages=[pno]).get(pno, "")
            if text.strip():
                return text.strip(), "fallback→pymupdf"
        except Exception as exc:  # noqa: BLE001
            logger.warning("page %d: pymupdf fallback failed: %s", pno + 1, exc)
    return "", "empty"


def parse_pdf(path: str | Path) -> str:
    """Единый интерфейс ТЗ: ``parse_pdf(path) -> markdown_str``.

    Опциональный ``parse_pdf(path, config=...)`` переходит в
    ``run_pipeline`` с параметрами из ``core.config.settings``.
    """
    from rag_parser.core.config import settings

    result = run_pipeline(
        Path(path),
        min_chars_per_page=settings.PDF_TEXT_MIN_CHARS_PER_PAGE,
        text_pages_min_ratio=settings.PDF_TEXT_PAGES_MIN_RATIO,
        min_area_ratio=settings.PDF_TEXT_MIN_AREA_RATIO,
        min_valid_char_ratio=settings.PDF_TEXT_MIN_VALID_CHAR_RATIO,
        easyocr_langs=tuple(
            lang for lang in settings.EASYOCR_LANGS.split(",") if lang
        ) or ("ru", "en"),
        render_dpi=settings.PDF_RENDER_DPI,
    )
    return result.markdown


__all__ = ["PdfPipelineResult", "run_pipeline", "parse_pdf"]
