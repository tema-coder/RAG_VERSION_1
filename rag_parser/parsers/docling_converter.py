"""Код №2: вызов Docling (PDF → Markdown).

Промпт ``promt_for_parse_pdf.txt``: «Код №2: вызов Docling (PDF → Markdown)».
В репозитории готового кода не было — написан с нуля.

Два профиля (промпт, «Ключевые решения»):
1. **С OCR** — для сканов и страниц без текстового слоя. Языки ru+en,
   движок настраиваемый (``DOCLING_OCR_ENGINE``).
2. **Без OCR** — для табличных страниц текстового PDF.
   ``do_table_structure=True``, OCR выключен.

``DocumentConverter`` создаётся лениво и один раз на процесс (singleton):
загрузка моделей тяжёлая. В юнит-тестах Docling не загружается —
``convert`` мокается через ``monkeypatch``.

Отказоустойчивость: если Docling не установлен или упал — ``convert``
кидает ``DoclingError`` с понятным ``X-Error-Type``, вызывающий решает,
что делать (fallback на Код №1 или 422 для скана).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class DoclingError(Exception):
    """Ошибка Docling с понятным типом для API."""

    def __init__(self, message: str, *, error_type: str = "docling_error") -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


# --------------------------------------------------------------------------- #
# Singleton-конвертеры (два профиля)
# --------------------------------------------------------------------------- #

_ocr_converter: Any = None
_no_ocr_converter: Any = None
_docling_available: Optional[bool] = None


def is_docling_available() -> bool:
    """Проверяет, установлен ли Docling (без загрузки моделей)."""
    global _docling_available
    if _docling_available is None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            _docling_available = True
        except ImportError:
            _docling_available = False
    return _docling_available


def _build_converter(*, with_ocr: bool, ocr_engine: str = "easyocr",
                     ocr_langs: Sequence[str] = ("ru", "en")) -> Any:
    """Создаёт ``DocumentConverter`` с нужным профилем.

    Lazy import — модели загружаются только при первом реальном вызове.
    В юнит-тестах эта функция не вызывается (Docling мокается).

    API Docling (v2.50+): ``DocumentConverter(format_options={InputFormat.PDF:
    PdfFormatOption(pipeline_options=PdfPipelineOptions(...))})``.

    ``ocr_options`` — конкретный dataclass (``EasyOcrOptions`` /
    ``TesseractOcrOptions`` / …), НЕ dict. У каждого свой ``kind``
    (ClassVar), по нему Docling находит класс движка. Передача dict без
    ``kind`` → ``No class found with the name ''``.

    ``force_full_page_ocr`` deprecated — вместо него ``OcrMode.FULL_PAGE``
    в ``ocr_options.mode``.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMode,
        PdfPipelineOptions,
        TableStructureOptions,
    )

    if with_ocr:
        ocr_options = _make_ocr_options(
            ocr_engine=ocr_engine,
            ocr_langs=list(ocr_langs),
            full_page=True,
        )
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=ocr_options,
        )
    else:
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(do_cell_matching=True),
        )

    pdf_option = PdfFormatOption(pipeline_options=pipeline_options)
    return DocumentConverter(
        format_options={InputFormat.PDF: pdf_option},
    )


def _make_ocr_options(*, ocr_engine: str, ocr_langs: list, full_page: bool):
    """Создаёт dataclass OCR-опций для выбранного движка.

    Docling требует конкретный подкласс ``OcrOptions`` (с ``kind``) — dict
    не принимается. Поддерживаемые движки: ``easyocr``, ``tesseract``,
    ``tesserocr``, ``rapidocr``. Если движок недоступен — fallback на
    ``easyocr`` (ставится через ``pip install docling[easyocr]``).
    """
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMode,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
    )
    mode = OcrMode.FULL_PAGE if full_page else None
    engine = (ocr_engine or "easyocr").lower()

    if engine in ("easyocr", "easy_ocr"):
        return EasyOcrOptions(lang=ocr_langs or ["en"], use_gpu=False,
                              mode=mode)
    if engine in ("tesseract",):
        return TesseractCliOcrOptions(lang=ocr_langs or ["eng"], mode=mode)
    if engine in ("tesserocr",):
        return TesseractOcrOptions(lang=ocr_langs or ["eng"], mode=mode)
    # rapidocr — опционально, импортируем лениво (не во всех версиях).
    try:
        from docling.datamodel.pipeline_options import RapidOcrOptions
        if engine == "rapidocr":
            return RapidOcrOptions(lang=ocr_langs or ["en"], mode=mode)
    except ImportError:
        pass
    # Fallback — easyocr (самый распространённый, ставится через docling[easyocr]).
    return EasyOcrOptions(lang=ocr_langs or ["en"], use_gpu=False, mode=mode)


def get_ocr_converter(ocr_engine: str = "easyocr",
                      ocr_langs: Sequence[str] = ("ru", "en")) -> Any:
    """Singleton OCR-конвертер (создаётся при первом вызове)."""
    global _ocr_converter
    if _ocr_converter is None:
        logger.info("docling_ocr_converter_init", extra={
            "ocr_engine": ocr_engine, "ocr_langs": list(ocr_langs)})
        _ocr_converter = _build_converter(
            with_ocr=True, ocr_engine=ocr_engine, ocr_langs=ocr_langs)
    return _ocr_converter


def get_no_ocr_converter() -> Any:
    """Singleton no-OCR конвертер (для табличных страниц текстового PDF)."""
    global _no_ocr_converter
    if _no_ocr_converter is None:
        logger.info("docling_no_ocr_converter_init")
        _no_ocr_converter = _build_converter(with_ocr=False)
    return _no_ocr_converter


# --------------------------------------------------------------------------- #
# Очистка артефактов Docling
# --------------------------------------------------------------------------- #

import re

# Артефакты Docling в Markdown, которые нужно убрать.
_DOCLING_IMAGE_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_DOCLING_PAGEBREAK_RE = re.compile(r"<!--\s*page-break\s*-->", re.IGNORECASE)
_DOCLING_MISC_RE = re.compile(r"<!--\s*(?:figure|table|caption)\s*-->", re.IGNORECASE)
# Лишние пустые строки (≥3 → 2).
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_docling_markdown(md: str) -> str:
    """Убирает артефакты Docling (``<!-- image -->``, лишние переносы)."""
    md = _DOCLING_IMAGE_RE.sub("", md)
    md = _DOCLING_PAGEBREAK_RE.sub("", md)
    md = _DOCLING_MISC_RE.sub("", md)
    md = _MULTI_NEWLINE_RE.sub("\n\n", md)
    return md.strip()


# --------------------------------------------------------------------------- #
# Основной API
# --------------------------------------------------------------------------- #

@dataclass
class DoclingResult:
    """Результат вызова Docling на диапазоне страниц."""

    markdown: str
    page_range: tuple  # (start, end) — 0-based, inclusive
    source: str  # "docling" | "docling_ocr"


def convert(
    path: Path,
    *,
    pages: Optional[Sequence[int]] = None,
    with_ocr: bool = False,
    ocr_engine: str = "easyocr",
    ocr_langs: Sequence[str] = ("ru", "en"),
) -> DoclingResult:
    """Конвертирует PDF (или диапазон страниц) в Markdown через Docling.

    :param path: путь к PDF.
    :param pages: номера страниц (0-based). ``None`` — весь файл.
        Если передан, создаётся временный sub-PDF с этими страницами
        (маппинг номеров сохраняется).
    :param with_ocr: профиль — ``True`` для сканов/страниц без слоя,
        ``False`` для табличных страниц текстового PDF.
    :return: ``DoclingResult`` с Markdown и диапазоном страниц.
    :raises DoclingError: если Docling не установлен или упал.
    """
    if not is_docling_available():
        raise DoclingError(
            "Docling is not installed. Install with: pip install docling",
            error_type="docling_not_installed",
        )

    try:
        if with_ocr:
            converter = get_ocr_converter(ocr_engine, ocr_langs)
        else:
            converter = get_no_ocr_converter()

        # Docling поддерживает page_range напрямую (0-based, inclusive).
        # Если передан диапазон страниц — используем его, без sub-PDF.
        if pages is not None and len(pages) > 0:
            sorted_pages = sorted(set(pages))
            page_range_tuple = (sorted_pages[0], sorted_pages[-1])
            conv_doc = converter.convert(
                str(path), page_range=page_range_tuple
            )
            md = conv_doc.document.export_to_markdown()
            page_range = page_range_tuple
        else:
            conv_doc = converter.convert(str(path))
            md = conv_doc.document.export_to_markdown()
            # Весь файл — диапазон (0, total-1).
            import fitz
            doc = fitz.open(str(path))
            page_range = (0, len(doc) - 1)
            doc.close()

    except DoclingError:
        raise
    except Exception as exc:
        profile = "ocr" if with_ocr else "no_ocr"
        raise DoclingError(
            f"Docling failed ({profile}) on {path.name}: {exc}",
            error_type=f"docling_{profile}_failure",
        ) from exc

    md = _clean_docling_markdown(md)
    source = "docling_ocr" if with_ocr else "docling"
    return DoclingResult(markdown=md, page_range=page_range, source=source)


def _create_sub_pdf(path: Path, pages: Sequence[int]) -> Path:
    """Создаёт временный sub-PDF с указанными страницами (PyMuPDF)."""
    import fitz
    import tempfile

    doc = fitz.open(str(path))
    try:
        new_doc = fitz.open()
        sorted_pages = sorted(set(pages))
        for pno in sorted_pages:
            if 0 <= pno < len(doc):
                new_doc.insert_pdf(doc, from_page=pno, to_page=pno)
        # Временный файл.
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        handle.close()
        new_doc.save(handle.name)
        new_doc.close()
        return Path(handle.name)
    finally:
        doc.close()


def reset_singletons() -> None:
    """Сбрасывает singleton-конвертеры (для тестов)."""
    global _ocr_converter, _no_ocr_converter, _docling_available
    _ocr_converter = None
    _no_ocr_converter = None
    _docling_available = None


__all__ = [
    "DoclingError",
    "DoclingResult",
    "convert",
    "is_docling_available",
    "reset_singletons",
]
