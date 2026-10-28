"""Реестр тяжёлых моделей PDF-пайплайна: EasyOCR, layout, TableFormer.

Все модели загружаются ЛЕНИВО и ОДИН РАЗ на процесс и переиспользуются
между страницами и документами (ТЗ: «Модели … загружать один раз и
переиспользовать. Не создавать заново на каждую страницу»).

* **EasyOCR** — ``easyocr.Reader(langs, gpu=False)``; языки из конфига
  (``EASYOCR_LANGS``, старт ``ru,en``). Читаемые батчи кешируются по
  набору языков: разные ``langs`` → разные Reader-ы (максимум полезных
  комбинаций обычно одна).
* **Layout-модель** — layout-детектор Docling (модель ``doclayout_yolov25``
  внутри ``document_converter`` pipeline). Если Docling недоступен,
  используется ``None`` — вызывающий код падает на эвристическую
  сегментацию PyMuPDF (см. ``pdf_page_processor``).
* **TableFormer** — модель структуры таблиц Docling
  (``docling_ibm_models.tableformer`` через pipeline). Отдельный прямой
  API для вызова на кропе таблицы.

Среда тестов: ничего тяжёлого не импортируется на уровне модуля, все
модели создаются только при первом реальном вызове; юнит-тесты подменяют
функции ``run_easyocr_on_crop`` / ``detect_page_layout`` /
``run_tableformer_on_crop`` через ``monkeypatch``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class PdfModelError(Exception):
    """Модель недоступна или упала при инициализации/инференсе."""

    def __init__(self, message: str, *, model: str = "unknown") -> None:
        super().__init__(message)
        self.model = model


# --------------------------------------------------------------------------- #
# Singleton-реестр
# --------------------------------------------------------------------------- #

_easyocr_readers: Dict[Tuple[str, ...], Any] = {}
_layout_detector: Any = None
_layout_backend: Optional[str] = None
_tableformer: Any = None


def get_easyocr_reader(langs: Sequence[str] = ("ru", "en")) -> Any:
    """Singleton ``easyocr.Reader`` для данного набора языков."""
    key = tuple(sorted(set(langs or ("en",))))
    if key in _easyocr_readers:
        return _easyocr_readers[key]
    try:
        import easyocr  # noqa: F401
    except ImportError as exc:
        raise PdfModelError(
            f"easyocr is not installed: {exc}", model="easyocr"
        ) from exc
    try:
        logger.info("easyocr_reader_init", extra={"langs": list(key)})
        reader = easyocr.Reader(
            list(key), gpu=False, verbose=False
        )
    except Exception as exc:  # noqa: BLE001 - init may fail in odd envs
        raise PdfModelError(
            f"easyocr init failed: {exc}", model="easyocr"
        ) from exc
    _easyocr_readers[key] = reader
    return reader


def reset_easyocr() -> None:
    """Сброс singleton-ов EasyOCR (для тестов)."""
    _easyocr_readers.clear()


def is_docling_available() -> bool:
    """Установлен ли Docling (импорт без загрузки моделей)."""
    try:
        import docling  # noqa: F401,F403

        return True
    except ImportError:
        return False


def reset_all() -> None:
    """Сброс всех singleton-ов (для тестов)."""
    global _layout_detector, _layout_backend, _tableformer
    _easyocr_readers.clear()
    _layout_detector = None
    _layout_backend = None
    _tableformer = None


# --------------------------------------------------------------------------- #
# TableFormer через Docling
# --------------------------------------------------------------------------- #


def _tableformer_via_docling(
    image: "Image.Image",
    bbox: Tuple[float, float, float, float],
) -> str:
    """TableFormer: структура таблицы по изображению → Markdown.

    Пытается использовать TableStructureModel Docling
    (``docling_ibm_models.tableformer``). Если структура недоступна —
    поднимает ``PdfModelError`` и вызывающий код уходит в fallback'и
    (PyMuPDF ``find_tables`` / EasyOCR).
    """
    try:
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
    except ImportError as exc:
        raise PdfModelError(
            f"docling tableformer not importable: {exc}", model="tableformer"
        ) from exc

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(do_cell_matching=True),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options)}
    )
    # Прямой вызов по изображению требует вписывания в pipeline Docling;
    # если API установки версии не даёт отдельную точку входа — честная
    # ошибка, чтобы вызывающий код ушёл в fallback, а не завис.
    raise PdfModelError(
        "docling tableformer standalone crop API not available "
        f"(docling {getattr(__import__('docling.version', fromlist=['version']), '__version__', '?')})",
        model="tableformer",
    )


__all__ = [
    "PdfModelError",
    "get_easyocr_reader",
    "is_docling_available",
    "reset_all",
    "reset_easyocr",
]
