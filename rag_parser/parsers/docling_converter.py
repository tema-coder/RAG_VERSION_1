"""Единственный адаптер Docling: PDF и изображения → Markdown.

Логика повторяет эталонный пример из задачи:

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()

Отличия, продиктованные архитектурой проекта:

* ``DocumentConverter`` создаётся лениво и один раз на процесс (singleton):
  загрузка layout/table/OCR-моделей тяжёлая;
* включены встроенные в PDF-пайплайн Docling OCR (EasyOCR/Tesseract — по
  настройке) и распознавание структуры таблиц;
* изображения (``.png``, ``.jpg``, ``.jpeg``) сначала упаковываются в
  временный PDF через ``img2pdf`` (только техническая упаковка, не OCR);
* временный PDF создаётся под уникальным именем во временном каталоге
  проекта и удаляется даже при исключении;
* любой сбой Docling превращается в ``DoclingError``. Никакого перехода на
  PyMuPDF/EasyOCR-вне-Docling/pytesseract и т.п. нет.

Юнит-тесты не загружают модели: они подменяют ``get_converter`` или
``convert_to_markdown``.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

#: Расширения изображений, которые упаковываются в PDF перед Docling.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

#: Поддерживаемые расширения входа этого модуля.
SUPPORTED_EXTENSIONS = frozenset({".pdf", *IMAGE_EXTENSIONS})


class DoclingError(Exception):
    """Ошибка Docling с типом, пригодным для API/логов."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "docling_error",
        path: Optional[Path] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.path = path


# --------------------------------------------------------------------------- #
# Singleton-конвертер
# --------------------------------------------------------------------------- #

_converter: Any = None
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


def _tesseract_langs(langs: Sequence[str]) -> list:
    """ISO 639-1 → ISO 639-2 для Tesseract (ru→rus, en→eng)."""
    mapping = {"ru": "rus", "en": "eng", "de": "deu", "fr": "fra",
               "es": "spa", "it": "ita"}
    return [mapping.get(lang.lower(), lang) for lang in langs]


def _make_ocr_options(ocr_engine: str, ocr_langs: Sequence[str]) -> Any:
    """Создаёт dataclass OCR-опций, совместимый с установленным Docling.

    Docling принимает конкретный подкласс ``OcrOptions`` (у него есть
    ``kind``), а не dict. Поддерживаем EasyOCR (по умолчанию), Tesseract и
    RapidOCR; всё это — движки ВНУТРИ пайплайна Docling.
    """
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMode,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
    )

    engine = (ocr_engine or "easyocr").lower()
    langs = list(ocr_langs) or ["ru", "en"]
    # DEFAULT — OCR только там, где нужен (сканы/картинки без текстового
    # слоя). Не FULL_PAGE: иначе OCR прогоняется и по нормальному тексту.
    mode = OcrMode.DEFAULT

    if engine in ("easyocr", "easy_ocr"):
        return EasyOcrOptions(lang=langs, use_gpu=False, mode=mode)
    if engine in ("tesseract", "tesseract_cli"):
        return TesseractCliOcrOptions(
            lang=_tesseract_langs(langs), mode=mode
        )
    if engine in ("tesserocr",):
        return TesseractOcrOptions(lang=_tesseract_langs(langs), mode=mode)
    if engine in ("rapidocr", "rapid_ocr"):
        from docling.datamodel.pipeline_options import RapidOcrOptions

        return RapidOcrOptions(lang=langs, mode=mode)
    logger.warning(
        "unknown DOCLING_OCR_ENGINE=%r, falling back to easyocr", ocr_engine
    )
    return EasyOcrOptions(lang=langs, use_gpu=False, mode=mode)


def _build_converter() -> Any:
    """Создаёт ``DocumentConverter`` с OCR и структурой таблиц."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from rag_parser.core.config import settings

    ocr_langs = [
        lang.strip()
        for lang in settings.DOCLING_OCR_LANGS.split(",")
        if lang.strip()
    ]
    pipeline_options = PdfPipelineOptions(
        do_ocr=settings.DOCLING_DO_OCR,
        do_table_structure=settings.DOCLING_DO_TABLE_STRUCTURE,
        ocr_options=_make_ocr_options(settings.DOCLING_OCR_ENGINE, ocr_langs),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def get_converter() -> Any:
    """Ленивый singleton ``DocumentConverter``."""
    global _converter
    if _converter is None:
        if not is_docling_available():
            raise DoclingError(
                "Docling is not installed. Install with: pip install docling",
                error_type="docling_not_installed",
            )
        logger.info("docling_converter_init")
        _converter = _build_converter()
    return _converter


def reset_singletons() -> None:
    """Сбрасывает singleton-конвертер (для тестов)."""
    global _converter, _docling_available
    _converter = None
    _docling_available = None


# --------------------------------------------------------------------------- #
# Очистка артефактов Docling
# --------------------------------------------------------------------------- #

_DOCLING_IMAGE_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_DOCLING_PAGEBREAK_RE = re.compile(r"<!--\s*page-break\s*-->", re.IGNORECASE)
_DOCLING_MISC_RE = re.compile(
    r"<!--\s*(?:figure|table|caption|formula-not-decoded)\s*-->", re.IGNORECASE
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_docling_markdown(md: str) -> str:
    """Убирает служебные HTML-комментарии Docling и лишние пустые строки."""
    md = _DOCLING_IMAGE_RE.sub("", md)
    md = _DOCLING_PAGEBREAK_RE.sub("", md)
    md = _DOCLING_MISC_RE.sub("", md)
    md = _MULTI_NEWLINE_RE.sub("\n\n", md)
    return md.strip()


# --------------------------------------------------------------------------- #
# Упаковка изображений в PDF
# --------------------------------------------------------------------------- #


def _unique_temp_pdf_path(source: Path) -> Path:
    """Уникальный путь временного PDF (без перезаписи чужих файлов)."""
    from rag_parser.core.config import settings

    temp_dir = settings.ensure_temp_dir()
    return temp_dir / f"docling_{uuid.uuid4().hex}_{source.stem}.pdf"


def _normalize_image_for_pdf(source: Path) -> Optional[Path]:
    """Готовит RGB-копию изображения, если img2pdf не умеет исходник.

    Это НЕ извлечение содержимого и не OCR — только приведение картинки
    (например, PNG с альфа-каналом) к формату, который упаковывается в PDF.
    Возвращает путь к временному изображению или ``None``.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow есть в зависимостях docling
        return None
    try:
        with Image.open(source) as image:
            if image.mode in ("RGB", "L"):
                return None
            rgb = image.convert("RGB")
            target = _unique_temp_pdf_path(source).with_suffix(".png")
            rgb.save(target, format="PNG")
            return target
    except Exception as exc:  # noqa: BLE001 - упаковка не должна падать молча
        logger.warning("image_normalize_failed path=%s err=%s", source, exc)
        return None


def pack_image_to_pdf(source: Path) -> Path:
    """Упаковывает изображение в уникальный временный PDF через img2pdf.

    :raises DoclingError: если img2pdf отсутствует или не смог упаковать файл.
    """
    try:
        import img2pdf
    except ImportError as exc:
        raise DoclingError(
            "img2pdf is not installed. Install with: pip install img2pdf",
            error_type="img2pdf_not_installed",
            path=source,
        ) from exc

    target = _unique_temp_pdf_path(source)
    normalized: Optional[Path] = None
    try:
        try:
            payload = img2pdf.convert(str(source))
        except Exception as first_exc:  # noqa: BLE001 - альфа-канал и т.п.
            normalized = _normalize_image_for_pdf(source)
            if normalized is None:
                raise DoclingError(
                    f"Failed to pack image into PDF ({source}): {first_exc}",
                    error_type="image_to_pdf_failure",
                    path=source,
                ) from first_exc
            try:
                payload = img2pdf.convert(str(normalized))
            except Exception as exc:  # noqa: BLE001
                raise DoclingError(
                    f"Failed to pack image into PDF ({source}): {exc}",
                    error_type="image_to_pdf_failure",
                    path=source,
                ) from exc
        target.write_bytes(payload)
    except DoclingError:
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise DoclingError(
            f"Failed to write temporary PDF for {source}: {exc}",
            error_type="image_temp_pdf_write_failure",
            path=source,
        ) from exc
    finally:
        if normalized is not None:
            normalized.unlink(missing_ok=True)

    return target


# --------------------------------------------------------------------------- #
# Основной API
# --------------------------------------------------------------------------- #


@dataclass
class DoclingResult:
    """Результат конвертации документа Docling."""

    markdown: str
    page_count: int = 0
    source: str = "pdf"  # "pdf" | "image"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _run_docling(pdf_path: Path, *, source_path: Path, source: str) -> DoclingResult:
    """Конвертирует готовый PDF через Docling и экспортирует Markdown."""
    converter = get_converter()
    started = time.perf_counter()
    try:
        result = converter.convert(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - любые ошибки Docling
        raise DoclingError(
            f"Docling failed on {source_path}: {exc}",
            error_type="docling_conversion_failure",
            path=source_path,
        ) from exc

    from docling.datamodel.document import ConversionStatus

    status = getattr(result, "status", None)
    errors = getattr(result, "errors", None) or []
    document = getattr(result, "document", None)
    if status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        detail = "; ".join(str(error) for error in errors[:5]) or str(status)
        raise DoclingError(
            f"Docling could not convert {source_path}: {detail}",
            error_type="docling_conversion_failure",
            path=source_path,
        )
    if document is None:
        raise DoclingError(
            f"Docling returned no document for {source_path}",
            error_type="docling_empty_result",
            path=source_path,
        )
    if status == ConversionStatus.PARTIAL_SUCCESS:
        logger.warning(
            "docling_partial_success path=%s errors=%d",
            source_path,
            len(errors),
        )

    try:
        markdown = document.export_to_markdown()
    except Exception as exc:  # noqa: BLE001
        raise DoclingError(
            f"Docling failed to export Markdown for {source_path}: {exc}",
            error_type="docling_export_failure",
            path=source_path,
        ) from exc

    markdown = _clean_docling_markdown(markdown or "")
    if not markdown.strip():
        raise DoclingError(
            f"Docling produced no text for {source_path}",
            error_type="docling_empty_markdown",
            path=source_path,
        )

    pages = getattr(result, "pages", None) or {}
    try:
        page_count = len(pages)
    except TypeError:  # pragma: no cover - защита от неожиданного типа
        page_count = 0

    return DoclingResult(
        markdown=markdown,
        page_count=page_count,
        source=source,
        metadata={
            "docling_status": getattr(status, "value", str(status)),
            "docling_errors": [str(error) for error in errors],
            "docling_seconds": round(time.perf_counter() - started, 3),
            "page_count": page_count,
        },
    )


def _convert_to_markdown_impl(path: Path) -> DoclingResult:
    """Конвертирует PDF или изображение в Markdown через Docling.

    :raises DoclingError: файл не существует/пуст/не поддержан, Docling или
        img2pdf упали, Markdown пуст.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise DoclingError(
            f"File not found: {source_path}",
            error_type="file_not_found",
            path=source_path,
        )
    try:
        size = source_path.stat().st_size
    except OSError as exc:
        raise DoclingError(
            f"Cannot read file {source_path}: {exc}",
            error_type="file_read_failure",
            path=source_path,
        ) from exc
    if size == 0:
        raise DoclingError(
            f"File is empty: {source_path}",
            error_type="empty_file",
            path=source_path,
        )

    extension = source_path.suffix.lower()
    if extension == ".pdf":
        return _run_docling(source_path, source_path=source_path, source="pdf")
    if extension in IMAGE_EXTENSIONS:
        temp_pdf = pack_image_to_pdf(source_path)
        try:
            result = _run_docling(
                temp_pdf, source_path=source_path, source="image"
            )
        finally:
            temp_pdf.unlink(missing_ok=True)
        result.metadata["source_extension"] = extension
        return result

    raise DoclingError(
        f"Unsupported format for Docling: {extension!r} ({source_path})",
        error_type="unsupported_format",
        path=source_path,
    )


def convert_to_markdown(path: Path) -> DoclingResult:
    """Публичный вход адаптера: логирует путь и причину любой ошибки.

    ``DoclingError`` не подавляется и не заменяется результатом другого
    парсера.
    """
    try:
        return _convert_to_markdown_impl(Path(path))
    except DoclingError as exc:
        logger.error(
            "docling_failed path=%s error_type=%s reason=%s",
            path,
            exc.error_type,
            exc.message,
        )
        raise


__all__ = [
    "DoclingError",
    "DoclingResult",
    "IMAGE_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "convert_to_markdown",
    "get_converter",
    "is_docling_available",
    "pack_image_to_pdf",
    "reset_singletons",
]
