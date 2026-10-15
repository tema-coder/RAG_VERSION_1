"""Конвертация PDF → DOCX через ``pdf2docx`` (по флагу ``PDF_VIA_DOCX``).

Зачем
----
Иногда таблицы в PDF видны глазом, но не извлекаются нативным
``PdfReader``. Классический обходной путь — сконвертировать PDF в DOCX и
разобрать его как обычный вордовский документ.

Почему это **выключено по умолчанию**
------------------------------------
Замер на ``input/gost_report.pdf`` (``scripts/debug_pdf_via_docx.py``)::

    текстовых блоков      66 -> 24   # потеряно ~64% текста
    таблиц                 9 -> 10   # появилась лишняя
    привязка таблиц       порядок нарушен (таблица 6.2.1 -> раздел 1)

Причина в конвертере, а не в dedoc: ``pdf2docx`` восстанавливает таблицы
приблизительно, раскладывая их в floating-текстовые блоки по эвристическим
координатам. Порядок чтения самого DOCX dedoc соблюдает корректно — это
видно по тому, что исходный ``gost_report.docx`` разбирается без единого
смещения. Поэтому позиционная целостность таблиц при конвертации PDF→DOCX
**не гарантируется**, и маршрут включается только явным флагом.

Использование
-------------
Включить в ``.env``::

    PDF_VIA_DOCX=true
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["convert_pdf_to_docx", "pdf2docx_available"]


def pdf2docx_available() -> bool:
    """Установлен ли ``pdf2docx``."""
    try:
        import pdf2docx
    except ImportError:
        return False
    return bool(pdf2docx)


def convert_pdf_to_docx(pdf_path: Path, target_dir: Path) -> Optional[Path]:
    """Конвертирует PDF в DOCX. Возвращает ``None``, если не вышло.

    Вызывающий обязан удалить результат — он лежит во временной папке.
    """
    if not pdf2docx_available():
        logger.warning(
            "pdf_via_docx_unavailable",
            extra={"path": str(pdf_path), "hint": "pip install pdf2docx==0.5.13"},
        )
        return None

    import pdf2docx

    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{pdf_path.stem}_via_pdf2docx.docx"

    # Конструктор pdf2docx сам открывает файл и бросает FileNotFoundError
    # на отсутствующем/битом PDF — поэтому он тоже внутри try.
    try:
        converter = pdf2docx.Converter(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - pdf2docx бросает разные ошибки
        logger.warning(
            "pdf_via_docx_open_failed",
            extra={"path": str(pdf_path), "error": str(exc)},
        )
        return None

    try:
        converter.convert(str(out_path))
    except Exception as exc:  # noqa: BLE001 - pdf2docx бросает разные ошибки
        logger.warning(
            "pdf_via_docx_failed",
            extra={"path": str(pdf_path), "error": str(exc)},
        )
        return None
    finally:
        try:
            converter.close()
        except Exception:  # noqa: BLE001,S110 - закрытие не критично
            pass

    if not out_path.exists() or out_path.stat().st_size == 0:
        logger.warning("pdf_via_docx_empty", extra={"path": str(pdf_path)})
        return None

    logger.info("pdf_via_docx_converted", extra={"path": str(out_path)})
    return out_path