"""Защита от падения dedoc на bbox, выходящем за границы рендера страницы.

Симптом
-------
``PdfminerExtractor`` пересчитывает координаты текста из единиц PDF в пиксели
рендера через ``create_bbox(height, ...)``, где ``height``/``width`` берутся из
``mediabox``. На части реальных PDF (landscape-страницы, страницы с
обрезкой/поворотом) координаты текста оказываются **за пределами**
изображения. Тогда в ``LineMetadataExtractor.__get_color_annotation`` срез

    image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]

даёт пустой массив, и ``cv2.inRange`` падает с
``error: (-215:Assertion failed) !_src.empty()`` — исключение уходит наверх и
весь разбор PDF завершается ``DocumentParseError``, хотя текстовый слой в
документе есть.

Примеры из ``input/``: ``пдф_таблица.pdf`` (A4, но строки таблицы вынесены за
правый край), ``hard_pdf.pdf`` (mediabox 963x538 — альбомная страница).

Решение
-------
Мы не чиним координаты «на глаз»: bbox **обрезается** по границам
изображения, а срез, который после обрезки остался пустым, получает
нейтральный цвет (0, 0, 0) и пустую аннотацию. Цвет фона — исключительно
косметическая аннотация ``ColorAnnotation``, она не участвует в
структуре документа, чанках и Markdown, поэтому потеря ничего не значит
для RAG-пайплайна.

Почему патчится приватный метод
-------------------------------
Это баг самого dedoc 2.8. Обходится точечно и только на чтение: подменяется
один метод ``__get_color_annotation``; всё остальное поведение ридера
(текст, структура, таблицы) не меняется. Альтернативных публичных хуков у
dedoc для этого нет.

Патч применяется **только в основном потоке парсинга** (``DedocParser.parse``
через ``get_manager``) и только для PDF. Он идемпотентен: повторный вызов
не ставит патч дважды.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

_PATCH_FLAG = "_rag_parser_bbox_guard_installed"


@dataclass
class _ClampedTextWithBBox:
    """Минимальный аналог ``TextWithBBox`` с обрезанным bbox.

    Подменяется в аргумент оригинального метода, который читает только
    ``.bbox`` и ``.text``.
    """

    bbox: Any
    text: str
    annotations: List[Any] = field(default_factory=list)


def _patched_color_annotation(self: Any, bbox_with_text: Any, image: Any) -> Any:
    """Обрезает bbox по границам ``image`` до вычисления цвета текста."""
    from dedocutils.data_structures.bbox import BBox
    from dedoc.data_structures.concrete_annotations.color_annotation import (
        ColorAnnotation,
    )

    box = bbox_with_text.bbox
    height, width = int(image.shape[0]), int(image.shape[1])

    y_top = max(0, box.y_top_left)
    y_bottom = min(height, box.y_bottom_right)
    x_left = max(0, box.x_top_left)
    x_right = min(width, box.x_bottom_right)

    original = getattr(self, "_rag_parser_original_color_annotation", None)
    if original is None:  # страховка: патч не установлен — не рискуем падать
        raise RuntimeError("pdf_bbox_guard: original method is not available")

    if y_bottom <= y_top or x_right <= x_left:
        # Срез пуст даже после обрезки: цвета нет, отдаём нейтральный.
        return ColorAnnotation(
            start=0, end=len(bbox_with_text.text), red=0, green=0, blue=0
        )

    if (
        y_top == box.y_top_left
        and y_bottom == box.y_bottom_right
        and x_left == box.x_top_left
        and x_right == box.x_bottom_right
    ):
        return original(self, bbox_with_text, image)

    clamped = _ClampedTextWithBBox(
        bbox=BBox(x_left, y_top, x_right - x_left, y_bottom - y_top),
        text=bbox_with_text.text,
    )
    return original(self, clamped, image)


def install_bbox_guard() -> bool:
    """Устанавливает защиту bbox. Возвращает ``True``, если патч применён.

    Идемпотентна: повторный вызов ничего не делает и возвращает ``True``.
    """
    try:
        from dedoc.readers.pdf_reader.pdf_image_reader.line_metadata_extractor import (
            metadata_extractor,
        )
    except ImportError:
        return False

    extractor = metadata_extractor.LineMetadataExtractor
    method_name = f"_{extractor.__name__}__get_color_annotation"
    if getattr(extractor, _PATCH_FLAG, False):
        return True

    original = getattr(extractor, method_name, None)
    if original is None:  # структура dedoc изменилась — не ломаем разбор
        return False

    extractor._rag_parser_original_color_annotation = staticmethod(original)
    setattr(extractor, method_name, _patched_color_annotation)
    setattr(extractor, _PATCH_FLAG, True)
    return True


__all__ = ["install_bbox_guard"]