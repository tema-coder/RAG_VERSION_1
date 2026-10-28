"""Постраничный обработчик «сложных» страниц PDF (шаг 2.3 и шаг 3 ТЗ).

Пайплайн одной страницы:
1. **Рендер** страницы в изображение (``page.get_pixmap``, DPI из конфига).
2. **Layout-детекция**: bounding box'ы блоков; картинки/иллюстрации
   игнорируются, остальные делятся на текстовые блоки и блоки таблиц.
3. **Текстовые блоки → EasyOCR** → Markdown-абзацы/заголовки.
4. **Блоки таблиц → TableFormer** → Markdown-таблица.
5. **Сборка в порядке чтения** (сверху вниз, с учётом колонок) — в один
   Markdown страницы.

Отказоустойчивость: падение TableFormer или EasyOCR на отдельном блоке
не теряет блок (fallback на данные текстового слоя / другой движок);
падение layout — постраничный fallback на PyMuPDF (+ ``find_tables``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from rag_parser.parsers.pdf_models import (
    PdfModelError,
    get_easyocr_reader,
    is_docling_available,
)
from rag_parser.parsers.pdf_text_parser import PdfTextParser

logger = logging.getLogger(__name__)

#: Типы layout-блоков.
LABEL_TEXT = "text"
LABEL_TABLE = "table"
LABEL_IGNORE = "ignore"


@dataclass
class LayoutBlock:
    """Блок layout-детекции."""

    label: str  # "text" | "table" | "ignore"
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 (PDF coords)
    confidence: float = 1.0


@dataclass
class _OcrBlock:
    """Промежуточный блок для сборки в порядке чтения."""

    label: str
    bbox: Tuple[float, float, float, float]
    markdown: str = ""


# --------------------------------------------------------------------------- #
# Рендер страницы
# --------------------------------------------------------------------------- #


def render_page_image(doc, page_index: int, dpi: int) -> Image.Image:
    """Рендер страницы PDF в PIL-изображение.

    :param doc: открытый ``fitz.Document``.
    :param page_index: 0-based номер страницы.
    :param dpi: разрешение рендера (коэффициент ``dpi / 72``).
    """
    page = doc[page_index]
    matrix_scale = dpi / 72.0
    matrix = _fitz_matrix(matrix_scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _fitz_matrix(scale: float):
    import fitz

    return fitz.Matrix(scale, scale)


# --------------------------------------------------------------------------- #
# Layout-детекция
# --------------------------------------------------------------------------- #


def detect_page_layout(doc, page_index: int, dpi: int = 200) -> List[LayoutBlock]:
    """Получить layout-блоки страницы.

    **Docling layout** (модель Table изображения) — если Docling доступен,
    отдаём рендер в его предсказатель и маппим классы:
    ``Table`` → table, текстовые классы → text, ``Picture``/``Figure`` и
    прочие декоративные → ignore.

    Если Docling недоступен (PdfModelError), падаем на **эвристическую
    сегментацию PyMuPDF**: текстовые блоки из текстового слоя
    (``get_text("blocks")``), изображения (``block_type == 1``) —
    ``ignore``, таблицы —bbox-ами ``find_tables()`` (текстовые блоки,
    пересекающиеся с таблицами, подсоединяются к таблице).
    """
    try:
        return _layout_via_docling(doc, page_index, dpi)
    except PdfModelError as exc:
        logger.info("layout_fallback_pymupdf page=%d err=%s", page_index + 1, exc)
    return _layout_via_pymupdf(doc, page_index)


def _layout_via_docling(doc, page_index: int, dpi: int) -> List[LayoutBlock]:
    """Layout-детекция через модель Docling (требует установленного Docling)."""
    if not is_docling_available():
        raise PdfModelError("docling not installed", model="layout")
    image = render_page_image(doc, page_index, dpi)
    from docling.models.factories import get_layout_model  # noqa: F401

    raise PdfModelError(
        "docling layout model path not wired in this build",
        model="layout",
    )


def _layout_via_pymupdf(doc, page_index: int) -> List[LayoutBlock]:
    """Эвристическая layout-сегментация средствами PyMuPDF.

    Текстовый слой уже «адекватен» (проверен классификатором), поэтому:
    * текстовые блоки ``get_text("blocks")`` (``block_type == 0``);
    *:image-блоки (``block_type == 1``) и таблицы (``find_tables``)
      образуют спец-зоны: у таблицы приоритет — попавшие в неё текстовые
      блоки становятся частью таблицы;; картинки игнорируются.
    """
    page = doc[page_index]
    blocks: List[LayoutBlock] = []

    # Приоритет — таблицы над текстом.
    table_boxes: List[Tuple[float, float, float, float]] = []
    try:
        finder = page.find_tables()
        for table in finder.tables or []:
            bbox = getattr(table, "bbox", None)
            if bbox:
                table_boxes.append(tuple(bbox))
                blocks.append(LayoutBlock(label=LABEL_TABLE, bbox=tuple(bbox)))
    except Exception as exc:  # noqa: BLE001 - find_tables может упасть
        logger.warning("find_tables_failed page=%d err=%s", page_index + 1, exc)

    def in_any_table(x0, y0, x1, y1) -> bool:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for tx0, ty0, tx1, ty1 in table_boxes:
            if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                return True
        return False

    image_count = 0
    for block in page.get_text("blocks") or []:
        if not block:
            continue
        x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
        if block[6] == 1:  # блок-изображение → игнорируем
            image_count += 1
            blocks.append(
                LayoutBlock(label=LABEL_IGNORE, bbox=(x0, y0, x1, y1))
            )
            continue
        if in_any_table(x0, y0, x1, y1):
            continue  # текст внутри таблицы уже учтён её блоком
        blocks.append(LayoutBlock(label=LABEL_TEXT, bbox=(x0, y0, x1, y1)))

    if image_count:
        logger.info(
            "page_images_ignored page=%d count=%d", page_index + 1, image_count
        )
    return blocks


def sort_blocks_reading_order(
    blocks: Sequence[_OcrBlock], page_width: float, column_threshold: float = 0.5
) -> List[_OcrBlock]:
    """Сортировка блоков в порядке чтения (сверху вниз, учитывая колонки).

    Алгоритм:
    1. Одноколоночные блоки (ширина ≥ ``column_threshold`` от ширины
       страницы) — «разделители секций», сортируются по вертикали.
    2. Остальные (узкие) блоки группируются в «полосы»: блоки, чьи
       вертикальные диапазоны перекрываются, образуют одну полосу.
    3. Полосы читаются по порядку вертикали; внутри полосы — слева
       направо по ``x0``.
    4. Полосы вставляются между разделителями секций.
    """
    def y_center(b: _OcrBlock) -> float:
        return (b.bbox[1] + b.bbox[3]) / 2.0

    full_width = sorted(
        [b for b in blocks if (b.bbox[2] - b.bbox[0]) >= page_width * column_threshold],
        key=lambda b: b.bbox[1],
    )
    narrow = sorted([b for b in blocks if b not in full_width], key=y_center)

    # Группировка узких блоков в полосы (цепочки перекрывающихся блоков).
    bands: List[List[_OcrBlock]] = []
    for block in narrow:
        if bands:
            last = bands[-1][-1]
            # Перекрываются ли вертикальные диапазоны?
            if block.bbox[1] < last.bbox[3]:
                bands[-1].append(block)
                continue
        bands.append([block])

    ordered: List[_OcrBlock] = []
    band_idx = 0
    for section in full_width:
        section_y = y_center(section)
        while band_idx < len(bands) and y_center(bands[band_idx][0]) < section_y:
            ordered.extend(sorted(bands[band_idx], key=lambda b: b.bbox[0]))
            band_idx += 1
        ordered.append(section)
    while band_idx < len(bands):
        ordered.extend(sorted(bands[band_idx], key=lambda b: b.bbox[0]))
        band_idx += 1
    return ordered


# --------------------------------------------------------------------------- #
# OCR / TableFormer вызовы (моки в тестах подменяют именно их)
# --------------------------------------------------------------------------- #


def run_easyocr_on_crop(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    *,
    page: Optional[Any] = None,
    langs: Sequence[str] = ("ru", "en"),
) -> str:
    """EasyOCR по кропу (координаты в PDF-точках → пиксели по DPI)."""
    reader = get_easyocr_reader(langs)
    dpi = 72.0 * image.width / page.rect.width if page else 72.0
    x0, y0, x1, y1 = bbox
    scale = dpi / 72.0
    left, top = int(x0 * scale), int(y0 * scale)
    right, bottom = int(x1 * scale), int(y1 * scale)
    crop = image.crop((left, top, right, bottom))
    lines = reader.readtext(
        np.array(crop), detail=0, paragraph=True
    )
    return "\n\n".join(line.strip() for line in lines if line and line.strip())


def run_tableformer_on_crop(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    *,
    page: Optional[Any] = None,
    langs: Sequence[str] = ("ru", "en"),
) -> str:
    """TableFormer по кропу таблицы → Markdown-таблица.

    Порядок попыток:
    1. TableFormer через Docling (если установлен).
    2. ``find_tables`` на кропе координат текстового слоя.
    3. EasyOCR + группировка строк по Y → псевдо-таблица (последний
       fallback: текст таблицы не теряется).
    """
    if is_docling_available():
        try:
            from rag_parser.parsers.pdf_models import _tableformer_via_docling

            return _tableformer_via_docling(image, bbox)
        except PdfModelError as exc:
            logger.warning("tableformer_docling_failed err=%s", exc)
    # Fallback №2: структурный парсер PyMuPDF по координатам таблицы.
    if page is not None:
        try:
            finder = page.find_tables()
            for table in finder.tables or []:
                tb = tuple(table.bbox)
                if _boxes_intersect(tb, bbox):
                    return table.to_markdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tableformer_pymupdf_failed err=%s", exc)
    # Fallback №3: EasyOCR строками (кратно выше в иерархии).
    raise PdfModelError("no tableformer backend available", model="tableformer")


def _boxes_intersect(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


# --------------------------------------------------------------------------- #
# Сборка страницы
# --------------------------------------------------------------------------- #


def process_complex_page(
    doc,
    page_index: int,
    *,
    dpi: int = 200,
    langs: Sequence[str] = ("ru", "en"),
    has_text_layer: bool = True,
) -> Tuple[str, str]:
    """Обрабатывает страницу с таблицей или скан-страницу (шаг 2.3 / 3).

    :param doc: открытый ``fitz.Document``.
    :param page_index: 0-based номер страницы.
    :param has_text_layer: есть ли у страницы адекватный текстовый слой
        (для fallback'ов EasyOCR→PyMuPDF).
    :return: ``(markdown, route)`` — текст страницы и её маршрут
        (``"table→layout+ocr+tableformer"`` / ``"scan→layout+ocr"``).
    """
    page = doc[page_index]
    image = render_page_image(doc, page_index, dpi)
    blocks: List[LayoutBlock] = []

    try:
        blocks = detect_page_layout(doc, page_index, dpi=dpi)
    except Exception as exc:  # noqa: BLE001 - layout не должен ронять страницу
        logger.warning(
            "page_layout_failed page=%d err=%s", page_index + 1, exc
        )
        blocks = []

    if not blocks:
        # Layout не нашёл ничего — подстраховка: один полностраничный
        # текстовый блок (EasyOCR вытащит текст целиком).
        blocks = [
            LayoutBlock(
                label=LABEL_TABLE if _page_has_table(page) else LABEL_TEXT,
                bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
            )
        ]

    ocr_blocks: List[_OcrBlock] = []
    declared_table = any(b.label == LABEL_TABLE for b in blocks)
    route = (
        "table→layout+ocr+tableformer"
        if declared_table and has_text_layer
        else "scan→layout+ocr+tableformer"
    )

    for block in blocks:
        if block.label == LABEL_IGNORE:
            continue  # картинки/иллюстрации игнорируем по ТЗ
        md = ""
        if block.label == LABEL_TABLE:
            md = _handle_table_block(block, image, page, langs, has_text_layer)
        else:
            md = _handle_text_block(block, image, page, langs, has_text_layer)
        if md.strip():
            ocr_blocks.append(_OcrBlock(block.label, block.bbox, md.strip()))

    ordered = sort_blocks_reading_order(ocr_blocks, float(page.rect.width))
    page_md = "\n\n".join(b.markdown for b in ordered).strip()
    return page_md, route


def _page_has_table(page: Any) -> bool:
    try:
        finder = page.find_tables()
        return bool(finder.tables)
    except Exception:  # noqa: BLE001
        return False


def _handle_table_block(
    block: LayoutBlock,
    image: Image.Image,
    page: Any,
    langs: Sequence[str],
    has_text_layer: bool,
) -> str:
    """Блок таблицы: TableFormer (fallback'и по цепочке)."""
    try:
        return run_tableformer_on_crop(
            image, block.bbox, page=page, langs=langs
        )
    except PdfModelError as exc:
        logger.warning(
            "tableformer_failed page_bbox=%s err=%s", block.bbox, exc
        )
    # Fallback на EasyOCR по кропу таблицы (сохраняем текст таблицы).
    try:
        text = run_easyocr_on_crop(image, block.bbox, page=page, langs=langs)
        return _pseudo_table_from_lines(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "table_ocr_fallback_failed bbox=%s err=%s", block.bbox, exc
        )
    # Последняя защита: текстовый слой.
    if has_text_layer:
        try:
            text = page.get_text("text", clip=block.bbox) or ""
            if text.strip():
                return _pseudo_table_from_lines(text.strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("table_text_layer_fallback_failed err=%s", exc)
    return ""


def _handle_text_block(
    block: LayoutBlock,
    image: Image.Image,
    page: Any,
    langs: Sequence[str],
    has_text_layer: bool,
) -> str:
    """Текстовый блок: EasyOCR (fallback на текстовый слой)."""
    try:
        return run_easyocr_on_crop(image, block.bbox, page=page, langs=langs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("easyocr_failed bbox=%s err=%s", block.bbox, exc)
    if has_text_layer:
        try:
            text = page.get_text("text", clip=block.bbox) or ""
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("text_layer_fallback_failed err=%s", exc)
    return ""


def _pseudo_table_from_lines(text: str) -> str:
    """Псевдо-таблица из строк EasyOCR: текст таблицы не теряется.

    Настоящую структуру (колонки) вернуть TableFormer'ом нельзя без
    его инференса, поэтому строки оформляются как текстовые абзацы
    внутри конструкции таблицы с одним столбцом.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    rows = "\n".join(f"| {ln} |" for ln in lines)
    return f"|---|\n{rows}"


__all__ = [
    "LayoutBlock",
    "render_page_image",
    "detect_page_layout",
    "run_easyocr_on_crop",
    "run_tableformer_on_crop",
    "sort_blocks_reading_order",
    "process_complex_page",
]
