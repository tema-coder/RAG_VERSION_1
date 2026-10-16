"""Препроцессинг структурированного XML (ГОСТ и подобные) → HTML.

Зачем это нужно
---------------
``.xml`` уходил в ``RawTextReader``, который игнорирует теги: документ
превращался в мешанину текста без единого заголовка и без таблиц.

Здесь XML разбирается настоящим парсером (``ElementTree``) и переписывается
в HTML по явным правилам, после чего идёт тем же маркерным маршрутом, что
и обычный HTML (см. ``html_preprocessor``).

Правила маппинга
----------------
=========================  ==========================================
XML                        HTML
=========================  ==========================================
``<title>``                ``<h1>``
``<preface>``              ``<h2>`` «Предисловие»
``<toc>``                  ``<h2>`` «Содержание»
``<introduction>``         ``<h2>`` «Введение»
``<bibliography>``         ``<h2>`` «Библиография»
``<section number title>`` ``<h{2+depth}>`` ``number title``
``<appendix letter title>````<h{2+depth}>`` ``Приложение letter type``
``<paragraph>`` и прочие   ``<p>`` (префикс из ``number``/``term``)
``<table>``                обычный ``<table>`` с ``<th>``/``<td>``
=========================  ==========================================

Уровень заголовка задаётся **глубиной вложенности** ``<section>``: разделы
1–6 — ``<h2>``, 5.1/6.2 — ``<h3>``, 6.2.1 — ``<h4>``. Это надёжнее, чем
считать точки в ``number``: у приложений номера нет вовсе (только буква).

Неизвестные теги не теряются: их дети рендерятся как есть, поэтому
схема, которую мы не знаем, не приводит к молчаливой потере текста.
"""

import re
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag_parser.parsers.html_preprocessor import inject_table_markers

__all__ = [
    "xml_to_html",
    "prepare_xml_for_dedoc",
    "local_name",
]

#: Контейнеры с неявным названием: тег -> заголовок раздела.
_SECTION_TITLES: Dict[str, str] = {
    "preface": "Предисловие",
    "toc": "Содержание",
    "contents": "Содержание",
    "introduction": "Введение",
    "bibliography": "Библиография",
    "abstract": "Реферат",
}

#: Теги, содержимое которых становится абзацем.
_PARAGRAPH_TAGS = frozenset(
    {
        "paragraph",
        "definition_text",
        "item",
        "entry",
        "reference",
        "note",
        "term",
        "value",
        "requirement",
        "clause",
        "step",
        "keywords",
        "editor",
        "tech_editor",
        "corrector",
        "layout",
        "udc",
        "oks",
    }
)

#: Терминологическая статья: в DOCX это абзац-заголовок уровня 2, поэтому
#: ``<definition>`` рендерим заголовком с глубиной из номера («3.1» -> ``<h3>``).
_DEFINITION_TAG = "definition"

#: Маркер номера страницы в строке оглавления. ``header_promoter`` гасит
#: строки оглавления только если в них есть «… N»; без него оглавление
#: превращается в 19 фантомных заголовков (см. ``_is_toc_entry``).
TOC_PAGE_MARKER = "…"

#: Строковая классификация, которая в DOCX тоже заголовок («ОКС 35.240.99»).
#: «УДК 004.9:006.354» в эталоне остаётся абзацем.
_CLASSIFICATION_TAG = "oks"

#: Теги, содержимое которых не должно попадать в текст.
_SKIP_TAGS = frozenset({"thead", "tbody", "tfoot"})

#: Структурные обёртки: не даём заголовку, просто рендерим детей.
_CONTAINER_TAGS = frozenset(
    {"gost-standard", "standard", "document", "sections", "appendices", "root"}
)

_WHITESPACE = re.compile(r"\s+")
_MAX_HEADING_LEVEL = 6


def local_name(tag: str) -> str:
    """Имя тега без XML-namespace (``{urn:x}section`` -> ``section``)."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _clean(text: Optional[str]) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _collect_text(element: ET.Element) -> str:
    """Плоский текст элемента вместе с текстом вложенных узлов."""
    parts = [element.text or ""]
    for child in element:
        parts.append(_collect_text(child))
        parts.append(child.tail or "")
    return _clean(" ".join(parts))


def _heading_text(element: ET.Element, tag: str) -> str:
    """Заголовок раздела: явный title + номер, либо неявное имя тега."""
    if tag in _SECTION_TITLES:
        return _SECTION_TITLES[tag]

    title = _clean(element.get("title") or element.get("name"))
    number = _clean(element.get("number"))

    if tag == "appendix":
        letter = _clean(element.get("letter")) or number
        kind = _clean(element.get("type"))
        head = f"Приложение {letter}".strip()
        if kind:
            head = f"{head} ({kind})"
        return head

    if tag == "title":
        return _collect_text(element) or title

    if tag == "standard_number":
        return number

    if number and title:
        return f"{number} {title}"
    return title or number


def _appendix_headings(element: ET.Element, depth: int) -> List[Tuple[int, str]]:
    """Два заголовка приложения: «Приложение А (справочное)» и его название.

    В DOCX это два абзаца: метка приложения (уровень 1) и название (уровень 2).
    Сливать их в один заголовок нельзя — в эталоне «Пример использования
    настоящего стандарта» вложен в «Приложение А (справочное)».
    """
    level = min(_MAX_HEADING_LEVEL, 2 + depth)
    label = _heading_text(element, "appendix")
    title = _clean(element.get("title"))
    headings = [(level, label)] if label else []
    if title:
        headings.append((level + 1, title))
    return headings


def _definition_heading(element: ET.Element) -> str:
    """Заголовок термина: ``3.1 термин: текст определения``.

    В DOCX вся терминологическая статья — один абзац, помеченный стилем
    заголовка, поэтому в заголовок уходит и определение тоже. Термин при этом
    остаётся в тексте: в эталоне «3.1 синтетический документ (synthetic
    document): Документ, созданный программным способом…».
    """
    number = _clean(element.get("number"))
    term = _clean(element.get("term"))
    text = _collect_text(element)
    head = f"{number} {term}".strip() if number and term else (number or term)
    if not head:
        return text
    if text.lower().startswith(head.lower()):
        return text
    return f"{head}: {text}".strip()


def _toc_entry_text(element: ET.Element) -> str:
    """Строка оглавления: ``1 Область применения … 1``.

    Номер страницы нужен самому ``header_promoter``: строки оглавления
    опознаются по маркеру «… N», и без него оглавление становится
    фантомными заголовками (в ``gost_report.xml`` это 19 узлов).
    """
    number = _clean(element.get("number"))
    title = _clean(element.get("title")) or _collect_text(element)
    page = _clean(element.get("page"))
    head = f"{number} {title}".strip()
    return f"{head} {TOC_PAGE_MARKER} {page}".strip() if page else head


def _paragraph_text(element: ET.Element) -> str:
    """Текст абзаца с префиксом из атрибутов нумерации/термина."""
    tag = local_name(element.tag)
    if tag == "entry":
        return _toc_entry_text(element)
    text = _collect_text(element)
    if tag == "keywords" and text and not text.lower().startswith("ключевые слова"):
        # Метка — часть смысла тега, в эталонном DOCX она набрана текстом.
        return f"Ключевые слова: {text}"
    prefix = _clean(element.get("term")) or _clean(element.get("number"))
    if prefix and not text.startswith(prefix):
        return f"{prefix} {text}".strip()
    return text


def _render_table(element: ET.Element) -> str:
    """``<table>`` → обычный HTML-таблица (её потом заберёт маршрут)."""
    rows: List[str] = []
    for row in element.iter():
        if local_name(row.tag) != "tr":
            continue
        cells: List[str] = []
        for cell in row:
            cell_tag = local_name(cell.tag)
            if cell_tag not in ("td", "th"):
                continue
            tag = "th" if cell_tag == "th" else "td"
            cells.append(f"<{tag}>{escape(_clean(_collect_text(cell)))}</{tag}>")
        if cells:
            rows.append("<tr>" + "".join(cells) + "</tr>")
    if not rows:
        return ""
    return "<table>" + "".join(rows) + "</table>"


def _heading_level_for_number(number: str) -> int:
    """Уровень заголовка по номеру: «3.1» -> 3, «6.2.2» -> 4 (для ``<hN>``).

    В DOCX «3.1» — это заголовок второго уровня, поэтому в HTML-разметке он
    должен быть на уровень глубже (``<h3>``): сдвиг на единицу выполняет
    ``dedoc_tree.normalize_markup_levels`` по реальному уровню разделов.
    """
    depth = len([part for part in (number or "").split(".") if part])
    return min(_MAX_HEADING_LEVEL, depth + 1)


def _render(element: ET.Element, depth: int) -> str:
    tag = local_name(element.tag)
    parts: List[str] = []

    if tag == "table":
        return _render_table(element)

    if tag in _SKIP_TAGS:
        return ""

    if tag == _CLASSIFICATION_TAG:
        # ОКС — заголовок в DOCX, УДК — обычный абзац.
        text = _clean(element.text)
        if text:
            parts.append(f"<h1>{escape(text)}</h1>")
    elif tag == "colophon":
        # Порядок как в DOCX: УДК, затем ОКС-заголовок, затем ключевые слова.
        # ОКС — заголовок, УДК — обычный абзац (в DOCX так же).
        for attribute, tag_name in (("udc", "p"), ("oks", "h1")):
            value = _clean(element.get(attribute))
            if value:
                parts.append(f"<{tag_name}>{escape(value)}</{tag_name}>")
    elif tag in _CONTAINER_TAGS:
        # Номер стандарта («ГОСТ Р 99999—2026») в атрибуте корня — в DOCX это
        # заголовок титула, который иначе молча теряется; «iso» — абзац.
        number = _clean(element.get("number"))
        if number:
            parts.append(f"<h1>{escape(number)}</h1>")
        iso = _clean(element.get("iso"))
        if iso:
            parts.append(f"<p>{escape(iso)}</p>")
    elif tag == "preface":
        # Пункты предисловия («<item number="1">») — заголовки уровня 1, как в
        # DOCX. Рендерим их здесь и НЕ спускаемся в общий обход детей, иначе
        # каждый пункт попадёт в документ дважды: заголовком и абзацем.
        parts.append(f"<h2>{escape(_SECTION_TITLES['preface'])}</h2>")
        for index, item in enumerate(element, start=1):
            number = _clean(item.get("number")) or str(index)
            text = _clean(_collect_text(item))
            if text:
                parts.append(f"<h1>{escape(f'{number} {text}')}</h1>")
        return "".join(parts)
    elif tag == _DEFINITION_TAG:
        text = _definition_heading(element)
        if text:
            level = _heading_level_for_number(_clean(element.get("number")))
            parts.append(f"<h{level}>{escape(text)}</h{level}>")
    elif tag == "appendix":
        for level, text in _appendix_headings(element, depth):
            parts.append(f"<h{level}>{escape(text)}</h{level}>")
    elif tag in _SECTION_TITLES or tag in ("section", "title"):
        text = _heading_text(element, tag)
        if text:
            level = 1 if tag == "title" else min(_MAX_HEADING_LEVEL, 2 + depth)
            parts.append(f"<h{level}>{escape(text)}</h{level}>")
    elif tag in _PARAGRAPH_TAGS:
        text = _paragraph_text(element)
        if text:
            parts.append(f"<p>{escape(text)}</p>")
    else:
        # Неизвестный тег: текст сохраняем, разметку не выдумываем.
        text = _clean(element.text)
        if text:
            parts.append(f"<p>{escape(text)}</p>")

    # Дети рендерим всегда: у раздела/абзаца они тоже нужны.
    child_depth = depth + 1 if tag in ("section", "appendix") else depth
    for child in element:
        rendered = _render(child, child_depth)
        if rendered:
            parts.append(rendered)
            tail = _clean(child.tail)
            if tail:
                parts.append(f"<p>{escape(tail)}</p>")

    return "".join(parts)


def xml_to_html(xml_text: str) -> str:
    """Разбирает XML-текст и возвращает готовый HTML-документ."""
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError:
        return ""
    body = _render(root, 0)
    return f"<html><body>{body}</body></html>"


def _read_text(file_path: Path) -> Optional[str]:
    try:
        raw = file_path.read_bytes()
    except OSError:
        return None
    head = raw[:65536]
    if head.startswith(b"\xef\xbb\xbf"):
        head = raw[3:65536]
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def prepare_xml_for_dedoc(
    file_path: Path, target_dir: Path
) -> Tuple[Optional[Path], List[str]]:
    """Готовит ``.xml`` к разбору dedoc.

    Возвращает ``(путь_к_временному_html, markdown_таблицы)``. Если XML
    невалиден — ``(None, [])``, и вызывающий отдаст файл как есть.
    """
    text = _read_text(file_path)
    if text is None:
        return None, []

    html = xml_to_html(text)
    if not html.strip():
        return None, []

    marked, tables = inject_table_markers(html)
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = target_dir / f"{file_path.stem}_from_xml.html"
    temp_path.write_text(marked, encoding="utf-8")
    return temp_path, tables