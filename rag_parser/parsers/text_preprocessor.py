"""Препроцессинг простого текста (``.txt``) → HTML.

Зачем это нужно
---------------
``RawTextReader`` не детектирует таблицы: ASCII-таблица из ГОСТ приходит
обычными строками и в дереве становится абзацами. Здесь такие блоки
находятся регулярными выражениями, заменяются на ``<table>`` и уходят в
``HtmlReader``, а дальше работает общий маркерный маршрут
(см. ``html_preprocessor``).

Что считается таблицей
----------------------
Блок начинается со строки с ``|`` (минимум одна ячейка), за которой идёт
строка-разделитель, и продолжается строками с ``|``. Разделителем может
быть как Markdown-вид ``|---|---|``, так и «гостовский» — сплошная
полоса дефисов во всю ширину::

    Показатель | Q1 | Q2
    -------------------------------
    Выручка    | 1000 | 1100

Второй вид неоднозначен: та же полоса дефисов идёт под заголовком
(``Предисловие`` + ``------``). Различает их требование, чтобы строка над
полосой содержала ``|``: у заголовка его нет.

Чего модуль НЕ делает
---------------------
Заголовки в ``.txt`` не выделяются. В отличие от ``.html``/``.md``/``.xml``
у простого текста нет разметки, поэтому классификация заголовков
остаётся прежней (``header_promoter``). Таблицы лишь оборачиваются в
HTML — они не приходят «с разметкой» и на заголовки не влияют.
"""

import re
from html import escape
from pathlib import Path
from typing import List, Optional, Tuple

from rag_parser.parsers.html_preprocessor import inject_table_markers

__all__ = [
    "find_table_blocks",
    "text_to_html",
    "prepare_text_for_dedoc",
]

#: Строка-разделитель Markdown-таблицы: ``|---|---|``.
_MD_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")

#: Строка-разделитель «гостовской» таблицы: сплошные дефисы.
_ASCII_SEPARATOR = re.compile(r"^\s*-{3,}\s*$")

#: Декоративная строка-разделитель: «==========», «--------------». Это
#: оформление, а не текст: в эталонном дереве таких абзацев нет, и без
#: фильтра каждая превращается в лишний чанк (в gost_report.txt их 21).
_DECORATION_LINE = re.compile(r"^[-=_~*·.\s]{3,}$")

#: Метка приложения («Приложение А (справочное)») — строка состоит ТОЛЬКО из
#: метки. Строка оглавления «Приложение А (справочное) Пример использования
#: … 34» под это не подпадает: там после метки идёт название и номер страницы.
_APPENDIX_HEADER_RE = re.compile(r"^Приложение\s+[^\s(]+(?:\s*\([^)]*\))?$", re.IGNORECASE)

#: Запись оглавления: номер страницы в конце строки («… 34»). Название
#: приложения из оглавления заголовком быть не может.
_TOC_ENTRY_RE = re.compile(r"…\s*\d+\s*$")

#: Предельная длина названия приложения. Названия приложений короткие
#: («Пример использования настоящего стандарта»), а за меткой приложения
#: всегда идёт обычный текст — длиннее этой границы он не заголовок.
_APPENDIX_TITLE_MAX_LEN = 120


#: Пустая подпись таблицы («Таблица:» без номера и названия) прямо перед
#: таблицей. В эталонном DOCX такой подписи нет вовсе, а из текста она
#: попадала отдельным чанком-однострочником; содержания в ней нет, поэтому
#: выбрасываем, а не тащим в чанки.
_EMPTY_TABLE_CAPTION = re.compile(r"^Таблица\s*:\s*$", re.IGNORECASE)


def _is_content_line(line: str) -> bool:
    """Строка, из которой получается абзац: не пустая и не разделитель."""
    stripped = (line or "").strip()
    return bool(stripped) and not _DECORATION_LINE.match(stripped)


def _is_empty_caption(line: str) -> bool:
    return bool(_EMPTY_TABLE_CAPTION.match((line or "").strip()))

def _is_pipe_row(line: str) -> bool:
    """Строка-таблица: хотя бы один ``|`` (двухколоночная таблица — это
    одна черта на строку) и строка не является разделителем."""
    return "|" in line and not _MD_SEPARATOR.match(line)


def _is_separator(line: str) -> bool:
    return bool(_ASCII_SEPARATOR.match(line) or _MD_SEPARATOR.match(line))


def _split_cells(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def find_table_blocks(lines: List[str]) -> List[Tuple[int, int, List[List[str]]]]:
    """Находит таблицы в тексте.

    :return: список ``(начало, конец, строки_ячеек)`` с полуоткрытым
        интервалом ``[начало, конец)``. Пересекающихся блоков не бывает.
    """
    blocks: List[Tuple[int, int, List[List[str]]]] = []
    index = 0
    total = len(lines)
    while index < total:
        if not _is_pipe_row(lines[index]):
            index += 1
            continue
        separator_index = index + 1
        if separator_index >= total or not _is_separator(lines[separator_index]):
            index += 1
            continue

        rows = [_split_cells(lines[index])]
        cursor = separator_index + 1
        while cursor < total and _is_pipe_row(lines[cursor]):
            rows.append(_split_cells(lines[cursor]))
            cursor += 1
        if len(rows) < 2:
            # Заголовок без данных — не таблица.
            index = cursor
            continue
        blocks.append((index, cursor, rows))
        index = cursor
    return blocks


def _render_table(rows: List[List[str]]) -> str:
    width = max(len(row) for row in rows)
    parts = ["<table>"]
    for position, row in enumerate(rows):
        padded = row + [""] * (width - len(row))
        tag = "th" if position == 0 else "td"
        cells = "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in padded)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")
    return "".join(parts)


def _blocks(lines: List[str]) -> List[Tuple[str, object]]:
    """Разбирает строки на абзацы и таблицы: ``[(kind, payload), ...]``."""
    items: List[Tuple[str, object]] = []
    cursor = 0
    for start, end, rows in find_table_blocks(lines):
        index = cursor
        while index < start:
            line = lines[index]
            if (
                _is_empty_caption(line)
                and not any(_is_content_line(between) for between in lines[index + 1 : start])
            ):
                # Подпись без номера и названия прямо перед таблицей.
                index += 1
                continue
            if _is_content_line(line):
                items.append(("p", line.strip()))
            index += 1
        items.append(("table", rows))
        cursor = end
    for line in lines[cursor:]:
        if _is_content_line(line):
            items.append(("p", line.strip()))
    return items


def _mark_appendix_titles(items: List[Tuple[str, object]]) -> List[Tuple[str, object]]:
    """Метка приложения -> ``<h1>``, следующая за ней строка -> ``<h2>``.

    В DOCX «Приложение А (справочное)» и «Пример использования настоящего
    стандарта» — два заголовка (уровни 1 и 2). В простом тексте между ними
    стоит линия-разделитель, и без этого правила название остаётся абзацем:
    подраздел приложения теряется, а его чанки получают неверную крошку.

    Обе строки размечаем явно: ``HtmlReader`` считает уровни от самой глубокой
    встретившейся метки, поэтому один ``<h2>`` без ``<h1>`` дал бы уровень 1.
    """
    marked = list(items)
    for index, (kind, payload) in enumerate(items[:-1]):
        if kind != "p" or not isinstance(payload, str):
            continue
        if not _APPENDIX_HEADER_RE.match(payload):
            continue
        marked[index] = ("h1", payload)
        next_kind, next_payload = marked[index + 1]
        if next_kind != "p" or not isinstance(next_payload, str):
            continue
        if _TOC_ENTRY_RE.search(next_payload):
            continue
        if len(next_payload) <= _APPENDIX_TITLE_MAX_LEN:
            marked[index + 1] = ("h2", next_payload)
    return marked


def text_to_html(text: str) -> str:
    """Оборачивает текст в HTML, заменяя найденные ASCII/Markdown-таблицы."""
    if not text or not text.strip():
        return "<html><body></body></html>"

    items = _mark_appendix_titles(_blocks(text.split("\n")))
    parts: List[str] = []
    for kind, payload in items:
        if kind == "table":
            parts.append(_render_table(payload))  # type: ignore[arg-type]
        else:
            tag = kind if kind in ("h1", "h2") else "p"
            parts.append(f"<{tag}>{escape(str(payload))}</{tag}>")

    return "<html><body>" + "".join(parts) + "</body></html>"


def _read_text(file_path: Path) -> Optional[str]:
    try:
        raw = file_path.read_bytes()
    except OSError:
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def prepare_text_for_dedoc(
    file_path: Path, target_dir: Path
) -> Tuple[Optional[Path], List[str]]:
    """Готовит ``.txt`` к разбору dedoc.

    Возвращает ``(временный_html, markdown_таблицы)``. Если таблиц нет,
    отдаём ``(None, [])`` — дешевле и надёжнее разбирать исходный текст
    нативным ``RawTextReader``, который лучше сохраняет переносы строк.
    """
    text = _read_text(file_path)
    if text is None:
        return None, []

    html = text_to_html(text)
    marked, tables = inject_table_markers(html)
    if not tables:
        return None, []

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = target_dir / f"{file_path.stem}_from_txt.html"
    temp_path.write_text(marked, encoding="utf-8")
    return temp_path, tables