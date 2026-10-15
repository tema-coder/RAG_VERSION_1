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


def text_to_html(text: str) -> str:
    """Оборачивает текст в HTML, заменяя найденные ASCII/Markdown-таблицы."""
    if not text or not text.strip():
        return "<html><body></body></html>"

    lines = text.split("\n")
    blocks = find_table_blocks(lines)

    parts: List[str] = []
    cursor = 0
    for start, end, rows in blocks:
        for line in lines[cursor:start]:
            if line.strip():
                parts.append(f"<p>{escape(line.strip())}</p>")
        parts.append(_render_table(rows))
        cursor = end
    for line in lines[cursor:]:
        if line.strip():
            parts.append(f"<p>{escape(line.strip())}</p>")

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