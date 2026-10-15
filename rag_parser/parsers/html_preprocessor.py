"""Подготовка HTML для сохранения позиции таблиц.

Проблема
--------
``HtmlReader`` в dedoc 2.8 **не создаёт ``TableAnnotation`` ни для одной
таблицы** (``html_reader.py`` даже не импортирует этот класс). Таблицы
приходят плоским списком ``content.tables`` без привязки к тексту, поэтому
в дереве они не имеют позиции и уходят в конец документа (orphan).

Хуже того: при ``handle_invisible_table=true`` содержимое таблицы
**дублируется** обычным текстом в узлах дерева:

    <h2>Данные</h2>
    <table>...</table>
    -> узел: '__RAGTABLE_0__\\nРегион\\tQ1Север\\t100\\nПосле таблицы.'

Решение
-------
Таблицы извлекаются **здесь, нами** (BeautifulSoup + собственный рендер в
Markdown), а в HTML остаётся только маркер-абзац:

    <h2>Данные</h2>
    <p>__RAGTABLE_0__</p>

Далее dedoc разбирает чистый HTML: в дереве нет ни таблиц, ни их дублирующего
текста, но **остаются заголовки и точные позиции**. Затем
:func:`extract_and_attach_tables` идёт по дереву в порядке документа,
запоминает последний заголовок и превращает каждый найденный маркер в
узел-таблицу, подвешенный к этому заголовку. Так ``parent_id`` таблицы
становится корректным — ровно то, что нужно для Child-Parent выдачи.

Почему не дедуп по тексту
-------------------------
Вариант «оставить таблицы dedoc и вырезать их текст из соседних абзацев»
отвергнут: дедуп строился бы на угадывании, сколько именно символов
принадлежит таблице, и ломался бы на любом нестандартном содержимом ячеек.
Извлечение таблиц до отдачи dedoc исключает проблему, а не маскирует её.
"""

import re
from typing import List, Sequence, Tuple

from rag_parser.core.interfaces import DocumentNode

#: Маркер позиции таблицы. Число — индекс таблицы в порядке документа.
MARKER_RE = re.compile(r"__RAGTABLE_(\d+)__")
#: Префикс маркера в тексте узла.
MARKER_PREFIX = "__RAGTABLE_"

#: Ограничение на размер ячейки при рендере таблицы в Markdown.
MAX_CELL_LENGTH = 512


def _rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """Рендерит строки в Markdown-таблицу (та же функция, что у парсера)."""
    from rag_parser.parsers.base import rows_to_markdown_table

    materialized = [list(row) for row in rows]
    width = max((len(row) for row in materialized), default=0)
    normalized: List[List[str]] = []
    for row in materialized:
        cells = [
            " ".join(str(cell).split()).strip()[:MAX_CELL_LENGTH]
            for cell in row
        ]
        normalized.append(cells + [""] * (width - len(cells)) if width else cells)
    if not normalized:
        return ""
    markdown, _, _ = rows_to_markdown_table(normalized)
    return markdown


def extract_table_rows(table_node) -> List[List[str]]:
    """Строки ``<table>`` как список списков текстов ячеек.

    Пустые строки отбрасываются (частая верстка оставляет ``<tr></tr>``).
    """
    rows: List[List[str]] = []
    for row in table_node.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if any(values):
            rows.append(values)
    return rows


def inject_table_markers(html_content: str) -> Tuple[str, List[str]]:
    """Заменяет каждую ``<table>`` маркером-абзацем.

    :return: ``(html с маркерами, список markdown-таблиц в порядке документа)``.
        Индекс маркера совпадает с индексом в списке таблиц.
    """
    from bs4 import BeautifulSoup

    if not html_content or not html_content.strip():
        return html_content, []

    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception:  # noqa: BLE001 - битый HTML читаем как есть
        return html_content, []

    tables: List[str] = []
    for table_node in soup.find_all("table"):
        markdown = _rows_to_markdown(extract_table_rows(table_node))
        if not markdown:
            continue
        marker = soup.new_tag("p")
        marker.string = f"{MARKER_PREFIX}{len(tables)}__"
        table_node.replace_with(marker)
        tables.append(markdown)
    return str(soup), tables


def strip_markers(text: str) -> Tuple[str, List[int]]:
    """Убирает маркеры из текста, возвращает очищенный текст и индексы.

    Маркер может оказаться склеенным с соседним содержимым
    (``'Ещё.\\n__RAGTABLE_1__'``), поэтому ищем по регулярке, а не сравниваем.
    """
    found = [int(value) for value in MARKER_RE.findall(text or "")]
    cleaned = MARKER_RE.sub("", text or "")
    # Схлопываем пустые строки, оставшиеся от удаления маркера.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), found


def _table_node(markdown: str, index: int) -> DocumentNode:
    return DocumentNode(
        type="table",
        level=0,
        text=markdown,
        children=[],
        metadata={"marker_index": index, "source": "html_table"},
    )


def extract_and_attach_tables(
    tree: DocumentNode, markdown_tables: Sequence[str]
) -> Tuple[int, int]:
    """Превращает маркеры в узлы-таблицы, сохраняя их место в документе.

    Таблица встаёт **на место маркера**, а не в конец раздела. Разница
    видна на документе с вложенными подразделами: если у раздела «5» есть
    своя таблица и вложенные «5.1»/«5.2», то дописывание в конец детей
    раздела «5» выводило её уже после поддерева «5.2» — таблица раздела 5
    выглядела как вторая таблица раздела 5.2, а сам раздел 5 оставался
    без таблицы.

    Таблица выше первого заголовка остаётся orphan — привязывать её не к
    чему.

    :return: ``(привязано, orphan)``
    """
    state = {"attached": 0, "seen_header": False, "orphans": []}

    def walk(node: DocumentNode) -> List[DocumentNode]:
        """Возвращает узлы, которые должны стоять **на месте** ``node``.

        Обычно это ``[node]``, но абзац-маркер без остаточного текста
        заменяется таблицей (или несколькими, если маркеров в тексте
        было больше одного).
        """
        text, found = strip_markers(node.text)
        node.text = text

        tables: List[DocumentNode] = []
        for index in found:
            if index < 0 or index >= len(markdown_tables):
                continue
            table_node = _table_node(markdown_tables[index], index)
            if state["seen_header"]:
                tables.append(table_node)
                state["attached"] += 1
            else:
                state["orphans"].append(table_node)

        if node.type == "header":
            state["seen_header"] = True

        children: List[DocumentNode] = []
        for child in node.children:
            children.extend(walk(child))
        node.children = children

        # Таблицы из текста самого узла идут сразу за его текстом.
        if tables and not node.text and not children:
            # Чистый абзац-маркер: узел выбрасываем, остаются таблицы.
            return tables
        out = [node] if (node.text or children or node.type in ("header", "table")) else []
        out.extend(tables)
        return out

    # Корень обрабатываем на уровне детей: walk() вернул бы [tree], и тогда
    # root попал бы в список собственных детей (бесконечная рекурсия).
    root_nodes: List[DocumentNode] = []
    for child in tree.children:
        root_nodes.extend(walk(child))
    root_nodes.extend(state["orphans"])
    tree.children = root_nodes
    return state["attached"], len(state["orphans"])


__all__ = [
    "MARKER_PREFIX",
    "MARKER_RE",
    "extract_and_attach_tables",
    "extract_table_rows",
    "inject_table_markers",
    "strip_markers",
]