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

from bs4 import BeautifulSoup, NavigableString

from rag_parser.core.interfaces import DocumentNode

#: Маркер позиции таблицы. Число — индекс таблицы в порядке документа.
MARKER_RE = re.compile(r"__RAGTABLE_(\d+)__")
#: Префикс маркера в тексте узла.
MARKER_PREFIX = "__RAGTABLE_"

#: Ограничение на размер ячейки при рендере таблицы в Markdown.
MAX_CELL_LENGTH = 512

# ---------------------------------------------------------------------------
# Приведение авторской разметки к структуре ГОСТ
# ---------------------------------------------------------------------------
#: Классы титульного блока, которые в эталонном DOCX являются заголовками
#: уровня 1. ``iso``/``edition``/``publisher`` в перечень НЕ входят: в DOCX это
#: обычные абзацы, иначе дерево разъехалось бы с эталоном лишними разделами.
TITLE_PAGE_HEADER_CLASSES = ("org", "country", "num")

#: Заголовок, после которого идёт нумерованный список пунктов предисловия.
PREFACE_WORDS = ("предисловие",)

#: Строка классификации, которая в DOCX тоже заголовок («ОКС 35.240.99»).
#: «УДК 004.9:006.354» в эталоне абзац, поэтому проверяем именно ОКС.
CLASSIFICATION_RE = re.compile(r"^ОКС\s+[\d.]+", re.IGNORECASE)

#: Нумерованный термин/подраздел: «3.1 синтетический документ: …». Требуем
#: хотя бы одну точку, иначе под правило попал бы обычный абзац списка.
DOTTED_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+\S")

#: Номер, который рендерер добавил в текст пункта списка («1. ПОДГОТОВЛЕН …»).
#: В эталонном дереве номер без точки: «1 ПОДГОТОВЛЕН …».
LIST_NUMBER_PREFIX_RE = re.compile(r"^\d+\s*[.)]\s*")

#: Пункт списка, уже пронумерованный в самом тексте («[1] ISO/IEC …»).
#: В DOCX это обычный абзац библиографии, поэтому оставляем его абзацем:
#: иначе рендерер добавит вторую нумерацию («1. [1] ISO/IEC …»), которой нет
#: в эталоне.
BRACKETED_ITEM_RE = re.compile(r"^\[\d+\]")

#: Теги жирного начертания: термин в ГОСТ набран полужирным в начале абзаца.
BOLD_TAGS = ("b", "strong")

_MAX_HEADING_LEVEL = 6


def _text_of(node) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def _leading_bold(node):
    """Первый значимый узел ``<p>``, если это ``<b>``/``<strong>``."""
    for child in node.contents:
        if isinstance(child, NavigableString) and not child.strip():
            continue
        return child if child.name in BOLD_TAGS else None
    return None


def promote_gost_structure(html_content: str) -> str:
    """Проставляет заголовки там, где разметка источника их не выражает.

    Зачем
    ----
    В DOCX заголовками являются титульные строки, пункты предисловия и
    термины — там их видно по стилю абзаца. В HTML/Markdown то же содержание
    лежит обычными абзацами (``<p class='org'>``, ``<li>`` предисловия,
    ``<b>3.1 …</b>``), и без разметки эти строки уезжают в чанки как обычный
    текст: ``parent_id`` у всех один, структура документа теряется.

    Что делаем (правила явные, по одному на вид ГОСТ-блока):

    1. ``<p class='org'|'country'|'num'>`` → ``<h1>`` — титульный блок;
    2. пункты ``<ol>`` сразу после заголовка «Предисловие» → ``<h1>`` с
       номером ``N`` (номер из текста рендерера отбрасывается — эталон пишет
       «1 ПОДГОТОВЛЕН», а не «1. ПОДГОТОВЛЕН»);
    3. абзац, начинающийся с ``<b>``/``<strong>`` и с многоточечным номером
       («3.1 …», «6.2.2 …»), → ``<h{глубина+1}>``: глубина номера и есть
       уровень заголовка;
    4. абзац «ОКС …» → ``<h1>`` (в DOCX он тоже заголовок).

    Чего правило НЕ делает: не трогает оглавление (``<ul>`` со строками
    «1 Область применения … 1» — их гасит ``header_promoter``) и не превращает
    в заголовки обычные абзацы. Пункты библиографии «[1] …» тоже не задеты:
    номер там в квадратных скобках, а правило 3 требует ``<b>`` в начале.
    """
    if not html_content or not html_content.strip():
        return html_content

    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception:  # noqa: BLE001 - битый HTML читаем как есть
        return html_content

    # 1. Титульный блок.
    for node in soup.find_all("p"):
        classes = set(node.get("class") or [])
        if classes & set(TITLE_PAGE_HEADER_CLASSES):
            node.name = "h1"

    # 2. Пункты предисловия.
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        title = _text_of(heading).casefold()
        if not any(title.startswith(word) for word in PREFACE_WORDS):
            continue
        ordered_list = heading.find_next("ol")
        if ordered_list is None:
            continue
        items = ordered_list.find_all("li", recursive=False)
        if not items:
            continue
        for index, item in enumerate(items, start=1):
            text = LIST_NUMBER_PREFIX_RE.sub("", _text_of(item), count=1)
            heading = soup.new_tag("h1")
            heading.string = f"{index} {text}"
            item.replace_with(heading)
        ordered_list.unwrap()

    # 3. Термины и подразделы, набранные полужирным в начале абзаца.
    for paragraph in soup.find_all("p"):
        if _leading_bold(paragraph) is None:
            continue
        match = DOTTED_NUMBER_RE.match(_text_of(paragraph))
        if not match:
            continue
        depth = len(match.group(1).split("."))
        paragraph.name = f"h{min(_MAX_HEADING_LEVEL, depth + 1)}"

    # 4. Строка классификации.
    for paragraph in soup.find_all("p"):
        if CLASSIFICATION_RE.match(_text_of(paragraph)):
            paragraph.name = "h1"

    # 5. Пункты списка, пронумерованные в самом тексте («[1] ISO/IEC …»), —
    #    абзацы, а не элементы списка (см. BRACKETED_ITEM_RE).
    for item in soup.find_all("li"):
        if not BRACKETED_ITEM_RE.match(_text_of(item)):
            continue
        item.name = "p"
        parent = item.parent
        if parent is not None and parent.name == "ol":
            parent.unwrap()

    return str(soup)


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