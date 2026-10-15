"""Тесты маркерного маршрута HTML (parsers/html_preprocessor).

Задача маршрута: ``HtmlReader`` в dedoc 2.8 не создаёт ``TableAnnotation``,
поэтому таблицы теряют позицию. Мы извлекаем их сами (в Markdown) и
оставляем в HTML маркер-абзац, по которому потом восстанавливаем
привязку к ближайшему заголовку выше.
"""

import re
from pathlib import Path

import pytest

from rag_parser.core.interfaces import DocumentNode
from rag_parser.parsers.dedoc_tree import attach_tables_by_position
from rag_parser.parsers.html_preprocessor import (
    MARKER_PREFIX,
    extract_and_attach_tables,
    inject_table_markers,
    strip_markers,
)


def _node(node_type, text, children=None, level=1):
    return DocumentNode(
        type=node_type,
        level=level,
        text=text,
        children=children or [],
        metadata={},
    )


class TestInjectMarkers:
    def test_inject_markers_simple(self):
        """Одна таблица -> один маркер."""
        html = "<html><body><table><tr><th>A</th></tr><tr><td>1</td></tr></table></body></html>"
        marked, tables = inject_table_markers(html)
        assert len(tables) == 1
        assert marked.count(f"{MARKER_PREFIX}0__") == 1
        assert "<table>" not in marked

    def test_inject_markers_multiple(self):
        """Три таблицы -> маркеры с индексами 0, 1, 2."""
        cell = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        marked, tables = inject_table_markers(f"<html><body>{cell}{cell}{cell}</body></html>")
        assert len(tables) == 3
        for index in (0, 1, 2):
            assert marked.count(f"{MARKER_PREFIX}{index}__") == 1

    def test_markers_are_own_paragraphs(self):
        """Маркер — отдельный <p>, а не хвост предыдущего тега."""
        html = '<html><body><h1>З</h1><table><tr><td>1</td></tr></table></body></html>'
        marked, _ = inject_table_markers(html)
        assert f"<p>{MARKER_PREFIX}0__</p>" in marked

    def test_no_tables(self):
        """Нет таблиц -> маркеры не добавляются, html не меняется."""
        html = "<html><body><h1>Только текст</h1></body></html>"
        marked, tables = inject_table_markers(html)
        assert tables == []
        assert MARKER_PREFIX not in marked

    def test_table_rows_extracted(self):
        html = (
            "<table><tr><th>Регион</th><th>Q1</th></tr>"
            "<tr><td>Север</td><td>100</td></tr></table>"
        )
        _, tables = inject_table_markers(html)
        assert len(tables) == 1
        assert "| Регион | Q1 |" in tables[0]
        assert "| Север | 100 |" in tables[0]

    def test_empty_html_is_safe(self):
        assert inject_table_markers("") == ("", [])

    def test_empty_table_skipped(self):
        """Таблица без непустых ячеек не даёт маркера."""
        html = "<html><body><table></table></body></html>"
        marked, tables = inject_table_markers(html)
        assert tables == []


class TestStripMarkers:
    def test_strip_plain_marker(self):
        cleaned, found = strip_markers("__RAGTABLE_3__")
        assert cleaned == ""
        assert found == [3]

    def test_strip_marker_stuck_to_text(self):
        """Маркер может склеиться с соседним текстом."""
        cleaned, found = strip_markers("Ещё.\n__RAGTABLE_1__\nПосле.")
        assert found == [1]
        assert MARKER_PREFIX not in cleaned
        assert "Ещё." in cleaned and "После." in cleaned

    def test_strip_multiple_markers(self):
        cleaned, found = strip_markers("__RAGTABLE_0__ и __RAGTABLE_4__")
        assert sorted(found) == [0, 4]
        assert MARKER_PREFIX not in cleaned

    def test_strip_without_markers(self):
        cleaned, found = strip_markers("обычный текст")
        assert cleaned == "обычный текст"
        assert found == []


class TestExtractAndAttach:
    def test_extract_and_attach(self):
        """Маркер находит ближайший заголовок выше."""
        tree = _node(
            "document", "", level=0,
            children=[
                _node("header", "Раздел 1", [_node("paragraph", f"{MARKER_PREFIX}0__")]),
            ],
        )
        attached, orphans = extract_and_attach_tables(tree, ["| A |\n|---|\n| 1 |"])
        assert attached == 1 and orphans == 0
        header = tree.children[0]
        assert header.children[0].type == "table"
        assert header.children[0].text.startswith("| A |")

    def test_marker_stuck_to_text_is_found_and_removed(self):
        tree = _node(
            "document", "", level=0,
            children=[
                _node("header", "Раздел", [
                    _node("paragraph", f"Текст до.\n{MARKER_PREFIX}0__"),
                ]),
            ],
        )
        attached, orphans = extract_and_attach_tables(tree, ["| A |\n|---|\n| 1 |"])
        assert attached == 1 and orphans == 0
        paragraph = tree.children[0].children[0]
        assert MARKER_PREFIX not in paragraph.text
        assert "Текст до." in paragraph.text

    def test_orphan_table_before_first_header(self):
        """Таблица выше первого заголовка остаётся orphan."""
        tree = _node(
            "document", "", level=0,
            children=[
                _node("paragraph", f"{MARKER_PREFIX}0__"),
                _node("header", "Раздел"),
            ],
        )
        attached, orphans = extract_and_attach_tables(tree, ["| A |\n|---|\n| 1 |"])
        assert attached == 0 and orphans == 1
        assert any(child.type == "table" for child in tree.children)

    def test_nearest_header_wins(self):
        """Берётся ПОСЛЕДНИЙ заголовок выше маркера."""
        tree = _node(
            "document", "", level=0,
            children=[
                _node("header", "Первый", [_node("header", "Второй", [
                    _node("paragraph", f"{MARKER_PREFIX}0__"),
                ])]),
            ],
        )
        extract_and_attach_tables(tree, ["| A |\n|---|\n| 1 |"])
        second = tree.children[0].children[0]
        assert second.text == "Второй"
        assert second.children[0].type == "table"

    def test_marker_out_of_range_ignored(self):
        """Индекс за пределами списка таблиц не роняет разбор."""
        tree = _node(
            "document", "", level=0,
            children=[_node("header", "Р", [_node("paragraph", f"{MARKER_PREFIX}9__")])],
        )
        attached, orphans = extract_and_attach_tables(tree, ["| A |"])
        assert attached == 0 and orphans == 0

    def test_empty_table_list_is_noop(self):
        tree = _node("document", "", level=0, children=[_node("header", "Р")])
        attached, orphans = extract_and_attach_tables(tree, [])
        assert attached == 0 and orphans == 0

    def test_every_marker_in_a_section_is_attached(self):
        """РЕГРЕССИЯ: маркеры подряд не должны теряться.

        Пустые абзацы-маркеры выбрасываются, а раньше это делалось
        удалением из ``parent.children`` прямо во время обхода того же
        списка. Список сдвигался, и следующий маркер не обрабатывался —
        таблица молча исчезала (gost_report.md отдавал 8 таблиц из 9).
        """
        tables = [f"| T{i} |\n|---|\n| {i} |" for i in range(4)]
        section = _node(
            "header", "Раздел",
            [_node("raw_text", f"{MARKER_PREFIX}{i}__") for i in range(4)],
        )
        tree = _node("document", "", level=0, children=[section])

        attached, orphans = extract_and_attach_tables(tree, tables)
        assert attached == 4
        assert orphans == 0
        assert len([c for c in section.children if c.type == "table"]) == 4
        # пустые абзацы-маркеры убраны
        assert all(child.type == "table" for child in section.children)

    def test_markers_in_nested_sections_all_attached(self):
        """Три маркера в трёх вложенных разделах — все три привязаны."""
        tables = [f"| T{i} |\n|---|\n| {i} |" for i in range(3)]
        inner = _node("header", "6.2", [_node("raw_text", f"{MARKER_PREFIX}2__")])
        middle = _node(
            "header", "6", [_node("raw_text", f"{MARKER_PREFIX}1__"), inner]
        )
        tree = _node(
            "document", "", level=0,
            children=[
                _node("raw_text", f"{MARKER_PREFIX}0__"),
                middle,
            ],
        )
        attached, orphans = extract_and_attach_tables(tree, tables)
        assert attached == 2
        assert orphans == 1
        # Таблица занимает место маркера: в «6» маркер шёл ДО вложенного
        # заголовка, поэтому таблица тоже стоит до него. Дописывание её в
        # конец детей вынесло бы её за пределы поддерева 5.1/5.2.
        assert middle.children[0].type == "table"
        assert middle.children[1] is inner
        assert inner.children[0].type == "table"
        assert any(c.type == "table" for c in tree.children)

    def test_non_empty_paragraphs_are_preserved(self):
        """Режистектура не должна съедать обычный текст."""
        tree = _node(
            "document", "", level=0,
            children=[
                _node("header", "Р", [
                    _node("paragraph", "текст до"),
                    _node("raw_text", f"{MARKER_PREFIX}0__"),
                    _node("paragraph", "текст после"),
                ]),
            ],
        )
        extract_and_attach_tables(tree, ["| A |\n|---|\n| 1 |"])
        texts = [c.text for c in tree.children[0].children]
        assert "текст до" in texts and "текст после" in texts
        assert MARKER_PREFIX not in "".join(texts)


class TestAttachTablesByPosition:
    def test_wrapper_updates_metadata(self):
        tree = _node(
            "document", "", level=0,
            children=[
                _node("header", "Раздел", [_node("paragraph", f"{MARKER_PREFIX}0__")]),
            ],
        )
        orphans = attach_tables_by_position(tree, ["| A |\n|---|\n| 1 |"])
        assert orphans == 0
        assert tree.metadata["tables_by_marker"] == 1
        assert tree.metadata["orphan_tables"] == 0

    def test_wrapper_counts_orphans(self):
        tree = _node(
            "document", "", level=0,
            children=[_node("paragraph", f"{MARKER_PREFIX}0__")],
        )
        orphans = attach_tables_by_position(tree, ["| A |\n|---|\n| 1 |"])
        assert orphans == 1

    def test_wrapper_without_tables(self):
        tree = _node("document", "", level=0, children=[_node("header", "Р")])
        assert attach_tables_by_position(tree, []) == 0


class TestRealHtmlFile:
    """Сквозная проверка на реальном файле из input/."""

    INPUT = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent.parent / "input"

    def test_real_html_tables_attached(self, tmp_path, monkeypatch):
        from rag_parser.core.config import settings
        from rag_parser.parsers.dedoc_parser import DedocParser

        source = self.INPUT / "простая_html.html"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "out")
        )
        monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "temp"))
        result = DedocParser().parse(source)
        assert result.tree.metadata["tables_by_marker"] == 2
        assert result.tree.metadata["orphan_tables"] == 0

    def test_real_html_tables_not_duplicated_in_text(self):
        """Текст таблиц не дублируется в абзацах (причина ручного
        извлечения вместо rely на dedoc)."""
        from rag_parser.parsers.dedoc_parser import DedocParser

        source = self.INPUT / "простая_html.html"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        texts = []

        def walk(node):
            texts.append(node.text)
            for child in node.children:
                walk(child)

        walk(result.tree)
        for text in texts:
            if node_is_table_text(text):
                pytest.fail(f"таблица продублирована текстом: {text[:60]!r}")


def node_is_table_text(text: str) -> bool:
    """Эвристика: строка абзаца с разобранными ячейками таблицы."""
    if not text or text.lstrip().startswith("|"):
        return False
    return "\t" in text and len(text.split("\t")) >= 2


#: Все форматы gost_report: у каждого ровно 9 таблиц.
MARKER_ROUTE_FORMATS = ("gost_report.md", "gost_report.html", "gost_report.txt",
                        "gost_report.xml")


def _table_nodes(node):
    """Все узлы-таблицы поддерева (рекурсивно)."""
    found = []
    if node.type == "table":
        found.append(node)
    for child in node.children:
        found.extend(_table_nodes(child))
    return found


def _section(node, number):
    """Заголовок раздела с номером ``number``.

    Строки оглавления («5.2 Второе положение … 5») в ``.txt`` сами имеют
    type="header", поэтому их отсекаем по многоточию — иначе под раздел
    попадёт строка TOC без таблицы.
    """
    if node.type == "header":
        text = (node.text or "").strip()
        if text.startswith(number) and "…" not in text:
            return node
    for child in node.children:
        found = _section(child, number)
        if found is not None:
            return found
    return None


def _table_anchors(node, owner=None):
    """Номера разделов, к которым привязаны таблицы (ближайший заголовок).

    Универсально для обоих маршрутов: в HTML-маршруте таблица — прямой
    ребёнок заголовка, а в DOCX ``_attach_tables`` вставляет её первым
    ребёнком узла-якоря (обычно абзаца) ВНУТРИ раздела.
    """
    if node.type == "header" and (node.text or "").strip() and "…" not in node.text:
        owner = node.text.strip()
    if node.type == "table":
        match = re.match(r"^(\d+(?:\.\d+)*)", owner or "")
        return [match.group(1) if match else None]
    found = []
    for child in node.children:
        found.extend(_table_anchors(child, owner))
    return found


#: Раздел -> сколько таблиц в нём должно быть (эталон gost_report).
EXPECTED_ANCHORS = ["1", "5", "5.1", "5.2", "6.1", "6.2.1", "6.2.2", "6.2.3", "7"]


class TestTablePlacementRegression:
    """РЕГРЕССИЯ: таблица раздела с вложенными подразделами.

    Таблица дописывалась в конец детей раздела. У раздела «5» есть своя
    таблица и вложенные «5.1»/«5.2», поэтому таблица выводилась уже ПОСЛЕ
    поддерева 5.2 — в Markdown она выглядела как вторая таблица раздела
    5.2, а сам раздел 5 оставался без таблицы (всего 9, а не 10).
    """

    def _document(self, path):
        from rag_parser.parsers.dedoc_parser import DedocParser

        return DedocParser().parse(path).tree

    @pytest.mark.parametrize("filename", MARKER_ROUTE_FORMATS)
    def test_table_not_duplicated_in_section_5_2(self, filename):
        """В разделе 5.2 ровно одна таблица."""
        path = Path(__file__).resolve().parent.parent.parent / "input" / filename
        if not path.exists():
            pytest.skip(f"{filename} отсутствует в input/")

        tree = self._document(path)
        section_52 = _section(tree, "5.2")
        assert section_52 is not None, "раздел 5.2 не найден"
        assert len(_table_nodes(section_52)) == 1

    @pytest.mark.parametrize("filename", MARKER_ROUTE_FORMATS)
    def test_total_table_count(self, filename):
        """Во всём документе ровно 9 таблиц."""
        path = Path(__file__).resolve().parent.parent.parent / "input" / filename
        if not path.exists():
            pytest.skip(f"{filename} отсутствует в input/")

        tree = self._document(path)
        assert len(_table_nodes(tree)) == 9

    @pytest.mark.parametrize("filename", MARKER_ROUTE_FORMATS + ("gost_report.docx",))
    def test_every_table_in_its_own_section(self, filename):
        """Каждая таблица привязана ровно к своему разделу."""
        path = Path(__file__).resolve().parent.parent.parent / "input" / filename
        if not path.exists():
            pytest.skip(f"{filename} отсутствует в input/")

        tree = self._document(path)
        assert len(_table_nodes(tree)) == 9
        assert tree.metadata.get("orphan_tables", 0) == 0
        # Заголовок 5 сохранил СВОЮ таблицу (регрессия: она уезжала в 5.2).
        assert _table_anchors(tree) == EXPECTED_ANCHORS

    @pytest.mark.parametrize("filename", MARKER_ROUTE_FORMATS)
    def test_parent_section_table_precedes_subsections(self, filename):
        """Таблица раздела 5 стоит ПЕРЕД вложенными 5.1/5.2."""
        path = Path(__file__).resolve().parent.parent.parent / "input" / filename
        if not path.exists():
            pytest.skip(f"{filename} отсутствует в input/")

        section_5 = _section(self._document(path), "5 Общие") or _section(
            self._document(path), "5 ОБЩИЕ"
        )
        children = section_5.children
        first_table = next(i for i, c in enumerate(children) if c.type == "table")
        first_subsection = next(
            (i for i, c in enumerate(children) if c.type == "header"), len(children)
        )
        assert first_table < first_subsection, (
            "таблица раздела 5 обязана идти до вложенных подразделов, "
            "иначе она рендерится внутри последнего из них"
        )

    @pytest.mark.parametrize("filename", MARKER_ROUTE_FORMATS)
    def test_no_marker_text_left(self, filename):
        """Маркер не должен остаться в тексте ни одного узла."""
        path = Path(__file__).resolve().parent.parent.parent / "input" / filename
        if not path.exists():
            pytest.skip(f"{filename} отсутствует в input/")

        from rag_parser.parsers.dedoc_parser import DedocParser

        result = DedocParser().parse(path)
        assert MARKER_PREFIX not in result.markdown

        def walk(node):
            assert MARKER_PREFIX not in (node.text or ""), node.text[:60]
            for child in node.children:
                walk(child)

        walk(result.tree)

    def test_docx_route_not_degraded(self):
        """DOCX идёт по якорям TableAnnotation, а не по маркерам."""
        path = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.docx"
        if not path.exists():
            pytest.skip("файл отсутствует в input/")

        tree = self._document(path)
        assert len(_table_nodes(tree)) == 9
        assert tree.metadata.get("orphan_tables", 0) == 0
        assert tree.metadata.get("tables_by_marker") is None