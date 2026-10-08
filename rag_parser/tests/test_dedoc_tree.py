"""Unit-тесты DedocTree на моках ровно тех структур, что описаны в «граблях»."""

from rag_parser.parsers import dedoc_tree as tree_mod
from rag_parser.parsers.dedoc_tree import (
    build_full_tree,
    build_preview_tree,
    cell_to_text,
    collect_headers,
    extract_line_text,
    extract_paragraph_info,
    full_tree_to_markdown,
    iter_nodes,
    truncate,
)


class FakeHierarchy:
    def __init__(self, level_1=None, level_2=None, line_type="header"):
        self.level_1 = level_1
        self.level_2 = level_2
        self.line_type = line_type


class FakeMeta:
    """Метаданные внутренней структуры: .data с paragraph_type/hierarchy_level."""

    def __init__(self, paragraph_type, hierarchy=None):
        self.data = {"paragraph_type": paragraph_type}
        if hierarchy is not None:
            self.data["hierarchy_level"] = hierarchy


class FakeLine:
    """LineWithMeta: текст в .line, вложенность — тоже FakeLine."""

    def __init__(self, line, metadata=None, annotations=None):
        self.line = line
        self.metadata = metadata or FakeMeta("raw_text")
        self.annotations = annotations or []

    # NB: атрибута .text НЕТ — как у настоящего LineWithMeta.


class FakeCell:
    """CellWithMeta: НЕТ атрибута .text; текст через get_text()/lines."""

    def __init__(self, lines, with_get_text=True):
        self.lines = lines
        self.colspan = 1
        self.rowspan = 1
        self.invisible = False
        if with_get_text:
            self.get_text = lambda: "\n".join(
                extract_line_text(ln) for ln in self.lines
            )


def _api_node(text, paragraph_type="raw_text", children=None, annotations=None):
    return {
        "text": text,
        "metadata": {"paragraph_type": paragraph_type},
        "annotations": annotations or [],
        "subparagraphs": children or [],
    }


class TestExtractLineText:
    def test_plain_string(self):
        assert extract_line_text("hello") == "hello"

    def test_line_attr(self):
        assert extract_line_text(FakeLine("inner")) == "inner"

    def test_nested_line(self):
        assert extract_line_text(FakeLine(FakeLine("deep"))) == "deep"

    def test_api_dict_text(self):
        assert extract_line_text({"text": "api text"}) == "api text"

    def test_internal_dict_line(self):
        assert extract_line_text({"line": "inner"}) == "inner"
        assert extract_line_text({"line": {"line": "deep"}}) == "deep"


class TestExtractParagraphInfo:
    def test_internal_meta_data(self):
        meta = FakeMeta("header", FakeHierarchy(1, 2, "header"))
        paragraph_type, hierarchy = extract_paragraph_info(meta)
        assert paragraph_type == "header"
        assert hierarchy == {"level_1": 1, "level_2": 2, "line_type": "header"}

    def test_api_flat_meta(self):
        paragraph_type, hierarchy = extract_paragraph_info({"paragraph_type": "header"})
        assert paragraph_type == "header"
        assert hierarchy is None

    def test_no_type_line_attrs(self):
        assert not hasattr(FakeLine("x"), "type")
        assert not hasattr(FakeLine("x"), "level")


class TestCellText:
    def test_get_text_preferred(self):
        cell = FakeCell([FakeLine("A"), FakeLine("B")], with_get_text=True)
        assert cell_to_text(cell) == "A\nB"

    def test_lines_fallback_without_text_attr(self):
        cell = FakeCell([FakeLine("A"), FakeLine("B")], with_get_text=False)
        assert not hasattr(cell, "text")
        assert cell_to_text(cell) == "A\nB"

    def test_str_repr_is_ignored(self):
        assert cell_to_text("plain") == "plain"
        # мусор вида CellWithMeta(cs=1, rs=1, ...) никогда не возвращается
        assert "CellWithMeta(" not in cell_to_text(FakeCell([FakeLine("v")]))


class TestBuildTree:
    def _doc(self):
        return {
            "text": "",
            "metadata": {"paragraph_type": "root"},
            "annotations": [],
            "subparagraphs": [
                _api_node("Раздел 1. Общие сведения", "header", [
                    _api_node("Первый абзац тела.", "raw_text"),
                    _api_node("1.1 Метрики", "header", [
                        _api_node("Текст метрик.", "raw_text"),
                    ]),
                ]),
                _api_node("Обычный абзац в корне.", "paragraph"),
            ],
        }

    def test_hierarchy(self):
        root = build_full_tree(self._doc(), [])
        assert root.type == "document"
        headers = collect_headers(root)
        assert [h.text for h in headers] == ["Раздел 1. Общие сведения", "1.1 Метрики"]
        assert headers[0].level == 1
        assert headers[1].level == 2
        assert headers[1] in headers[0].children

    def test_heading_normalized_to_header(self):
        doc = {"text": "", "metadata": {"paragraph_type": "root"},
               "annotations": [], "subparagraphs": [_api_node("Title", "heading")]}
        headers = collect_headers(build_full_tree(doc, []))
        assert len(headers) == 1 and headers[0].type == "header"

    def test_tables_become_nodes(self):
        tables = [{"cells": [[{"lines": [{"text": "A"}]}, {"lines": [{"text": "B"}]}]]}]
        root = build_full_tree(self._doc(), tables)
        table_nodes = [n for n in root.children if n.type == "table"]
        assert len(table_nodes) == 1
        assert "| A | B |" in table_nodes[0].text

    def test_internal_objects(self):
        node = FakeLine(
            "Заголовок",
            FakeMeta("header", FakeHierarchy(1, 1, "header")),
            annotations=[{"name": "bold", "start": 0, "end": 8, "value": True}],
        )
        node_dict = {
            "line": node,
            "metadata": node.metadata,
            "annotations": [{"name": "bold", "start": 0, "end": 8, "value": True}],
            "subparagraphs": [],
        }
        root = build_full_tree(
            {"text": "", "metadata": {"paragraph_type": "root"},
             "annotations": [], "subparagraphs": [node_dict]}, [],
        )
        headers = collect_headers(root)
        assert len(headers) == 1
        assert headers[0].metadata["annotations_count"] == 1

    def test_truncate_last_step(self):
        root = build_full_tree(self._doc(), [])
        preview = truncate(root, 30)
        for node in [preview, *preview.children]:
            assert len(node.text) <= 31
            assert "annotations" not in node.metadata
            assert "annotations_count" in node.metadata
        # полное дерево не урезано
        assert root.children[0].text == "Раздел 1. Общие сведения"

    def test_build_preview_tree(self):
        preview = build_preview_tree(self._doc(), [], preview_len=30)
        assert preview.children[0].text == "Раздел 1. Общие сведения"

    def test_markdown_from_full_tree(self):
        markdown = full_tree_to_markdown(build_full_tree(self._doc(), []))
        assert "# Раздел 1. Общие сведения" in markdown
        assert "## 1.1 Метрики" in markdown
        assert "Первый абзац тела." in markdown
        assert "\r" not in markdown
        assert "\x0c" not in markdown


class TestModuleSurface:
    def test_no_str_cell_reliance(self):
        import inspect

        source = inspect.getsource(tree_mod)
        assert "str(cell)" not in source


def _struct_node(text, annotations=None, children=None, paragraph_type="raw_text"):
    return {
        "text": text,
        "metadata": {"paragraph_type": paragraph_type},
        "annotations": annotations or [],
        "subparagraphs": children or [],
    }


def _api_table(uid, rows):
    cells = [[{"lines": [{"text": value}], "colspan": 1, "rowspan": 1,
               "invisible": False} for value in row] for row in rows]
    return {"cells": cells, "metadata": {"uid": uid, "page_id": 0,
                                         "rotated_angle": 0.0, "title": ""}}


class TestTableAttach:
    def _doc(self):
        return {
            "text": "", "metadata": {"paragraph_type": "root"},
            "annotations": [], "subparagraphs": [
                _struct_node("Раздел A. Первый"),
                _struct_node(
                    "Абзац перед таблицей.",
                    annotations=[{"name": "table", "start": 0, "end": 5,
                                  "value": "uid-1"}],
                ),
                _struct_node("Раздел B. Второй"),
            ],
        }

    def test_attached_to_anchor_first_child(self):
        root = build_full_tree(
            self._doc(), [_api_table("uid-1", [["A", "B"], ["1", "2"]])],
        )
        assert root.metadata["orphan_tables"] == 0
        anchor = next(n for n in root.children if n.text == "Абзац перед таблицей.")
        assert anchor.children and anchor.children[0].type == "table"
        assert "| A | B |" in anchor.children[0].text

    def test_markdown_order_anchor_table(self):
        root = build_full_tree(
            self._doc(), [_api_table("uid-1", [["A", "B"], ["1", "2"]])],
        )
        markdown = full_tree_to_markdown(root)
        pos_a = markdown.index("Абзац перед таблицей.")
        pos_t = markdown.index("| A | B |")
        pos_b = markdown.index("Раздел B.")
        assert pos_a < pos_t < pos_b  # таблица между A-якорем и B, не в конце

    def test_unknown_uid_falls_back(self):
        root = build_full_tree(
            self._doc(), [_api_table("uid-zzz", [["A"]])],
        )
        assert root.metadata["orphan_tables"] == 1
        assert root.children[-1].type == "table"

    def test_eight_tables_attached_and_orphan(self, caplog):
        import logging

        tables = ([_api_table(f"uid-{i}", [[f"H{i}"]]) for i in range(4)]
                  + [_api_table(None, [[f"O{i}"]]) for i in range(4)])
        doc = self._doc()
        for i in range(4):
            doc["subparagraphs"].append(
                _struct_node(
                    f"Якорь {i}.",
                    annotations=[{"name": "table", "start": 0, "end": 3,
                                  "value": f"uid-{i}"}],
                )
            )
        with caplog.at_level(logging.WARNING):
            root = build_full_tree(doc, tables)
        assert root.metadata["orphan_tables"] == 4
        assert sum(1 for _ in iter_nodes(root) if _.type == "table") == 8
        assert any("orphan_tables" in r.message for r in caplog.records)


class TestObjectShapes:
    """Покрытие объектных (не dict) ветвей нормализации."""

    def test_object_tree_with_children_attr(self):
        header = FakeLine(
            "Глава", FakeMeta("header", FakeHierarchy(1, 3, "header")),
        )
        header.subparagraphs = [FakeLine("Тело", FakeMeta("raw_text"))]
        header.annotations = []
        root = build_full_tree(
            {"line": "", "metadata": FakeMeta("root"), "annotations": [],
             "subparagraphs": [header]}, [],
        )
        headers = collect_headers(root)
        assert len(headers) == 1
        assert headers[0].level == 3  # level_2 приоритетнее
        assert headers[0].children[0].text == "Тело"

    def test_list_structure_input(self):
        root = build_full_tree([_api_node("A", "header"), _api_node("b", "raw_text")], [])
        assert root.type == "document"
        assert collect_headers(root)[0].text == "A"

    def test_line_text_object_and_list(self):
        class ApiLine:
            def __init__(self, text):
                self.text = text

        assert extract_line_text(ApiLine("t")) == "t"
        assert extract_line_text(["a", "", "b"]) == "a\nb"

    def test_annotations_object_form(self):
        class Ann:
            name = "bold"
            start = 0
            end = 5
            value = True

        class Node:
            text = "hello"
            metadata = {"paragraph_type": "paragraph"}
            annotations = [Ann()]
            subparagraphs = []

        root = build_full_tree(
            {"text": "", "metadata": {"paragraph_type": "root"},
             "annotations": [], "subparagraphs": [Node()]}, [],
        )
        assert root.children[0].metadata["annotations_count"] == 1

    def test_markdown_list_and_table(self):
        root = build_full_tree({
            "text": "", "metadata": {"paragraph_type": "root"}, "annotations": [],
            "subparagraphs": [
                _api_node("пункт", "list_item"),
                _api_node("- готовый пункт", "list_item"),
                _api_node("сырой текст", "raw_text"),
            ],
        }, [{"cells": [[{"lines": [{"text": "A"}]}]]}])
        markdown = full_tree_to_markdown(root)
        assert "- пункт" in markdown
        assert "- готовый пункт" in markdown
        assert "| A |" in markdown

    def test_truncate_exact_length(self):
        from rag_parser.core.interfaces import DocumentNode

        node = DocumentNode(type="paragraph", level=0, text="x" * 30,
                            children=[], metadata={})
        assert truncate(node, 30).text == "x" * 30  # без "…" на границе

    def test_hierarchy_dict_line_type_fallback(self):
        paragraph_type, _ = extract_paragraph_info(
            {"data": {"hierarchy_level": {"level_1": 1, "level_2": 2,
                                          "line_type": "header"}}}
        )
        assert paragraph_type == "header"
