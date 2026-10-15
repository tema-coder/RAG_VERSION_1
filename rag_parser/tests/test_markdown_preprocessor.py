"""Тесты маршрута Markdown → HTML (parsers/markdown_preprocessor)."""

from pathlib import Path

import pytest

from rag_parser.parsers.markdown_preprocessor import (
    markdown_to_html,
    prepare_markdown_for_dedoc,
)

SAMPLE_MD = """\
# Заголовок первого уровня

Вводный текст.

## Раздел 5

| Показатель | Q1 | Q2 |
|---|---|---|
| Выручка | 100 | 120 |
| Прибыль | 30 | 40 |

### 5.1 Подраздел

| Регион | Доля |
|---|---|
| Север | 10% |
| Юг | 20% |
"""

INPUT = Path(__file__).resolve().parent.parent.parent / "input"


class TestMarkdownToHtml:
    def test_headings_become_h_tags(self):
        html = markdown_to_html(SAMPLE_MD)
        assert "<h1>Заголовок первого уровня</h1>" in html
        assert "<h2>Раздел 5</h2>" in html
        assert "<h3>5.1 Подраздел</h3>" in html

    def test_tables_become_html_tables(self):
        html = markdown_to_html(SAMPLE_MD)
        assert html.count("<table>") == 2
        assert "<th>Показатель</th>" in html
        assert "<td>Выручка</td>" in html

    def test_wrapped_in_html_document(self):
        html = markdown_to_html("просто текст")
        assert html.startswith("<html><body>")
        assert html.endswith("</body></html>")

    def test_empty_input_is_safe(self):
        assert markdown_to_html("") == "<html><body></body></html>"
        assert markdown_to_html("   \n  ") == "<html><body></body></html>"

    def test_raw_html_is_escaped(self):
        """Сырой HTML не должен попадать в dedoc как разметка."""
        html = markdown_to_html("текст <script>alert(1)</script> конец")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_pipe_line_without_separator_is_not_a_table(self):
        """Одиночная pipe-строка — обычный текст, не таблица."""
        html = markdown_to_html("| a | b |")
        assert "<table>" not in html


class TestPrepareMarkdownForDedoc:
    def test_writes_html_with_markers(self, tmp_path):
        source = tmp_path / "doc.md"
        source.write_text(SAMPLE_MD, encoding="utf-8")

        temp_path, tables = prepare_markdown_for_dedoc(source, tmp_path / "out")

        assert temp_path.exists() and temp_path.suffix == ".html"
        assert len(tables) == 2
        html = temp_path.read_text(encoding="utf-8")
        assert "<table>" not in html
        assert "__RAGTABLE_0__" in html and "__RAGTABLE_1__" in html

    def test_document_without_tables(self, tmp_path):
        source = tmp_path / "plain.md"
        source.write_text("# Только заголовок\n\nТекст.\n", encoding="utf-8")
        temp_path, tables = prepare_markdown_for_dedoc(source, tmp_path / "out")
        assert tables == []
        assert "<h1>Только заголовок</h1>" in temp_path.read_text(encoding="utf-8")

    def test_bom_is_stripped(self, tmp_path):
        source = tmp_path / "bom.md"
        source.write_bytes("﻿# Заголовок\n".encode("utf-8"))
        temp_path, _ = prepare_markdown_for_dedoc(source, tmp_path / "out")
        assert "﻿" not in temp_path.read_text(encoding="utf-8")

    def test_cp1251_is_decoded(self, tmp_path):
        source = tmp_path / "cp.md"
        source.write_bytes("# Заголовок\n\nТекст.\n".encode("cp1251"))
        temp_path, _ = prepare_markdown_for_dedoc(source, tmp_path / "out")
        assert "Заголовок" in temp_path.read_text(encoding="utf-8")


class TestMarkdownEndToEnd:
    """Сквозная проверка на реальном .md из input/."""

    def test_real_markdown_tables_and_hierarchy(self):
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.dedoc_tree import collect_headers

        source = INPUT / "gost_report.md"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        # 9 pipe-таблиц в файле — все должны быть привязаны, ни одна не orphan
        assert result.tree.metadata["tables_by_marker"] == 9
        assert result.tree.metadata["orphan_tables"] == 0

        headers = collect_headers(result.tree)
        assert len(headers) > 10
        # иерархия восстановлена: есть вложенные уровни
        assert {h.level for h in headers} >= {1, 2}

        # parent_id чанков распределён, а не схлопнут в один заголовок
        from rag_parser.chunkers.hybrid import HybridRagChunker

        chunks = HybridRagChunker(source="gost_report.md").chunk(
            result.markdown, {}, result.tree
        )
        parents = {c.metadata.get("parent_id") for c in chunks}
        assert len(parents) > 1

    def test_no_marker_leaks_into_markdown(self):
        from rag_parser.parsers.dedoc_parser import DedocParser

        source = INPUT / "gost_report.md"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert "__RAGTABLE_" not in result.markdown

class TestBoldSurvivesHtmlReader:
    """Жирный текст должен дойти до классификатора.

    ``HtmlReader`` извлекает только текст, поэтому ``<strong>``/
    ``<b>`` заменяются на текстовый маркер ``__BOLD__`` ДО разбора.
    """

    def test_bold_replaced_with_marker(self):
        html = markdown_to_html("**Жирный** и *курсив*")
        assert "__BOLD__Жирный__BOLD__" in html
        assert "<strong>" not in html

    def test_b_tag_also_replaced(self):
        html = markdown_to_html("текст")
        # mistune даёт <strong>; <b> обрабатывается тем же кодом
        assert "<b>" not in html

    def test_plain_text_untouched(self):
        html = markdown_to_html("обычный текст без жирного")
        assert "__BOLD__" not in html


class TestBoldEndToEnd:
    """Триггер жирного работает там, где включена промоция."""

    def test_txt_bold_becomes_nested_header(self, tmp_path):
        import logging

        from rag_parser.parsers.dedoc_parser import DedocParser

        logging.disable(logging.CRITICAL)
        source = tmp_path / "bold.txt"
        source.write_text(
            "1 ВВЕДЕНИЕ\n\nТекст.\n\n**Жирный подзаголовок**\n\n"
            "Ещё текст.\n\n***Глубже***\n\nФинал.\n",
            encoding="utf-8",
        )
        result = DedocParser().parse(source)

        headers = []
        stack = [result.tree]
        while stack:
            node = stack.pop()
            if node.type == "header":
                headers.append((node.text.strip(), node.level))
            stack.extend(node.children)

        titles = {text: level for text, level in headers}
        assert titles["Жирный подзаголовок"] == 2
        assert titles["Глубже"] == 3
        # служебные маркеры не должны попасть в итоговый Markdown
        assert "**" not in result.markdown
        assert "__BOLD__" not in result.markdown

    def test_markdown_uses_authoritative_markup(self):
        """``.md``: разметка источника важнее триггеров — промоции нет."""
        import logging

        from rag_parser.parsers.dedoc_parser import DedocParser

        logging.disable(logging.CRITICAL)
        source = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.md"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert "__BOLD__" not in result.markdown
