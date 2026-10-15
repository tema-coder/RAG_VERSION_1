"""Тесты детектора ASCII/Markdown-таблиц в .txt (parsers/text_preprocessor)."""

from pathlib import Path

import pytest

from rag_parser.parsers.text_preprocessor import (
    find_table_blocks,
    prepare_text_for_dedoc,
    text_to_html,
)

GOST_ASCII_TABLE = """\
Предисловие
------------------------------------------------------------

1 Область применения

Текст перед таблицей.

  Показатель | Q1 | Q2 | Q3 | Q4
  ------------------------------------------------------------
  Выручка, тыс. руб. | 1000 | 1100 | 1200 | 1300
  Себестоимость, тыс. руб. | 700 | 750 | 800 | 850

Текст после таблицы.
"""

MD_TABLE = """\
## Раздел

| Показатель | Q1 |
|---|---|
| Выручка | 100 |
"""

INPUT = Path(__file__).resolve().parent.parent.parent / "input"


class TestFindTableBlocks:
    def test_gost_ascii_table_found(self):
        lines = GOST_ASCII_TABLE.split("\n")
        blocks = find_table_blocks(lines)
        assert len(blocks) == 1
        start, end, rows = blocks[0]
        assert len(rows) == 3
        assert rows[0] == ["Показатель", "Q1", "Q2", "Q3", "Q4"]

    def test_heading_underline_is_not_a_table(self):
        """Полоса дефисов под заголовком — не таблица."""
        assert find_table_blocks(GOST_ASCII_TABLE.split("\n")[0:2]) == []

    def test_markdown_table_found(self):
        blocks = find_table_blocks(MD_TABLE.split("\n"))
        assert len(blocks) == 1
        assert blocks[0][2] == [["Показатель", "Q1"], ["Выручка", "100"]]

    def test_two_tables(self):
        text = GOST_ASCII_TABLE + "\n" + GOST_ASCII_TABLE
        assert len(find_table_blocks(text.split("\n"))) == 2

    def test_single_pipe_is_not_a_table(self):
        assert find_table_blocks(["условие | иначе", "------", "строка"]) == []

    def test_header_without_rows_is_not_a_table(self):
        """Заголовок без строк данных — не таблица."""
        assert find_table_blocks(["A | B", "---", ""]) == []

    def test_plain_text(self):
        assert find_table_blocks(["Просто", "текст", "без таблиц"]) == []

    def test_ragged_rows_are_kept(self):
        lines = ["A | B | C", "---", "1 | 2"]
        blocks = find_table_blocks(lines)
        assert blocks[0][2] == [["A", "B", "C"], ["1", "2"]]


class TestTextToHtml:
    def test_table_becomes_html(self):
        html = text_to_html(GOST_ASCII_TABLE)
        assert "<table>" in html
        assert "<th>Показатель</th>" in html
        assert "<td>Выручка, тыс. руб.</td>" in html

    def test_surrounding_text_preserved(self):
        html = text_to_html(GOST_ASCII_TABLE)
        assert "Текст перед таблицей." in html
        assert "Текст после таблицы." in html

    def test_empty_input(self):
        assert text_to_html("") == "<html><body></body></html>"
        assert text_to_html("  \n ") == "<html><body></body></html>"

    def test_special_chars_are_escaped(self):
        html = text_to_html("A & B <script>alert(1)</script>")
        assert "<script>" not in html
        assert "&amp;" in html

    def test_ragged_row_is_padded(self):
        html = text_to_html("A | B | C\n---\n1 | 2")
        assert html.count("<td>") == 3


class TestPrepareTextForDedoc:
    def test_writes_html_with_markers(self, tmp_path):
        source = tmp_path / "doc.txt"
        source.write_text(GOST_ASCII_TABLE, encoding="utf-8")
        temp_path, tables = prepare_text_for_dedoc(source, tmp_path / "out")
        assert temp_path is not None
        assert len(tables) == 1
        assert "__RAGTABLE_0__" in temp_path.read_text(encoding="utf-8")

    def test_text_without_tables_is_left_to_reader(self, tmp_path):
        """Без таблиц нативный RawTextReader лучше сохраняет переносы."""
        source = tmp_path / "plain.txt"
        source.write_text("Просто текст\n\nЕщё абзац.\n", encoding="utf-8")
        temp_path, tables = prepare_text_for_dedoc(source, tmp_path / "out")
        assert temp_path is None and tables == []


class TestTextEndToEnd:
    """Сквозная проверка на реальном .txt из input/."""

    def test_real_txt_tables_and_headers(self):
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.dedoc_tree import collect_headers

        source = INPUT / "gost_report.txt"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert result.tree.metadata["tables_by_marker"] == 9
        assert result.tree.metadata["orphan_tables"] == 0
        # заголовки в .txt по-прежнему распознаёт header_promoter
        assert len(collect_headers(result.tree)) > 10

    def test_no_marker_leaks_into_markdown(self):
        from rag_parser.parsers.dedoc_parser import DedocParser

        source = INPUT / "gost_report.txt"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert "__RAGTABLE_" not in result.markdown