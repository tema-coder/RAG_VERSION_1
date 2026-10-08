"""Покрытие хелперов parsers/base.py (файл по ТЗ — без изменений)."""

import pytest

from rag_parser.parsers.base import (
    build_result,
    decode_text,
    detect_dialect,
    escape_cell,
    normalize_row,
    parse_csv_rows,
    read_bytes,
    read_text_file,
    rows_to_markdown_table,
)


class TestReadHelpers:
    def test_read_bytes(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"data")
        assert read_bytes(path) == b"data"

    def test_read_bytes_missing(self, tmp_path):
        from rag_parser.core.interfaces import DocumentParseError

        with pytest.raises(DocumentParseError):
            read_bytes(tmp_path / "nope.bin")

    def test_decode_text(self):
        text, encoding = decode_text("привет".encode("utf-8"))
        assert text == "привет" and encoding == "utf-8"
        text, _ = decode_text("café".encode("latin-1"))
        assert "caf" in text

    def test_read_text_file(self, tmp_path):
        from rag_parser.core.interfaces import DocumentParseError

        path = tmp_path / "a.txt"
        path.write_text("hello", encoding="utf-8")
        text, encoding, size = read_text_file(path)
        assert text == "hello" and size == 5
        with pytest.raises(DocumentParseError):
            read_text_file(tmp_path / "missing.txt")
        empty = tmp_path / "empty.txt"
        empty.write_bytes(b"")
        with pytest.raises(DocumentParseError):
            read_text_file(empty)


class TestCells:
    def test_escape_none_bool_float(self):
        assert escape_cell(None) == ""
        assert escape_cell(True) == "true"
        assert escape_cell(False) == "false"
        assert escape_cell(3.0) == "3"

    def test_escape_pipes_newlines_truncation(self):
        assert escape_cell("a|b") == "a\\|b"
        assert escape_cell("a\nb\rc\rd") == "a b c d"
        long_text = "x" * 600
        assert escape_cell(long_text).endswith("…")
        assert len(escape_cell(long_text)) == 513

    def test_normalize_row(self):
        assert normalize_row(["a", None], width=3) == ["a", "", ""]
        assert normalize_row(["a", "b", "c"], width=2) == ["a", "b"]

    def test_rows_to_markdown(self):
        table, rows, width = rows_to_markdown_table([["A", "B"], ["1", "2"]])
        assert rows == 2 and width == 2
        assert table.splitlines()[0] == "| A | B |"
        assert rows_to_markdown_table([]) == ("", 0, 0)
        big = [[str(i)] for i in range(10)]
        table, rows, _ = rows_to_markdown_table(big, max_rows=3)
        assert rows == 3

    def test_build_result(self, tmp_path):
        path = tmp_path / "f.csv"
        path.write_text("a", encoding="utf-8")
        result = build_result(path, "csv", "| A |", {"rows": 1})
        assert result.filename == "f.csv"
        assert result.format == "csv"
        assert result.markdown == "| A |"


class TestDialect:
    def test_empty(self):
        assert detect_dialect("") == ","
        assert detect_dialect("   ") == ","

    def test_variants(self):
        assert detect_dialect("a;b\n1;2\n") == ";"
        assert detect_dialect("a\tb\n1\t2\n") == "\t"
        assert detect_dialect("a,b\n1,2\n") == ","

    def test_parse_rows(self):
        assert parse_csv_rows("a;b\n1;2\n", ";") == [["a", "b"], ["1", "2"]]
