import re
from pathlib import Path

import pytest

from rag_parser.core.interfaces import DocumentParseError
from rag_parser.parsers.csv_parser import CsvParser
from rag_parser.parsers.markitdown_parser import (
    SUPPORTED_EXTENSIONS,
    MarkItDownParser,
    get_converter,
)
from rag_parser.parsers.xlsx_parser import XlsxParser
from rag_parser.tests.conftest import CSV_SEMICOLON

MARKDOWN_TABLE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
MARKDOWN_SEPARATOR_RE = re.compile(r"^\|[\s\-|:]+\|$", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


def assert_valid_markdown(markdown: str) -> None:
    assert isinstance(markdown, str) and markdown.strip()
    assert "\x00" not in markdown
    assert "\f" not in markdown
    assert "\r" not in markdown
    assert not re.search(r"\n{3,}", markdown)
    for line in markdown.splitlines():
        if line.startswith("|"):
            assert line.count("|") >= 2, f"malformed table row: {line!r}"


def assert_metadata(result, required_keys):
    assert result.metadata, "metadata must not be empty"
    for key in required_keys:
        assert key in result.metadata, f"missing metadata key: {key}"


def assert_common(result, path: Path, expected_format: str):
    assert result.document_id
    assert result.filename == path.name
    assert result.format == expected_format
    assert_valid_markdown(result.markdown)


class TestMarkItDownParser:
    def test_supported_extensions(self):
        assert MarkItDownParser().supported_extensions() == SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS == [
            ".pdf",
            ".docx",
            ".html",
            ".htm",
            ".txt",
            ".md",
        ]

    def test_supported_extensions_excludes_archives_and_media(self):
        for extension in (".pptx", ".zip", ".mp3", ".mp4", ".doc", ".xlsm"):
            assert extension not in SUPPORTED_EXTENSIONS

    def test_xlsx_is_not_supported(self):
        assert ".xlsx" not in SUPPORTED_EXTENSIONS
        assert ".xls" not in SUPPORTED_EXTENSIONS

    def test_converter_is_reused(self):
        first = MarkItDownParser()
        second = MarkItDownParser()
        assert first.converter is second.converter
        assert get_converter() is first.converter

    def test_custom_converter_is_used(self):
        class FakeResult:
            text_content = "converted text"

        class FakeConverter:
            def __init__(self):
                self.calls = 0

            def convert(self, source):
                self.calls += 1
                return FakeResult()

        converter = FakeConverter()
        result = MarkItDownParser(converter=converter).parse(_make_file("x.txt"))
        assert converter.calls == 1
        assert result.markdown == "converted text"

    def test_txt(self, text_files):
        result = MarkItDownParser().parse(text_files["txt_utf8"])
        assert_common(result, text_files["txt_utf8"], "text")
        assert "Первая строка документа." in result.markdown
        assert_metadata(
            result, ["format", "extension", "size_bytes", "lines", "characters"]
        )
        assert result.metadata["converter"] == "markitdown"
        assert result.metadata["extension"] == ".txt"

    def test_markdown(self, text_files):
        result = MarkItDownParser().parse(text_files["md"])
        assert_common(result, text_files["md"], "markdown")
        assert HEADING_RE.search(result.markdown)

    def test_html(self, html_files):
        result = MarkItDownParser().parse(html_files["html_full"])
        assert_common(result, html_files["html_full"], "html")
        assert "Заголовок" in result.markdown
        assert "example.com" in result.markdown

    def test_docx(self, docx_files):
        result = MarkItDownParser().parse(docx_files["docx_headings"])
        assert_common(result, docx_files["docx_headings"], "docx")
        assert "Заголовок документа" in result.markdown
        assert "Первый абзац" in result.markdown
        assert HEADING_RE.search(result.markdown)

    def test_docx_table(self, docx_files):
        result = MarkItDownParser().parse(docx_files["docx_table"])
        assert_valid_markdown(result.markdown)
        assert MARKDOWN_TABLE_RE.search(result.markdown)

    def test_pdf_text(self, pdf_files):
        result = MarkItDownParser().parse(pdf_files["pdf_text"])
        assert_common(result, pdf_files["pdf_text"], "pdf")
        assert "First page of the document." in result.markdown
        assert "Second page body." in result.markdown

    def test_pdf_table_becomes_markdown_table(self, pdf_files):
        result = MarkItDownParser().parse(pdf_files["pdf_table"])
        assert "| Region" in result.markdown
        assert "North" in result.markdown and "120" in result.markdown
        assert_valid_markdown(result.markdown)

    def test_formfeeds_are_cleaned(self, pdf_files):
        result = MarkItDownParser().parse(pdf_files["pdf_text"])
        assert "\f" not in result.markdown

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DocumentParseError, match="not found"):
            MarkItDownParser().parse(tmp_path / "missing.pdf")

    def test_empty_file_raises(self, invalid_files):
        with pytest.raises(DocumentParseError, match="empty"):
            MarkItDownParser().parse(invalid_files["empty_pdf"])

    def test_empty_txt_raises(self, invalid_files):
        with pytest.raises(DocumentParseError, match="empty"):
            MarkItDownParser().parse(invalid_files["empty_txt"])

    def test_broken_pdf_raises(self, invalid_files):
        with pytest.raises(DocumentParseError):
            MarkItDownParser().parse(invalid_files["broken_pdf"])

    def test_fake_pdf_raises(self, invalid_files):
        with pytest.raises(DocumentParseError):
            MarkItDownParser().parse(invalid_files["not_a_pdf"])

    def test_broken_docx_raises(self, invalid_files):
        with pytest.raises(DocumentParseError):
            MarkItDownParser().parse(invalid_files["broken_docx"])

    def test_xlsx_rejected(self, xlsx_files):
        with pytest.raises(DocumentParseError, match="Unsupported extension"):
            MarkItDownParser().parse(xlsx_files["xlsx_multi"])

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "presentation.pptx"
        path.write_bytes(b"fake archive")
        with pytest.raises(DocumentParseError, match="Unsupported extension"):
            MarkItDownParser().parse(path)

    def test_zip_archive_raises(self, tmp_path):
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.txt", "text")
        path = tmp_path / "archive.zip"
        path.write_bytes(buffer.getvalue())
        with pytest.raises(DocumentParseError, match="Unsupported extension"):
            MarkItDownParser().parse(path)

    def test_conversion_failure_is_reported(self, monkeypatch):
        from markitdown._exceptions import FileConversionException

        class FailingConverter:
            def convert(self, source):
                raise FileConversionException(message="cannot convert")

        with pytest.raises(DocumentParseError, match="MarkItDown failed"):
            MarkItDownParser(converter=FailingConverter()).parse(_make_pdf_file())

    def test_unsupported_format_exception_is_reported(self):
        from markitdown._exceptions import UnsupportedFormatException

        class NoConverter:
            def convert(self, source):
                raise UnsupportedFormatException("no converter accepted the file")

        with pytest.raises(DocumentParseError, match="does not support"):
            MarkItDownParser(converter=NoConverter()).parse(_make_file("weird.txt"))

    def test_missing_dependency_is_reported(self, docx_files):
        from markitdown._exceptions import MissingDependencyException

        class MissingConverter:
            def convert(self, source):
                raise MissingDependencyException("optional dep missing")

        with pytest.raises(DocumentParseError, match="MarkItDown failed"):
            MarkItDownParser(converter=MissingConverter()).parse(
                docx_files["docx_headings"]
            )

    def test_unexpected_error_is_wrapped(self, text_files):
        class Boom:
            def convert(self, source):
                raise RuntimeError("unexpected")

        with pytest.raises(DocumentParseError, match="Failed to convert"):
            MarkItDownParser(converter=Boom()).parse(text_files["txt_utf8"])

    def test_empty_conversion_result_raises(self, text_files):
        class EmptyConverter:
            def convert(self, source):
                class Result:
                    text_content = "   \n\n"

                return Result()

        with pytest.raises(DocumentParseError, match="no text"):
            MarkItDownParser(converter=EmptyConverter()).parse(text_files["txt_utf8"])

    def test_none_text_content_is_rejected(self, text_files):
        class NoneConverter:
            def convert(self, source):
                class Result:
                    text_content = None

                return Result()

        with pytest.raises(DocumentParseError, match="no text"):
            MarkItDownParser(converter=NoneConverter()).parse(
                text_files["txt_utf8"]
            )

    def test_plaintext_fallback_is_rejected(self, invalid_files):
        with pytest.raises(DocumentParseError, match="not a valid document"):
            MarkItDownParser().parse(invalid_files["broken_pdf"])

    def test_unexpected_format_falls_back_is_rejected(self, tmp_path):
        path = tmp_path / "textual.docx"
        path.write_bytes(b"PK\x03\x04 pretend this is an office document")
        with pytest.raises(DocumentParseError):
            MarkItDownParser().parse(path)


class TestXlsxParser:
    def test_multi_sheet(self, xlsx_files):
        result = XlsxParser().parse(xlsx_files["xlsx_multi"])
        assert_common(result, xlsx_files["xlsx_multi"], "xlsx")
        assert "## Лист: Данные" in result.markdown
        assert "## Лист: Итоги" in result.markdown
        assert "Колонка1" in result.markdown
        assert MARKDOWN_TABLE_RE.search(result.markdown)
        assert result.metadata["sheets"] == 2
        assert result.metadata["parser"] == "calamine"

    def test_single_sheet(self, xlsx_files):
        result = XlsxParser().parse(xlsx_files["xlsx_single"])
        assert "Single" in result.markdown
        assert "row-0" in result.markdown
        assert_valid_markdown(result.markdown)

    def test_merged_cells_are_filled(self, xlsx_files):
        result = XlsxParser().parse(xlsx_files["xlsx_merged"])
        assert_valid_markdown(result.markdown)
        assert "## Лист: Data" in result.markdown
        # Merged title row A1:C1 is forward-filled across the range.
        title_rows = [
            line
            for line in result.markdown.splitlines()
            if "Report Title" in line
        ]
        assert title_rows
        assert title_rows[0].count("Report Title") == 3
        merged_rows = [
            line for line in result.markdown.splitlines() if "merged-row" in line
        ]
        assert merged_rows
        assert merged_rows[0].count("merged-row") == 3

    def test_supported_extensions(self):
        assert XlsxParser().supported_extensions() == [".xlsx", ".xls"]

    def test_broken_xlsx_raises(self, invalid_files):
        with pytest.raises(DocumentParseError):
            XlsxParser().parse(invalid_files["broken_xlsx"])

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        path.write_bytes(b"")
        with pytest.raises(DocumentParseError, match="empty"):
            XlsxParser().parse(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DocumentParseError, match="not found"):
            XlsxParser().parse(tmp_path / "missing.xlsx")

    def test_unsupported_extension_raises(self, text_files):
        with pytest.raises(DocumentParseError, match="Unsupported extension"):
            XlsxParser().parse(text_files["txt_utf8"])

    def test_empty_workbook_raises(self, tmp_path):
        from openpyxl import Workbook

        workbook = Workbook()
        path = tmp_path / "blank.xlsx"
        workbook.save(path)
        with pytest.raises(DocumentParseError):
            XlsxParser().parse(path)


def _make_file(name: str, payload: bytes = b"payload") -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False)
    handle.write(payload)
    handle.close()
    return Path(handle.name)


def _make_pdf_file(payload: bytes = b"%PDF-1.4\nfake body") -> Path:
    return _make_file("fake.pdf", payload)


class TestCsvParser:
    def test_semicolon_delimiter(self, csv_files):
        result = CsvParser().parse(csv_files["csv_semicolon"])
        assert_common(result, csv_files["csv_semicolon"], "csv")
        assert result.metadata["delimiter"] == ";"
        assert result.metadata["columns"] == 3
        assert result.metadata["rows"] == 2
        assert result.markdown.startswith("| Колонка1 | Колонка2 | Колонка3 |")
        assert "Значение1" in result.markdown
        assert CSV_SEMICOLON.splitlines()[0] == "Колонка1;Колонка2;Колонка3"

    def test_comma_delimiter(self, csv_files):
        result = CsvParser().parse(csv_files["csv_comma"])
        assert result.metadata["delimiter"] == ","
        assert result.metadata["columns"] == 3
        assert "banana" in result.markdown

    def test_single_column(self, tmp_path):
        path = tmp_path / "one.csv"
        path.write_text("only\nvalue\n", encoding="utf-8")
        result = CsvParser().parse(path)
        assert result.metadata["columns"] == 1
        assert_valid_markdown(result.markdown)

    def test_pipe_escaping(self, tmp_path):
        path = tmp_path / "pipes.csv"
        path.write_text("a|b,c\n1|2,3\n", encoding="utf-8")
        result = CsvParser().parse(path)
        assert "\\|" in result.markdown

    def test_empty_file_raises(self, invalid_files):
        with pytest.raises(DocumentParseError, match="empty"):
            CsvParser().parse(invalid_files["empty_csv"])

    def test_blank_file_raises(self, invalid_files):
        with pytest.raises(DocumentParseError, match="empty"):
            CsvParser().parse(invalid_files["blank_csv"])
